import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 8

// Padding to reduce bank conflicts
#define PAD 4

__global__ void __launch_bounds__(128) skinny_matmul_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tid = ty * 8 + tx;

    int row_start = by * BM;
    int col_start = bx * BN;

    // Pad inner dimensions
    __shared__ float As[BM][BK + PAD];
    __shared__ float Bs[BK][BN + PAD];

    const float4* A_ptr = reinterpret_cast<const float4*>(A);
    const float4* B_ptr = reinterpret_cast<const float4*>(B);

    // Load A into Shared
    int load_idx_a = tid;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = load_idx_a + i * 128;
        int r = idx >> 3; // div 8
        int c = idx & 7;  // mod 8
        
        int global_r = row_start + r;
        
        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (global_r < M) {
             val = A_ptr[global_r * 8 + c]; 
        }
        
        // As[r][c*4]. Stride is BK+PAD.
        // We cast row pointer.
        float4* As_row_ptr = reinterpret_cast<float4*>(&As[r][0]);
        // NOTE: reinterpret_cast on 2D array with padding might be tricky if not careful about row start.
        // As[r] gives pointer to row start.
        // This is safe.
        As_row_ptr[c] = val;
    }

    // Load B into Shared
    int load_idx_b = tid;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = load_idx_b + i * 128;
        int r = idx >> 4; // div 16
        int c = idx & 15; // mod 16
        
        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < BK) {
             int global_c_vec = (col_start >> 2) + c;
             if (global_c_vec < (N >> 2)) {
                 val = B_ptr[r * (N >> 2) + global_c_vec];
             }
        }
        
        float4* Bs_row_ptr = reinterpret_cast<float4*>(&Bs[r][0]);
        Bs_row_ptr[c] = val;
    }

    __syncthreads();

    // Compute
    float acc[TM][TN];
    #pragma unroll
    for(int i=0; i<TM; ++i)
        for(int j=0; j<TN; ++j)
            acc[i][j] = 0.0f;
            
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        float a_col[TM];
        #pragma unroll
        for(int i=0; i<TM; ++i) {
            a_col[i] = As[ty * TM + i][k];
        }
        
        float b_row[TN];
        // Load TN=8 floats = 2 float4s
        // Bs[k][tx * TN]. Stride BN+PAD.
        const float4* B_row_ptr = reinterpret_cast<const float4*>(&Bs[k][tx * TN]);
        float4 v1 = B_row_ptr[0];
        float4 v2 = B_row_ptr[1];
        
        b_row[0] = v1.x; b_row[1] = v1.y; b_row[2] = v1.z; b_row[3] = v1.w;
        b_row[4] = v2.x; b_row[5] = v2.y; b_row[6] = v2.z; b_row[7] = v2.w;
        
        #pragma unroll
        for(int i=0; i<TM; ++i) {
            #pragma unroll
            for(int j=0; j<TN; ++j) {
                acc[i][j] += a_col[i] * b_row[j];
            }
        }
    }
    
    // Store C
    int global_row_base = row_start + ty * TM;
    int global_c_vec_base = (col_start >> 2) + tx * 2;
    
    float4* C_ptr_f4 = reinterpret_cast<float4*>(C);
    
    #pragma unroll
    for(int i=0; i<TM; ++i) {
        int r = global_row_base + i;
        if (r < M) {
             if (global_c_vec_base + 1 < (N >> 2)) {
                 float4 v1, v2;
                 v1.x = acc[i][0]; v1.y = acc[i][1]; v1.z = acc[i][2]; v1.w = acc[i][3];
                 v2.x = acc[i][4]; v2.y = acc[i][5]; v2.z = acc[i][6]; v2.w = acc[i][7];
                 
                 int idx1 = r * (N >> 2) + global_c_vec_base;
                 C_ptr_f4[idx1] = v1;
                 C_ptr_f4[idx1 + 1] = v2;
             }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    // Block: 8x16 = 128 threads
    dim3 block(8, 16);
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    
    skinny_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    
    return C;
}
"""

module = load_inline(
    name="skinny_matmul_v5",
    cpp_sources=cpp_source,
    functions=["matmul_hip"],
    extra_cflags=['-O3'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.module = module
    
    def forward(self, A, B):
        return self.module.matmul_hip(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []
