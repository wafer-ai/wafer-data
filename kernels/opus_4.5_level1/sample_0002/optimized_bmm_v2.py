import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

bmm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized tiled GEMM with register blocking
// Each thread computes a TM x TN tile of the output
// Block computes BM x BN tile using BK strip of shared memory

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__global__ void batched_matmul_kernel_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size, int M, int K, int N)
{
    int batch = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Threads per block: (BN/TN) x (BM/TM) = 16 x 16 = 256
    const int threads_x = BN / TN;  // 16
    const int threads_y = BM / TM;  // 16
    
    // Flatten thread index for loading
    int tid = ty * threads_x + tx;
    
    // Shared memory for A and B tiles
    __shared__ float As[BK][BM];  // Transposed for coalesced access
    __shared__ float Bs[BK][BN];
    
    // Pointers to batch data
    const float* A_batch = A + batch * M * K;
    const float* B_batch = B + batch * K * N;
    float* C_batch = C + batch * M * N;
    
    // Starting positions for this block
    int row_start = by * BM;
    int col_start = bx * BN;
    
    // Register array to accumulate results
    float acc[TM][TN] = {0.0f};
    
    // Register arrays for A and B tiles loaded by this thread
    float a_reg[TM];
    float b_reg[TN];
    
    // Number of K tiles
    int num_k_tiles = (K + BK - 1) / BK;
    
    // Total threads = 256
    // Load As: BK * BM = 16 * 128 = 2048 elements, each thread loads 8
    // Load Bs: BK * BN = 16 * 128 = 2048 elements, each thread loads 8
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k_start = k_tile * BK;
        
        // Load A tile into shared memory (BM x BK -> stored as BK x BM transposed)
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < (BK * BM) / 256; i++) {
            int idx = tid + i * 256;
            int load_k = idx / BM;  // which row in BK
            int load_m = idx % BM;  // which col in BM
            int global_m = row_start + load_m;
            int global_k = k_start + load_k;
            
            if (global_m < M && global_k < K) {
                As[load_k][load_m] = A_batch[global_m * K + global_k];
            } else {
                As[load_k][load_m] = 0.0f;
            }
        }
        
        // Load B tile into shared memory (BK x BN)
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int load_k = idx / BN;
            int load_n = idx % BN;
            int global_k = k_start + load_k;
            int global_n = col_start + load_n;
            
            if (global_k < K && global_n < N) {
                Bs[load_k][load_n] = B_batch[global_k * N + global_n];
            } else {
                Bs[load_k][load_n] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load A registers (TM elements from column kk of As)
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                a_reg[tm] = As[kk][ty * TM + tm];
            }
            
            // Load B registers (TN elements from row kk of Bs)
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                b_reg[tn] = Bs[kk][tx * TN + tn];
            }
            
            // Outer product
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    acc[tm][tn] += a_reg[tm] * b_reg[tn];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        int global_row = row_start + ty * TM + tm;
        if (global_row < M) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                int global_col = col_start + tx * TN + tn;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = acc[tm][tn];
                }
            }
        }
    }
}

torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match");
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    auto C = torch::empty({batch_size, M, N}, A.options());
    
    // Block: 16 x 16 threads (256 total)
    // Each block computes BM x BN = 128 x 128 output tile
    dim3 block(BN / TN, BM / TM);  // (16, 16)
    dim3 grid((N + BN - 1) / BN, 
              (M + BM - 1) / BM, 
              batch_size);
    
    batched_matmul_kernel_v2<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );
    
    return C;
}
"""

bmm_cpp_source = """
torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

bmm_module = load_inline(
    name="batched_matmul_v2",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_hip_source,
    functions=["batched_matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_op = bmm_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_op.batched_matmul_hip(A, B)


def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]


def get_init_inputs():
    return []
