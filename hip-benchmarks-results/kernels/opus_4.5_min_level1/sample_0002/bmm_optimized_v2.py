import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

bmm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Use larger tiles and multiple elements per thread
#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__global__ void batched_matmul_kernel_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    int batch_idx = blockIdx.z;
    
    // Thread block tile coordinates
    int bx = blockIdx.x;  // N dimension
    int by = blockIdx.y;  // M dimension
    
    // Thread indices within the tile
    int tx = threadIdx.x;  // 0-15 (handles TN=8 elements in N dimension)
    int ty = threadIdx.y;  // 0-15 (handles TM=8 elements in M dimension)
    
    // Shared memory
    __shared__ float As[BK][BM];  // Transposed for better access
    __shared__ float Bs[BK][BN];
    
    // Register storage for thread's portion of C
    float acc[TM][TN] = {0.0f};
    
    // Register storage for A and B fragments
    float a_reg[TM];
    float b_reg[TN];
    
    // Batch pointers
    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;
    
    // Global starting positions
    int row_start = by * BM + ty * TM;
    int col_start = bx * BN + tx * TN;
    
    // Number of threads for loading
    int num_threads = blockDim.x * blockDim.y;  // 256
    int tid = ty * blockDim.x + tx;
    
    // Loop over K dimension
    for (int k_block = 0; k_block < K; k_block += BK) {
        // Load A tile (BM x BK) into shared memory
        // Each thread loads multiple elements
        for (int i = tid; i < BM * BK; i += num_threads) {
            int sm_row = i % BM;
            int sm_col = i / BM;  // k dimension
            int g_row = by * BM + sm_row;
            int g_col = k_block + sm_col;
            if (g_row < M && g_col < K) {
                As[sm_col][sm_row] = A_batch[g_row * K + g_col];
            } else {
                As[sm_col][sm_row] = 0.0f;
            }
        }
        
        // Load B tile (BK x BN) into shared memory
        for (int i = tid; i < BK * BN; i += num_threads) {
            int sm_row = i / BN;  // k dimension
            int sm_col = i % BN;
            int g_row = k_block + sm_row;
            int g_col = bx * BN + sm_col;
            if (g_row < K && g_col < N) {
                Bs[sm_row][sm_col] = B_batch[g_row * N + g_col];
            } else {
                Bs[sm_row][sm_col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using register tiling
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A fragment into registers
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                a_reg[m] = As[k][ty * TM + m];
            }
            
            // Load B fragment into registers
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                b_reg[n] = Bs[k][tx * TN + n];
            }
            
            // Outer product
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    acc[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results to global memory
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int g_row = row_start + m;
        if (g_row < M) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int g_col = col_start + n;
                if (g_col < N) {
                    C_batch[g_row * N + g_col] = acc[m][n];
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
    
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    auto C = torch::empty({batch_size, M, N}, A.options());
    
    // 16x16 threads, each handling 8x8 elements = 128x128 tile
    dim3 block(16, 16);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM,
        batch_size
    );
    
    batched_matmul_kernel_v2<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        M,
        K,
        N
    );
    
    return C;
}
"""

bmm_cpp_source = """
torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

bmm_module = load_inline(
    name="bmm_hip_v2",
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
        return self.bmm_op.batched_matmul_hip(A.contiguous(), B.contiguous())


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
