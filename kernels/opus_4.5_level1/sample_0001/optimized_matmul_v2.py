import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Tile dimensions for shared memory
#define BM 128
#define BN 128
#define BK 16

// Thread tile dimensions (per thread)
#define TM 8
#define TN 8

__global__ void matmul_optimized_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C,
                                         int M, int K, int N) {
    // Shared memory tiles
    __shared__ float As[BK][BM];  // Transposed for better access
    __shared__ float Bs[BK][BN];
    
    // Thread coordinates within the block
    int tx = threadIdx.x;  // 0..15 (BN/TN = 128/8 = 16)
    int ty = threadIdx.y;  // 0..15 (BM/TM = 128/8 = 16)
    
    // Block coordinates
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Starting row and column for this block
    int rowStart = by * BM;
    int colStart = bx * BN;
    
    // Register tile for accumulation
    float regC[TM][TN] = {0.0f};
    
    // Register tiles for A and B
    float regA[TM];
    float regB[TN];
    
    // Number of threads
    int numThreads = blockDim.x * blockDim.y;
    int threadId = ty * blockDim.x + tx;
    
    // Load iterations
    int loadRowsA = (BM * BK + numThreads - 1) / numThreads;
    int loadRowsB = (BK * BN + numThreads - 1) / numThreads;
    
    // Iterate over tiles along K dimension
    for (int k = 0; k < K; k += BK) {
        // Load A tile into shared memory (transposed)
        for (int i = 0; i < loadRowsA; i++) {
            int idx = threadId + i * numThreads;
            if (idx < BM * BK) {
                int loadRow = idx / BK;  // row in tile
                int loadCol = idx % BK;  // col in tile
                int globalRow = rowStart + loadRow;
                int globalCol = k + loadCol;
                if (globalRow < M && globalCol < K) {
                    As[loadCol][loadRow] = A[globalRow * K + globalCol];
                } else {
                    As[loadCol][loadRow] = 0.0f;
                }
            }
        }
        
        // Load B tile into shared memory
        for (int i = 0; i < loadRowsB; i++) {
            int idx = threadId + i * numThreads;
            if (idx < BK * BN) {
                int loadRow = idx / BN;  // row in tile
                int loadCol = idx % BN;  // col in tile
                int globalRow = k + loadRow;
                int globalCol = colStart + loadCol;
                if (globalRow < K && globalCol < N) {
                    Bs[loadRow][loadCol] = B[globalRow * N + globalCol];
                } else {
                    Bs[loadRow][loadCol] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load A registers
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[kk][ty * TM + m];
            }
            
            // Load B registers
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[kk][tx * TN + n];
            }
            
            // Compute outer product
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int globalRow = rowStart + ty * TM + m;
        if (globalRow < M) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int globalCol = colStart + tx * TN + n;
                if (globalCol < N) {
                    C[globalRow * N + globalCol] = regC[m][n];
                }
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    
    // Ensure contiguous
    A = A.contiguous();
    B = B.contiguous();
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Block dim: (BN/TN, BM/TM) = (16, 16) = 256 threads
    dim3 blockDim(BN/TN, BM/TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    matmul_optimized_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B);
"""

matmul_module = load_inline(
    name="matmul_hip",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_hip_source,
    functions=["matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_hip(A, B)


def get_inputs():
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
