import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Tile sizes for the optimized kernel
#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__global__ void matmul_optimized_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C, 
                                         int N) {
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Number of threads per dimension
    const int numThreadsX = BN / TN; // 128/8 = 16
    const int numThreadsY = BM / TM; // 128/8 = 16
    
    // Shared memory for tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // Register file for thread results
    float threadResults[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];
    
    // Calculate global row and column indices for this thread's output tile
    const int threadRow = ty * TM;
    const int threadCol = tx * TN;
    
    // Global position of the block
    const int globalRowStart = by * BM;
    const int globalColStart = bx * BN;
    
    // Thread linear index for loading
    const int threadId = ty * numThreadsX + tx;
    const int numThreads = numThreadsX * numThreadsY; // 256
    
    // Number of tiles in K dimension
    const int numTilesK = (N + BK - 1) / BK;
    
    for (int tileK = 0; tileK < numTilesK; tileK++) {
        // Cooperatively load tile of A into shared memory
        // A tile is BM x BK = 128 x 16 = 2048 elements
        // With 256 threads, each thread loads 8 elements
        #pragma unroll
        for (int loadIdx = 0; loadIdx < (BM * BK) / numThreads; loadIdx++) {
            int elemIdx = threadId + loadIdx * numThreads;
            int loadRow = elemIdx / BK;
            int loadCol = elemIdx % BK;
            int globalRow = globalRowStart + loadRow;
            int globalCol = tileK * BK + loadCol;
            if (globalRow < N && globalCol < N) {
                As[loadRow][loadCol] = A[globalRow * N + globalCol];
            } else {
                As[loadRow][loadCol] = 0.0f;
            }
        }
        
        // Cooperatively load tile of B into shared memory
        // B tile is BK x BN = 16 x 128 = 2048 elements
        #pragma unroll
        for (int loadIdx = 0; loadIdx < (BK * BN) / numThreads; loadIdx++) {
            int elemIdx = threadId + loadIdx * numThreads;
            int loadRow = elemIdx / BN;
            int loadCol = elemIdx % BN;
            int globalRow = tileK * BK + loadRow;
            int globalCol = globalColStart + loadCol;
            if (globalRow < N && globalCol < N) {
                Bs[loadRow][loadCol] = B[globalRow * N + globalCol];
            } else {
                Bs[loadRow][loadCol] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A values into registers
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[threadRow + m][k];
            }
            
            // Load B values into registers
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][threadCol + n];
            }
            
            // Compute outer product
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    threadResults[m][n] += regA[m] * regB[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int globalRow = globalRowStart + threadRow + m;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int globalCol = globalColStart + threadCol + n;
            if (globalRow < N && globalCol < N) {
                C[globalRow * N + globalCol] = threadResults[m][n];
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
    
    int N = A.size(0);
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must have same dimensions");
    
    auto C = torch::zeros({N, N}, A.options());
    
    // Block dimensions: 16x16 threads per block
    dim3 blockDim(BN / TN, BM / TM);  // (16, 16)
    dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);
    
    matmul_optimized_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
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
    N = 2048 * 2
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
