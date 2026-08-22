import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Block tile dimensions
#define BM 128
#define BN 128
#define BK 16

// Thread tile dimensions (each thread computes TM x TN elements)
#define TM 8
#define TN 8

__global__ void matmul_kernel_v2(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N) {
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Number of threads per block dimension
    const int numThreadsX = BN / TN;  // 128/8 = 16
    const int numThreadsY = BM / TM;  // 128/8 = 16
    
    // Linear thread index
    int tid = ty * numThreadsX + tx;
    
    // Shared memory for tiles
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    
    // Register array for accumulating results
    float regC[TM][TN] = {0.0f};
    
    // Register arrays for loading A and B tiles
    float regA[TM];
    float regB[TN];
    
    // Starting positions
    int rowC = by * BM + ty * TM;
    int colC = bx * BN + tx * TN;
    
    // Load parameters
    const int numLoadThreads = numThreadsX * numThreadsY;  // 256
    const int loadRowsA = BM * BK / numLoadThreads;  // Each thread loads this many elements from A
    const int loadRowsB = BK * BN / numLoadThreads;  // Each thread loads this many elements from B
    
    // Number of tiles
    int numTiles = (N + BK - 1) / BK;
    
    for (int t = 0; t < numTiles; t++) {
        // Collaborative loading of A tile (BM x BK) into shared memory
        // A is (N x N), we load a (BM x BK) tile
        for (int i = 0; i < loadRowsA; i++) {
            int loadIdx = tid + i * numLoadThreads;
            int loadRow = loadIdx / BK;
            int loadCol = loadIdx % BK;
            int globalRow = by * BM + loadRow;
            int globalCol = t * BK + loadCol;
            
            if (globalRow < N && globalCol < N) {
                As[loadCol][loadRow] = A[globalRow * N + globalCol];
            } else {
                As[loadCol][loadRow] = 0.0f;
            }
        }
        
        // Collaborative loading of B tile (BK x BN) into shared memory
        for (int i = 0; i < loadRowsB; i++) {
            int loadIdx = tid + i * numLoadThreads;
            int loadRow = loadIdx / BN;
            int loadCol = loadIdx % BN;
            int globalRow = t * BK + loadRow;
            int globalCol = bx * BN + loadCol;
            
            if (globalRow < N && globalCol < N) {
                Bs[loadRow][loadCol] = B[globalRow * N + globalCol];
            } else {
                Bs[loadRow][loadCol] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute the partial results
        for (int k = 0; k < BK; k++) {
            // Load from shared memory to registers
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[k][ty * TM + m];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][tx * TN + n];
            }
            
            // Outer product
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
    
    // Store results to global memory
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int globalRow = rowC + m;
            int globalCol = colC + n;
            if (globalRow < N && globalCol < N) {
                C[globalRow * N + globalCol] = regC[m][n];
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    dim3 block(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    
    matmul_kernel_v2<<<grid, block>>>(
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
    extra_cuda_cflags=["-O3"],
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
