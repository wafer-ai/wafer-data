import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized matmul for MI300X with register blocking
// Each thread computes a TM x TN block of output
#define BM 128      // Block size for M dimension
#define BN 128      // Block size for N dimension  
#define BK 16       // Block size for K dimension
#define TM 8        // Thread tile M
#define TN 8        // Thread tile N

__global__ void matmul_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Shared memory for tiles
    __shared__ float As[BK][BM];  // Transposed for coalesced access
    __shared__ float Bs[BK][BN];
    
    // Thread position within the block
    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15
    
    // Block position
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Starting position for this block
    int blockRowStart = by * BM;
    int blockColStart = bx * BN;
    
    // Thread ID within the block
    int threadId = ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y;  // 256 threads
    
    // Each thread computes TM x TN elements
    // We have 128/8 = 16 threads in each dimension, so 16x16 = 256 threads
    int threadRowInBlock = (threadId / (BN / TN)) * TM;
    int threadColInBlock = (threadId % (BN / TN)) * TN;
    
    // Register arrays for accumulation
    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];
    
    // Loop over K tiles
    for (int k = 0; k < K; k += BK) {
        // Load A tile into shared memory (with transpose)
        // A is M x K, we need BM x BK elements
        // Each thread loads (BM * BK) / 256 = 128 * 16 / 256 = 8 elements
        for (int i = threadId; i < BM * BK; i += numThreads) {
            int localRow = i / BK;
            int localCol = i % BK;
            int globalRow = blockRowStart + localRow;
            int globalCol = k + localCol;
            
            if (globalRow < M && globalCol < K) {
                As[localCol][localRow] = A[globalRow * K + globalCol];
            } else {
                As[localCol][localRow] = 0.0f;
            }
        }
        
        // Load B tile into shared memory
        // B is K x N, we need BK x BN elements
        for (int i = threadId; i < BK * BN; i += numThreads) {
            int localRow = i / BN;
            int localCol = i % BN;
            int globalRow = k + localRow;
            int globalCol = blockColStart + localCol;
            
            if (globalRow < K && globalCol < N) {
                Bs[localRow][localCol] = B[globalRow * N + globalCol];
            } else {
                Bs[localRow][localCol] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load A elements into registers
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                regA[tm] = As[kk][threadRowInBlock + tm];
            }
            
            // Load B elements into registers
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                regB[tn] = Bs[kk][threadColInBlock + tn];
            }
            
            // Compute outer product
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    regC[tm][tn] += regA[tm] * regB[tn];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        int globalRow = blockRowStart + threadRowInBlock + tm;
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int globalCol = blockColStart + threadColInBlock + tn;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = regC[tm][tn];
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Grid: one block per BM x BN output tile
    dim3 block(16, 16);  // 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    matmul_optimized_kernel<<<grid, block>>>(
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
