import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized tiled matmul for AMD MI300X
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized for MI300X with larger tiles and vectorized loads
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

__global__ __launch_bounds__(256) void matmul_kernel_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;
    
    __shared__ float As[BK][BM + 1];  // +1 to avoid bank conflicts
    __shared__ float Bs[BK][BN + 1];
    
    // Registers for the thread's computation
    float regC[TM][TN] = {{0.0f}};
    float regA[TM];
    float regB[TN];
    
    // Base positions
    const int baseM = by * BM;
    const int baseN = bx * BN;
    
    // Thread positions within the block for computing
    const int threadM = ty * TM;
    const int threadN = tx * TN;
    
    // Number of tiles
    const int numTiles = (N + BK - 1) / BK;
    
    for (int tile = 0; tile < numTiles; tile++) {
        // Collaborative loading of A (BM x BK) and B (BK x BN)
        // Each thread loads multiple elements
        const int tileK = tile * BK;
        
        // Load A tile
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = threadIdx.x + i * 256;
            int loadM = idx / BK;
            int loadK = idx % BK;
            int globalM = baseM + loadM;
            int globalK = tileK + loadK;
            
            As[loadK][loadM] = (globalM < N && globalK < N) ? A[globalM * N + globalK] : 0.0f;
        }
        
        // Load B tile
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = threadIdx.x + i * 256;
            int loadK = idx / BN;
            int loadN = idx % BN;
            int globalK = tileK + loadK;
            int globalN = baseN + loadN;
            
            Bs[loadK][loadN] = (globalK < N && globalN < N) ? B[globalK * N + globalN] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A values
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[k][threadM + m];
            }
            
            // Load B values
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][threadN + n];
            }
            
            // Outer product
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    regC[m][n] = __fmaf_rn(regA[m], regB[n], regC[m][n]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int globalM = baseM + threadM + m;
            int globalN = baseN + threadN + n;
            if (globalM < N && globalN < N) {
                C[globalM * N + globalN] = regC[m][n];
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    dim3 block(256);  // 16*16 = 256 threads
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    
    matmul_kernel_opt<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N);
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B);
"""

matmul_module = load_inline(
    name="matmul_hip_v5",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_hip_source,
    functions=["matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
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
