import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Tile sizes
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 16
#define THREAD_SIZE_M 4
#define THREAD_SIZE_N 4

__global__ __launch_bounds__(256) void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {
    
    // Thread block computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of C
    // Each thread computes a THREAD_SIZE_M x THREAD_SIZE_N subtile
    
    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];  // Transposed for coalesced access
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15
    
    // Thread ID for loading
    int threadId = ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y;  // 256
    
    // Starting positions
    int rowStart = by * BLOCK_SIZE_M;
    int colStart = bx * BLOCK_SIZE_N;
    
    // Register accumulation
    float accum[THREAD_SIZE_M][THREAD_SIZE_N] = {{0.0f}};
    
    // Loop over K dimension
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        // Load A tile: BLOCK_SIZE_M x BLOCK_SIZE_K = 64 x 16 = 1024 elements
        // 256 threads, so 4 elements per thread
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = threadId + i * numThreads;
            int row = idx % BLOCK_SIZE_M;
            int col = idx / BLOCK_SIZE_M;
            int gRow = rowStart + row;
            int gCol = k + col;
            float val = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
            As[col][row] = val;
        }
        
        // Load B tile: BLOCK_SIZE_K x BLOCK_SIZE_N = 16 x 64 = 1024 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = threadId + i * numThreads;
            int row = idx / BLOCK_SIZE_N;
            int col = idx % BLOCK_SIZE_N;
            int gRow = k + row;
            int gCol = colStart + col;
            float val = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
            Bs[row][col] = val;
        }
        
        __syncthreads();
        
        // Compute: each thread handles a 4x4 subtile
        // Thread (tx, ty) handles rows [ty*4, ty*4+4) and cols [tx*4, tx*4+4)
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE_K; kk++) {
            float a[THREAD_SIZE_M];
            float b[THREAD_SIZE_N];
            
            #pragma unroll
            for (int m = 0; m < THREAD_SIZE_M; m++) {
                a[m] = As[kk][ty * THREAD_SIZE_M + m];
            }
            
            #pragma unroll
            for (int n = 0; n < THREAD_SIZE_N; n++) {
                b[n] = Bs[kk][tx * THREAD_SIZE_N + n];
            }
            
            #pragma unroll
            for (int m = 0; m < THREAD_SIZE_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_SIZE_N; n++) {
                    accum[m][n] = __fmaf_rn(a[m], b[n], accum[m][n]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int m = 0; m < THREAD_SIZE_M; m++) {
        int gRow = rowStart + ty * THREAD_SIZE_M + m;
        if (gRow < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_SIZE_N; n++) {
                int gCol = colStart + tx * THREAD_SIZE_N + n;
                if (gCol < N) {
                    C[gRow * N + gCol] = accum[m][n];
                }
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Must be float32");
    
    A = A.contiguous();
    B = B.contiguous();
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 blockDim(16, 16);  // 256 threads
    dim3 gridDim((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, 
                 (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    
    matmul_kernel<<<gridDim, blockDim>>>(
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
    name="matmul_hip_v5",
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
