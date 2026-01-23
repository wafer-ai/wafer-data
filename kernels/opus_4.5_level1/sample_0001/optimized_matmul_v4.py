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
#define BK 8

// Thread tile dimensions (per thread)
#define TM 8
#define TN 8

// Number of threads per block dimension
#define NUM_THREADS_X (BN / TN)  // 16
#define NUM_THREADS_Y (BM / TM)  // 16

__global__ __launch_bounds__(256) void matmul_optimized_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int K, int N) {
    
    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ float As[BK][BM + 4];  // A is loaded transposed
    __shared__ float Bs[BK][BN + 4];
    
    // Thread coordinates within the block
    const int tx = threadIdx.x;  // 0..15 (BN/TN = 128/8 = 16)
    const int ty = threadIdx.y;  // 0..15 (BM/TM = 128/8 = 16)
    const int threadId = ty * blockDim.x + tx;
    
    // Block coordinates
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Starting row and column for this block
    const int rowStart = by * BM;
    const int colStart = bx * BN;
    
    // Register tile for accumulation
    float regC[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            regC[i][j] = 0.0f;
        }
    }
    
    // Register tiles for A and B
    float regA[TM];
    float regB[TN];
    
    // Total threads per block
    const int totalThreads = NUM_THREADS_X * NUM_THREADS_Y;  // 256
    
    // Elements to load per thread for A tile: BM * BK / 256 = 128 * 8 / 256 = 4
    // Elements to load per thread for B tile: BK * BN / 256 = 8 * 128 / 256 = 4
    
    // Precompute load indices for A
    const int loadIdxA = threadId;
    const int loadRowA = loadIdxA % BM;  // Which row of A (0-127)
    const int loadColA = loadIdxA / BM;  // Which col of A (0-3 for first iteration)
    
    // Precompute load indices for B
    const int loadIdxB = threadId;
    const int loadRowB = loadIdxB / BN;  // Which row of B (0-1)
    const int loadColB = loadIdxB % BN;  // Which col of B (0-127)
    
    // Iterate over tiles along K dimension
    for (int k = 0; k < K; k += BK) {
        // Load A tile into shared memory (transposed storage)
        // Each thread loads BM*BK/256 = 4 elements
        #pragma unroll
        for (int loadIter = 0; loadIter < (BM * BK) / totalThreads; loadIter++) {
            int idx = threadId + loadIter * totalThreads;
            int row = idx % BM;
            int col = idx / BM;
            int globalRow = rowStart + row;
            int globalCol = k + col;
            float val = 0.0f;
            if (globalRow < M && globalCol < K) {
                val = A[globalRow * K + globalCol];
            }
            As[col][row] = val;
        }
        
        // Load B tile into shared memory
        #pragma unroll
        for (int loadIter = 0; loadIter < (BK * BN) / totalThreads; loadIter++) {
            int idx = threadId + loadIter * totalThreads;
            int row = idx / BN;
            int col = idx % BN;
            int globalRow = k + row;
            int globalCol = colStart + col;
            float val = 0.0f;
            if (globalRow < K && globalCol < N) {
                val = B[globalRow * N + globalCol];
            }
            Bs[row][col] = val;
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load A registers from shared memory
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[kk][ty * TM + m];
            }
            
            // Load B registers from shared memory
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[kk][tx * TN + n];
            }
            
            // Compute outer product
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
    
    auto C = torch::empty({M, N}, A.options());
    
    // Block dim: (BN/TN, BM/TM) = (16, 16) = 256 threads
    dim3 blockDim(NUM_THREADS_X, NUM_THREADS_Y);
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
    name="matmul_hip_v4",
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
