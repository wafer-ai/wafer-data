import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>

// Row-wise approach with better memory coalescing
// Each block handles one row and multiple columns
__global__ void tall_skinny_matmul_rowwise(
    const float* A, const float* B, float* C,
    int M, int K, int N) {
    
    int row = blockIdx.x;
    int col_base = blockIdx.y * blockDim.x;
    
    float sum;
    
    if (row < M) {
        sum = 0.0f;
        
        // Load row of A into registers
        float a_vals[32];
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            a_vals[k] = A[row * 32 + k];
        }
        
        // Compute for this thread's column
        int col = col_base + threadIdx.x;
        
        // Compute dot product
        for (int k = 0; k < K; k++) {
            sum += a_vals[k] * B[k * N + col];
        }
        
        // Write result
        if (col < N) {
            C[row * N + col] = sum;
        }
    }
}

// Alternative approach: compute multiple columns per thread
__global__ void tall_skinny_matmul_vec4(
    const float* A, const float* B, float* C,
    int M, int K, int N) {
    
    int row = blockIdx.x;
    int col_base = blockIdx.y * blockDim.x * 4 + threadIdx.x * 4;
    
    float4 sums[8];  // Each thread computes 32 columns
    sums[0].x = sums[0].y = sums[0].z = sums[0].w = 0.0f;
    sums[1].x = sums[1].y = sums[1].z = sums[1].w = 0.0f;
    sums[2].x = sums[2].y = sums[2].z = sums[2].w = 0.0f;
    sums[3].x = sums[3].y = sums[3].z = sums[3].w = 0.0f;
    sums[4].x = sums[4].y = sums[4].z = sums[4].w = 0.0f;
    sums[5].x = sums[5].y = sums[5].z = sums[5].w = 0.0f;
    sums[6].x = sums[6].y = sums[6].z = sums[6].w = 0.0f;
    sums[7].x = sums[7].y = sums[7].z = sums[7].w = 0.0f;
    
    if (row < M) {
        // Load all 32 elements of A row into registers
        float a[32];
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            a[k] = A[row * 32 + k];
        }
        
        // Compute for all columns
        for (int k = 0; k < 32; k++) {
            float a_val = a[k];
            const float* b_col = &B[k * N + col_base];
            
            // Process 32 columns in 8 vector4 loads
            #pragma unroll
            for (int i = 0; i < 8 && col_base + i * 4 + 3 < N; i++) {
                float4 b = reinterpret_cast<const float4*>(B)[(k * N + col_base + i * 4) / 4];
                sums[i].x += a_val * b.x;
                sums[i].y += a_val * b.y;
                sums[i].z += a_val * b.z;
                sums[i].w += a_val * b.w;
            }
        }
        
        // Write results as vectorized stores
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = col_base + i * 4;
            if (col + 3 < N) {
                reinterpret_cast<float4*>(C)[(row * N + col) / 4] = sums[i];
            }
        }
    }
}

torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Use vectorized approach
    const int THREADS_PER_BLOCK = 64;  // Each thread computes 4 columns
    
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(M, (N + THREADS_PER_BLOCK * 4 - 1) / (THREADS_PER_BLOCK * 4));
    
    tall_skinny_matmul_vec4<<<gridDim, blockDim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N);
    
    return C;
}
"""

matmul = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=matmul_cpp_source,
    functions=["tall_skinny_matmul_hip"],
    verbose=True,
)


def custom_kernel(inputs):
    A, B = inputs
    A = A.cuda()
    B = B.cuda()
    C = matmul.tall_skinny_matmul_hip(A, B)
    return C


class ModelNew(nn.Module):
    """
    Optimized model with custom HIP kernel for tall and skinny matrix multiplication
    using vectorized loads and stores
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul
    
    def forward(self, A, B):
        return self.matmul.tall_skinny_matmul_hip(A, B)


M = 16384 * 2
N = 16 * 2


def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]


def get_init_inputs():
    return []