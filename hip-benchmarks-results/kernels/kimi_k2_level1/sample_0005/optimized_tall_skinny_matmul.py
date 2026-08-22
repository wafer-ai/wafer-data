import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/types.h>

// For tall-skinny (M=32768, K=32, N=32768)
// Use a simple but highly parallel kernel where each thread computes one output element
__global__ void tall_skinny_matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N) {
    
    // Each thread computes one element of C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= M * N) return;
    
    int row = idx / N;
    int col = idx % N;
    
    float sum = 0.0f;
    
    // Compute dot product - manually unrolled for K=32
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[idx] = sum;
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Use large grid to maximize parallelism
    // MI300X has 304 CUs, use many blocks per CU
    int total_elements = M * N;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Limit num_blocks to avoid overhead but still maximize parallelism
    num_blocks = min(num_blocks, 304 * 8);
    
    tall_skinny_matmul_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N);
    
    return C;
}
"""

tall_skinny_matmul = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=matmul_cpp_source,
    functions=["matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = tall_skinny_matmul
    
    def forward(self, A, B):
        return self.matmul.matmul_hip(A.cuda(), B.cuda())

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []
