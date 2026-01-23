import torch
import torch.nn as nn
import os
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Custom HIP kernel for fused matmul + bias + scaling + residual add
matmul_scaling_residual_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void matmul_scaling_residual_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M,
    int N,
    int K,
    float scaling_factor) {
    
    // Matrix dimensions:
    // A: M x K (input)
    // B: N x K (weight from nn.Linear, stored as out_features x in_features)
    // bias: N (bias vector)
    // C: M x N (output)
    // Operation: A @ B.T + bias
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product: A[row, :] @ B[col, :]
        // A[row * K + k] * B[col * K + k] where B is (N, K)
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        
        // Add bias
        sum += bias[col];
        
        // Apply scaling and add residual (original matmul+bias result)
        // result = sum * scaling_factor + sum = sum * (1.0 + scaling_factor)
        C[row * N + col] = sum * (1.0f + scaling_factor);
    }
}

torch::Tensor matmul_scaling_residual_hip(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias,
    float scaling_factor) {
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block_size(32, 32);
    dim3 num_blocks((N + block_size.x - 1) / block_size.x,
                    (M + block_size.y - 1) / block_size.y);
    
    hipLaunchKernelGGL(
        matmul_scaling_residual_kernel,
        num_blocks,
        block_size,
        0,
        0,
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K,
        scaling_factor
    );
    
    return C;
}
"""

matmul_scaling_residual = load_inline(
    name="matmul_scaling_residual",
    cpp_sources=matmul_scaling_residual_cpp_source,
    functions=["matmul_scaling_residual_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # nn.Linear stores weight as (out_features, in_features) = (N, K)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = scaling_factor
        self.matmul_scaling_residual = matmul_scaling_residual
        
    def forward(self, x):
        # Fused matmul + bias + scaling + residual add
        # x @ weight.T + bias, then scaling and residual add
        return self.matmul_scaling_residual.matmul_scaling_residual_hip(
            x, self.weight, self.bias, self.scaling_factor
        )

def get_inputs():
    batch_size = 16384
    in_features = 4096
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 4096
    out_features = 4096
    scaling_factor = 0.5
    return [in_features, out_features, scaling_factor]