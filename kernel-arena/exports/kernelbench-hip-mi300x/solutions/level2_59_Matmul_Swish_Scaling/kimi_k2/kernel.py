import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused HIP kernel for GEMM + Swish + Scaling
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256

__global__ void fused_gemm_swish_scale_kernel(
    const float* __restrict__ A,      // input: [M, K]
    const float* __restrict__ B,      // weight: [out_features, in_features]
    const float* __restrict__ bias,   // bias: [out_features]
    float* __restrict__ C,            // output: [M, N]
    int M, int N, int K, float scaling_factor) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;  // row in output
    int col = idx % N;  // col in output
    
    if (row >= M) return;
    
    float value = 0.0f;
    
    // Compute: x @ weight.t() + bias
    // where A is [M, K], B is [N, K], we want: output[row][col] = sum(A[row][k] * B[col][k])
    const float* a_row = A + row * K;
    const float* b_row = B + col * K;
    
    for (int k = 0; k < K; k++) {
        value += a_row[k] * b_row[k];
    }
    
    // Add bias
    value += bias[col];
    
    // Apply Swish activation: x * sigmoid(x)
    float sigmoid = 1.0f / (1.0f + expf(-value));
    value = value * sigmoid * scaling_factor;
    
    C[row * N + col] = value;
}

torch::Tensor fused_gemm_swish_scale_hip(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias,
    float scaling_factor) {
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
    
    int num_threads_per_block = BLOCK_SIZE;
    int num_blocks = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    fused_gemm_swish_scale_kernel<<<num_blocks, num_threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K, scaling_factor);
    
    return C;
}
"""

# Compile the kernel
fused_kernel = load_inline(
    name="fused_gemm_swish_scale",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_gemm_swish_scale_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        
        # Initialize weight as [out_features, in_features] (same as nn.Linear)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        self.fused_kernel = fused_kernel
        
    def forward(self, x):
        # In nn.Linear: x @ weight.t() + bias
        # So we pass weight.t() to compute dot product with each row
        return self.fused_kernel.fused_gemm_swish_scale_hip(
            x, self.weight.t().contiguous(), self.bias, self.scaling_factor
        )

# Input generation functions
def get_inputs():
    batch_size = 128
    in_features = 32768
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 32768
    out_features = 32768
    scaling_factor = 2.0
    return [in_features, out_features, scaling_factor]