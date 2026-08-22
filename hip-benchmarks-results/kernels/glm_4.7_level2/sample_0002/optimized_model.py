import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Simple fused kernel for matmul + Swish + bias
fused_matmul_swish_bias_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_matmul_swish_bias_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias_linear_ptr,
    const float* __restrict__ bias_external_ptr,
    float* __restrict__ C,
    int M,
    int N,
    int K,
    bool use_bias_linear,
    bool use_bias_external
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Matrix multiplication: C[row,col] = sum_k A[row,k] * B[col,k]
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        
        // Add linear layer bias (before Swish)
        if (use_bias_linear && bias_linear_ptr != nullptr) {
            sum += bias_linear_ptr[col];
        }
        
        // Swish activation
        float sigmoid_x = 1.0f / (1.0f + expf(-sum));
        float swish_out = sigmoid_x * sum;
        
        // Add external bias (after Swish)
        if (use_bias_external && bias_external_ptr != nullptr) {
            swish_out += bias_external_ptr[col];
        }
        
        C[row * N + col] = swish_out;
    }
}

torch::Tensor fused_matmul_swish_bias_hip(
    torch::Tensor A,
    torch::Tensor B,
    c10::optional<torch::Tensor> bias_linear_opt,
    c10::optional<torch::Tensor> bias_external_opt
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    const int blockDim_x = 16;
    const int blockDim_y = 16;
    
    dim3 blockDim(blockDim_x, blockDim_y);
    dim3 gridDim((N + blockDim_x - 1) / blockDim_x, (M + blockDim_y - 1) / blockDim_y);
    
    bool use_bias_linear = bias_linear_opt.has_value() && bias_linear_opt.value().numel() > 0;
    bool use_bias_external = bias_external_opt.has_value() && bias_external_opt.value().numel() > 0;
    
    const float* bias_linear_ptr = use_bias_linear ? bias_linear_opt.value().data_ptr<float>() : nullptr;
    const float* bias_external_ptr = use_bias_external ? bias_external_opt.value().data_ptr<float>() : nullptr;
    
    fused_matmul_swish_bias_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias_linear_ptr,
        bias_external_ptr,
        C.data_ptr<float>(),
        M,
        N,
        K,
        use_bias_linear,
        use_bias_external
    );
    
    return C;
}
"""

fused_matmul_swish_bias = load_inline(
    name="fused_matmul_swish_bias",
    cpp_sources=fused_matmul_swish_bias_cpp_source,
    functions=["fused_matmul_swish_bias_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused matmul + Swish + bias kernel.
    Replicates the exact computation of the reference model.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        
        # nn.Linear has its own weight and bias
        self.matmul = nn.Linear(in_features, out_features)
        
        # Custom fused kernel
        self.fused_matmul_swish_bias = fused_matmul_swish_bias
        
        # External bias parameter (same as reference)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # GroupNorm
        self.group_norm = nn.GroupNorm(num_groups, out_features)
    
    def forward(self, x):
        # Get linear layer weight
        weight = self.matmul.weight
        bias_linear = self.matmul.bias  # Can be None if bias=False
        
        # Apply fused matmul + add linear bias + swish + add external bias
        x = self.fused_matmul_swish_bias.fused_matmul_swish_bias_hip(x, weight, bias_linear, self.bias)
        
        # Apply GroupNorm
        x = self.group_norm(x)
        
        return x


def get_inputs():
    return [torch.rand(32768, 1024)]

def get_init_inputs():
    return [1024, 4096, 64, (4096,)]