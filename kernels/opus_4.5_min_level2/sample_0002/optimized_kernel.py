import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for Swish activation + bias addition
fused_swish_bias_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

__global__ void fused_swish_bias_kernel(const float* __restrict__ input, 
                                         const float* __restrict__ bias,
                                         float* __restrict__ output,
                                         int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        int feature_idx = idx % out_features;
        float x = input[idx];
        // Swish: sigmoid(x) * x
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float swish = sigmoid_x * x;
        // Add bias
        output[idx] = swish + bias[feature_idx];
    }
}

torch::Tensor fused_swish_bias_hip(torch::Tensor input, torch::Tensor bias) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    int total_size = batch_size * out_features;
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_swish_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_features);
    
    return output;
}
"""

fused_swish_bias = load_inline(
    name="fused_swish_bias",
    cpp_sources=fused_swish_bias_cpp_source,
    functions=["fused_swish_bias_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Swish + bias kernel.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.fused_swish_bias = fused_swish_bias

    def forward(self, x):
        x = self.matmul(x)
        # Fused Swish + bias addition
        x = self.fused_swish_bias.fused_swish_bias_hip(x, self.bias)
        x = self.group_norm(x)
        return x


def get_inputs():
    return [torch.rand(32768, 1024).cuda()]


def get_init_inputs():
    return [1024, 4096, 64, (4096,)]
