import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel for Swish activation + bias addition with vectorized loads
fused_swish_bias_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

// Vectorized kernel using float4 for coalesced memory access
__global__ void fused_swish_bias_kernel_vec4(const float4* __restrict__ input, 
                                              const float4* __restrict__ bias,
                                              float4* __restrict__ output,
                                              int batch_size, int out_features_div4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size_div4 = batch_size * out_features_div4;
    
    if (idx < total_size_div4) {
        int feature_idx = idx % out_features_div4;
        float4 x = input[idx];
        float4 b = bias[feature_idx];
        
        // Swish: sigmoid(x) * x + bias for each component
        float4 result;
        result.x = (1.0f / (1.0f + expf(-x.x))) * x.x + b.x;
        result.y = (1.0f / (1.0f + expf(-x.y))) * x.y + b.y;
        result.z = (1.0f / (1.0f + expf(-x.z))) * x.z + b.z;
        result.w = (1.0f / (1.0f + expf(-x.w))) * x.w + b.w;
        
        output[idx] = result;
    }
}

// Scalar fallback kernel
__global__ void fused_swish_bias_kernel(const float* __restrict__ input, 
                                         const float* __restrict__ bias,
                                         float* __restrict__ output,
                                         int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        int feature_idx = idx % out_features;
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = sigmoid_x * x + bias[feature_idx];
    }
}

torch::Tensor fused_swish_bias_hip(torch::Tensor input, torch::Tensor bias) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    int total_size = batch_size * out_features;
    
    auto output = torch::empty_like(input);
    
    // Use vectorized kernel if out_features is divisible by 4
    if (out_features % 4 == 0) {
        int out_features_div4 = out_features / 4;
        int total_size_div4 = batch_size * out_features_div4;
        
        const int block_size = 256;
        const int num_blocks = (total_size_div4 + block_size - 1) / block_size;
        
        fused_swish_bias_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<const float4*>(bias.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            batch_size, out_features_div4);
    } else {
        const int block_size = 256;
        const int num_blocks = (total_size + block_size - 1) / block_size;
        
        fused_swish_bias_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, out_features);
    }
    
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
    Optimized model with fused Swish + bias kernel using vectorization.
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
