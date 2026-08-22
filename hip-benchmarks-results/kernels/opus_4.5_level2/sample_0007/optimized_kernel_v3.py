import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Bias + Swish + Scaling kernel
# This combines the bias addition from Linear with swish and scaling
fused_bias_swish_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused kernel: output[i] = (input[i] + bias[i % out_features]) * sigmoid(input[i] + bias[i % out_features]) * scaling_factor
__global__ void fused_bias_swish_scale_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    const float scaling_factor,
    const int batch_size,
    const int out_features) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * out_features;
    const int stride = gridDim.x * blockDim.x;
    
    // Process elements with grid-stride loop
    for (int idx = tid; idx < total; idx += stride) {
        // Get bias index (column index)
        int bias_idx = idx % out_features;
        
        // Load input and add bias
        float val = input[idx] + bias[bias_idx];
        
        // Compute swish: val * sigmoid(val) * scaling_factor
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        output[idx] = val * sigmoid_val * scaling_factor;
    }
}

// Vectorized version for when out_features is divisible by 4
__global__ void fused_bias_swish_scale_kernel_vec4(
    const float* __restrict__ input, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    const float scaling_factor,
    const int batch_size,
    const int out_features) 
{
    const int total_vec = (batch_size * out_features) / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    for (int vec_idx = tid; vec_idx < total_vec; vec_idx += stride) {
        int idx = vec_idx * 4;
        int bias_idx = idx % out_features;
        
        // Load 4 input elements
        float4 in_val = *reinterpret_cast<const float4*>(input + idx);
        float4 bias_val = *reinterpret_cast<const float4*>(bias + bias_idx);
        float4 out_val;
        
        // Add bias and compute swish for each
        float val0 = in_val.x + bias_val.x;
        float val1 = in_val.y + bias_val.y;
        float val2 = in_val.z + bias_val.z;
        float val3 = in_val.w + bias_val.w;
        
        float sigmoid0 = 1.0f / (1.0f + expf(-val0));
        float sigmoid1 = 1.0f / (1.0f + expf(-val1));
        float sigmoid2 = 1.0f / (1.0f + expf(-val2));
        float sigmoid3 = 1.0f / (1.0f + expf(-val3));
        
        out_val.x = val0 * sigmoid0 * scaling_factor;
        out_val.y = val1 * sigmoid1 * scaling_factor;
        out_val.z = val2 * sigmoid2 * scaling_factor;
        out_val.w = val3 * sigmoid3 * scaling_factor;
        
        // Store result
        *reinterpret_cast<float4*>(output + idx) = out_val;
    }
}

torch::Tensor fused_bias_swish_scale_hip(
    torch::Tensor input, 
    torch::Tensor bias,
    float scaling_factor) 
{
    const int batch_size = input.size(0);
    const int out_features = input.size(1);
    auto output = torch::empty_like(input);
    
    const int total = batch_size * out_features;
    const int block_size = 256;
    
    // Use vectorized kernel if aligned
    if (out_features % 4 == 0) {
        const int num_blocks = min(65535, (total / 4 + block_size - 1) / block_size);
        fused_bias_swish_scale_kernel_vec4<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), 
            bias.data_ptr<float>(),
            output.data_ptr<float>(), 
            scaling_factor,
            batch_size,
            out_features
        );
    } else {
        const int num_blocks = min(65535, (total + block_size - 1) / block_size);
        fused_bias_swish_scale_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), 
            bias.data_ptr<float>(),
            output.data_ptr<float>(), 
            scaling_factor,
            batch_size,
            out_features
        );
    }
    
    return output;
}
"""

fused_bias_swish_scale_cpp = """
torch::Tensor fused_bias_swish_scale_hip(torch::Tensor input, torch::Tensor bias, float scaling_factor);
"""

fused_module = load_inline(
    name="fused_bias_swish_scale",
    cpp_sources=fused_bias_swish_scale_cpp,
    cuda_sources=fused_bias_swish_scale_source,
    functions=["fused_bias_swish_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused bias + Swish activation + scaling kernel.
    Uses matmul without bias, then fuses bias addition with swish and scaling.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep original linear layer to get weights and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        self.fused_module = fused_module
        
        # Initialize with same distribution as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1 / (in_features ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Matrix multiplication without bias
        x = F.linear(x, self.weight, bias=None)
        # Fused bias + swish + scale
        x = self.fused_module.fused_bias_swish_scale_hip(x, self.bias, self.scaling_factor)
        return x
