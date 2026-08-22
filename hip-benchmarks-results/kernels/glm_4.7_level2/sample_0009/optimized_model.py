import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for BatchNorm + Scaling with vectorization
# Processes 4 elements per thread for better utilization
bn_scale_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void bn_scale_kernel(
    const float* __restrict__ input,      // [N, OC, H, W]
    const float* __restrict__ bn_weight,  // [OC] - gamma
    const float* __restrict__ bn_bias,    // [OC] - beta
    const float* __restrict__ bn_mean,    // [OC]
    const float* __restrict__ bn_var,     // [OC]
    float* __restrict__ output,           // [N, OC, H, W]
    int batch_size,
    int out_channels,
    int height,
    int width,
    float scaling_factor,
    float eps) {
    
    // Process 4 elements per thread
    const int vec_size = 4;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    const int total_elements = batch_size * out_channels * height * width;
    
    if (idx >= total_elements) return;
    
    // Load BN parameters once for all 4 elements
    int channel = (idx / (height * width)) % out_channels;
    const float gamma = bn_weight[channel];
    const float beta = bn_bias[channel];
    const float mean = bn_mean[channel];
    const float var = bn_var[channel];
    const float inv_std = rsqrtf(var + eps);
    const float combined_scale = gamma * inv_std * scaling_factor;
    const float combined_offset = (beta - gamma * mean * inv_std) * scaling_factor;
    
    // Process 4 elements if they're all valid
    for (int i = 0; i < vec_size && idx + i < total_elements; i++) {
        const float x = input[idx + i];
        // BatchNorm + Scaling: y = scale * x + offset
        output[idx + i] = combined_scale * x + combined_offset;
    }
}

torch::Tensor bn_scale_hip(
    torch::Tensor input,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float scaling_factor,
    float eps) {
    
    const int batch_size = input.size(0);
    const int out_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    const int total_elements = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + 3) / (block_size * 4) + 1;  // Divide by 4 for vectorization
    
    bn_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        height,
        width,
        scaling_factor,
        eps
    );
    
    return output;
}
"""

bn_scale = load_inline(
    name="bn_scale",
    cpp_sources=bn_scale_cpp_source,
    functions=["bn_scale_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused BatchNorm + Scaling kernel.
    Vectorized to process 4 elements per thread.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        # Create standard layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.epsilon = 1e-5
        
        # Get the fused BN + scaling kernel
        self.bn_scale = bn_scale

    def forward(self, x):
        # Run conv (using optimized ROCm implementation)
        x = self.conv(x)
        # Apply fused BatchNorm + Scaling
        x = self.bn_scale.bn_scale_hip(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.scaling_factor,
            self.epsilon
        )
        return x