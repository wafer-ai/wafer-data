import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused BatchNorm + Scaling kernel with better memory coalescing per channel
fused_bn_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Process one channel at a time for better memory coalescing
// Each block handles elements from a single (N, C) pair
__global__ void fused_bn_scale_channel_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    float scale,
    int N, int C, int HW
) {
    // Each block processes one channel of one batch element
    int nc_idx = blockIdx.x;
    int n = nc_idx / C;
    int c = nc_idx % C;
    
    if (n >= N) return;
    
    // Compute transform parameters for this channel
    float inv_std = rsqrtf(var[c] + eps);
    float w = gamma[c] * scale * inv_std;
    float b = (beta[c] - mean[c] * gamma[c] * inv_std) * scale;
    
    // Base offset for this (n, c) slice
    int base_offset = (n * C + c) * HW;
    
    // Process all HW elements for this channel
    for (int hw = threadIdx.x; hw < HW; hw += blockDim.x) {
        int idx = base_offset + hw;
        output[idx] = fmaf(input[idx], w, b);
    }
}

// Vectorized version that processes 4 spatial elements at once
__global__ void fused_bn_scale_channel_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    float scale,
    int N, int C, int HW
) {
    // Each block processes one channel of one batch element
    int nc_idx = blockIdx.x;
    int n = nc_idx / C;
    int c = nc_idx % C;
    
    if (n >= N) return;
    
    // Compute transform parameters for this channel
    float inv_std = rsqrtf(var[c] + eps);
    float w = gamma[c] * scale * inv_std;
    float b = (beta[c] - mean[c] * gamma[c] * inv_std) * scale;
    
    // Base offset for this (n, c) slice
    int base_offset = (n * C + c) * HW;
    
    // Number of vec4 elements
    int hw_vec4 = HW / 4;
    int hw_remainder = HW % 4;
    
    // Process in groups of 4
    const float4* in_ptr = reinterpret_cast<const float4*>(input + base_offset);
    float4* out_ptr = reinterpret_cast<float4*>(output + base_offset);
    
    for (int i = threadIdx.x; i < hw_vec4; i += blockDim.x) {
        float4 in_val = in_ptr[i];
        float4 out_val;
        out_val.x = fmaf(in_val.x, w, b);
        out_val.y = fmaf(in_val.y, w, b);
        out_val.z = fmaf(in_val.z, w, b);
        out_val.w = fmaf(in_val.w, w, b);
        out_ptr[i] = out_val;
    }
    
    // Handle remainder
    if (threadIdx.x < hw_remainder) {
        int idx = base_offset + hw_vec4 * 4 + threadIdx.x;
        output[idx] = fmaf(input[idx], w, b);
    }
}

torch::Tensor fused_bn_scale_inference(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    float scale
) {
    auto output = torch::empty_like(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;
    
    // Launch one block per (N, C) pair
    int num_blocks = N * C;
    const int block_size = 256;  // Threads per block
    
    if (HW % 4 == 0) {
        fused_bn_scale_channel_vec4_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            eps,
            scale,
            N, C, HW
        );
    } else {
        fused_bn_scale_channel_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            eps,
            scale,
            N, C, HW
        );
    }
    
    return output;
}
"""

fused_bn_scale_cpp = """
torch::Tensor fused_bn_scale_inference(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    float scale
);
"""

fused_bn_scale = load_inline(
    name="fused_bn_scale",
    cpp_sources=fused_bn_scale_cpp,
    cuda_sources=fused_bn_scale_source,
    functions=["fused_bn_scale_inference"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses BatchNorm + Scaling into a single kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_bn_scale = fused_bn_scale

    def forward(self, x):
        # Use PyTorch's optimized convolution
        x = self.conv(x)
        
        # Use fused BN + scaling kernel for inference
        if not self.training:
            x = self.fused_bn_scale.fused_bn_scale_inference(
                x.contiguous(),
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.scaling_factor
            )
        else:
            # Fall back to standard ops for training
            x = self.bn(x)
            x = x * self.scaling_factor
        
        return x


def get_inputs():
    return [torch.rand(128, 8, 128, 128).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0]
