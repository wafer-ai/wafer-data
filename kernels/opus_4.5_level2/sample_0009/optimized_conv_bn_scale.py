import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused BatchNorm + Scaling kernel
fused_bn_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Kernel for fused BatchNorm inference + Scaling
// BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Scaled: y = ((x - mean) / sqrt(var + eps) * gamma + beta) * scale
// Combined: y = x * (gamma * scale / sqrt(var + eps)) + (beta * scale - mean * gamma * scale / sqrt(var + eps))
__global__ void fused_bn_scale_inference_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,  // BN weight
    const float* __restrict__ beta,   // BN bias
    const float* __restrict__ mean,   // Running mean
    const float* __restrict__ var,    // Running var
    float eps,
    float scale,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (idx < total) {
        // Calculate channel index
        int hw = H * W;
        int c = (idx / hw) % C;
        
        // Precompute multiplier and addend for this channel
        float inv_std = rsqrtf(var[c] + eps);
        float w = gamma[c] * scale * inv_std;
        float b = (beta[c] - mean[c] * gamma[c] * inv_std) * scale;
        
        output[idx] = input[idx] * w + b;
    }
}

// Optimized version with vectorized loads for better memory throughput
__global__ void fused_bn_scale_inference_kernel_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    float scale,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    int hw = H * W;
    
    // Process 4 elements at a time
    int idx4 = idx * 4;
    
    if (idx4 + 3 < total) {
        // Load 4 values
        float4 in_val = *reinterpret_cast<const float4*>(input + idx4);
        float4 out_val;
        
        // Process each element
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int elem_idx = idx4 + i;
            int c = (elem_idx / hw) % C;
            
            float inv_std = rsqrtf(var[c] + eps);
            float w = gamma[c] * scale * inv_std;
            float b = (beta[c] - mean[c] * gamma[c] * inv_std) * scale;
            
            float val = (i == 0) ? in_val.x : (i == 1) ? in_val.y : (i == 2) ? in_val.z : in_val.w;
            float result = val * w + b;
            
            if (i == 0) out_val.x = result;
            else if (i == 1) out_val.y = result;
            else if (i == 2) out_val.z = result;
            else out_val.w = result;
        }
        
        *reinterpret_cast<float4*>(output + idx4) = out_val;
    } else if (idx4 < total) {
        // Handle remaining elements
        for (int i = 0; i < 4 && idx4 + i < total; i++) {
            int elem_idx = idx4 + i;
            int c = (elem_idx / hw) % C;
            
            float inv_std = rsqrtf(var[c] + eps);
            float w = gamma[c] * scale * inv_std;
            float b = (beta[c] - mean[c] * gamma[c] * inv_std) * scale;
            
            output[elem_idx] = input[elem_idx] * w + b;
        }
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
    int total = N * C * H * W;
    
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_bn_scale_inference_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        scale,
        N, C, H, W
    );
    
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
    extra_cuda_cflags=["-O3"]
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
