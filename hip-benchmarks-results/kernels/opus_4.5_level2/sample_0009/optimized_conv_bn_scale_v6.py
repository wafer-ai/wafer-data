import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused BatchNorm + Scaling kernel optimized for MI300X
fused_bn_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Highly optimized kernel that each warp processes contiguous memory
// Using larger thread blocks and processing multiple elements per thread
__global__ __launch_bounds__(1024, 2)
void fused_bn_scale_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    float scale,
    int N, int C, int HW, int total
) {
    // Use LDS (Local Data Share) for channel parameters
    __shared__ float s_w[128];  // Max 128 channels
    __shared__ float s_b[128];
    
    // Precompute channel params cooperatively
    if (threadIdx.x < C) {
        float inv_std = rsqrtf(var[threadIdx.x] + eps);
        s_w[threadIdx.x] = gamma[threadIdx.x] * scale * inv_std;
        s_b[threadIdx.x] = (beta[threadIdx.x] - mean[threadIdx.x] * gamma[threadIdx.x] * inv_std) * scale;
    }
    __syncthreads();
    
    // Grid-stride loop with unrolling
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per iteration when possible
    int total4 = total - 3;
    for (int i = idx; i < total4; i += stride * 4) {
        #pragma unroll 4
        for (int j = 0; j < 4; j++) {
            int curr_idx = i + j * stride;
            if (curr_idx < total) {
                int c = (curr_idx / HW) % C;
                output[curr_idx] = fmaf(input[curr_idx], s_w[c], s_b[c]);
            }
        }
    }
    
    // Handle tail
    int remaining = idx + ((total4 / stride) * 4) * stride;
    for (int i = remaining; i < total; i += stride) {
        int c = (i / HW) % C;
        output[i] = fmaf(input[i], s_w[c], s_b[c]);
    }
}

// Alternative: Direct vectorized version
__global__ void fused_bn_scale_vec4_direct_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    float scale,
    int C, int HW, int total_vec4
) {
    __shared__ float s_w[128];
    __shared__ float s_b[128];
    
    // Precompute channel params
    if (threadIdx.x < C) {
        float inv_std = rsqrtf(var[threadIdx.x] + eps);
        s_w[threadIdx.x] = gamma[threadIdx.x] * scale * inv_std;
        s_b[threadIdx.x] = (beta[threadIdx.x] - mean[threadIdx.x] * gamma[threadIdx.x] * inv_std) * scale;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_vec4) {
        float4 in_val = input[idx];
        
        int base = idx * 4;
        int c0 = (base / HW) % C;
        int c1 = ((base + 1) / HW) % C;
        int c2 = ((base + 2) / HW) % C;
        int c3 = ((base + 3) / HW) % C;
        
        float4 out_val;
        out_val.x = fmaf(in_val.x, s_w[c0], s_b[c0]);
        out_val.y = fmaf(in_val.y, s_w[c1], s_b[c1]);
        out_val.z = fmaf(in_val.z, s_w[c2], s_b[c2]);
        out_val.w = fmaf(in_val.w, s_w[c3], s_b[c3]);
        
        output[idx] = out_val;
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
    int total = N * C * HW;
    
    // Use vectorized kernel for better memory throughput
    if (total % 4 == 0 && C <= 128) {
        int total_vec4 = total / 4;
        const int block_size = 256;
        int num_blocks = (total_vec4 + block_size - 1) / block_size;
        
        fused_bn_scale_vec4_direct_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            eps,
            scale,
            C, HW, total_vec4
        );
    } else {
        const int block_size = 1024;
        int num_blocks = (total + block_size - 1) / block_size;
        num_blocks = min(num_blocks, 65535);
        
        fused_bn_scale_optimized_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            eps,
            scale,
            N, C, HW, total
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
