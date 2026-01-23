import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused BatchNorm + Scaling kernel with maximum throughput optimization
fused_bn_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Highly optimized kernel using float4 and maximizing memory throughput
// Uses grid-stride loop for better occupancy
__global__ void fused_bn_scale_vec4_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    int C, int HW, int total_vec4
) {
    // Cache channel params in shared memory
    extern __shared__ float shared_params[];
    float* s_weights = shared_params;
    float* s_biases = shared_params + C;
    
    // Cooperatively load channel parameters
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        s_weights[c] = weights[c];
        s_biases[c] = biases[c];
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_vec4; i += stride) {
        float4 in_val = input[i];
        float4 out_val;
        
        int base_idx = i * 4;
        
        // Compute channel indices
        int c0 = (base_idx / HW) % C;
        int c1 = ((base_idx + 1) / HW) % C;
        int c2 = ((base_idx + 2) / HW) % C;
        int c3 = ((base_idx + 3) / HW) % C;
        
        // Use FMA for better precision and performance
        out_val.x = fmaf(in_val.x, s_weights[c0], s_biases[c0]);
        out_val.y = fmaf(in_val.y, s_weights[c1], s_biases[c1]);
        out_val.z = fmaf(in_val.z, s_weights[c2], s_biases[c2]);
        out_val.w = fmaf(in_val.w, s_weights[c3], s_biases[c3]);
        
        output[i] = out_val;
    }
}

// Scalar fallback for non-aligned sizes
__global__ void fused_bn_scale_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    int C, int HW, int total
) {
    extern __shared__ float shared_params[];
    float* s_weights = shared_params;
    float* s_biases = shared_params + C;
    
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        s_weights[c] = weights[c];
        s_biases[c] = biases[c];
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total; i += stride) {
        int c = (i / HW) % C;
        output[i] = fmaf(input[i], s_weights[c], s_biases[c]);
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
    
    // Precompute BN parameters on GPU
    auto inv_std = torch::rsqrt(running_var + eps);
    auto weights = gamma * scale * inv_std;
    auto biases = (beta - running_mean * gamma * inv_std) * scale;
    
    const int block_size = 512;  // Larger block size for better occupancy
    int shared_mem_size = 2 * C * sizeof(float);
    
    // Use vectorized kernel
    if (total % 4 == 0) {
        int total_vec4 = total / 4;
        // Use enough blocks to saturate the GPU but not too many
        int num_blocks = min((total_vec4 + block_size - 1) / block_size, 2048);
        
        fused_bn_scale_vec4_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            weights.data_ptr<float>(),
            biases.data_ptr<float>(),
            C, HW, total_vec4
        );
    } else {
        int num_blocks = min((total + block_size - 1) / block_size, 2048);
        
        fused_bn_scale_scalar_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            weights.data_ptr<float>(),
            biases.data_ptr<float>(),
            C, HW, total
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
