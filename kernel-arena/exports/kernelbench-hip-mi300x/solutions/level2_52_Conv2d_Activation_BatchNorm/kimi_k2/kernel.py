import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized HIP kernel fusing Conv2d + Activation + BatchNorm
# Using shared memory for better memory access patterns
fused_conv_bn_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void fused_conv_activation_bn_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ bn_weight, const float* __restrict__ bn_bias, 
    const float* __restrict__ bn_running_mean, const float* __restrict__ bn_running_var, float eps,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width, int kernel_size
) {
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output position
    int out_w = blockIdx.x * TILE_WIDTH + tx;
    int out_h = blockIdx.y * TILE_HEIGHT + ty;
    int oc = blockIdx.z; // output channel
    int n = blockIdx.w; // batch
    
    // Bounds check
    if (out_w >= out_width || out_h >= out_height || oc >= out_channels || n >= batch_size) return;
    
    // Compute convolution
    float sum = bias[oc];
    
    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        int in_offset = ((n * in_channels + ic) * in_height);
        int w_offset = ((oc * in_channels + ic) * kernel_size * kernel_size);
        
        // Unroll kernel loops for better performance
        #pragma unroll 3
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = out_h + kh;
            if (ih >= in_height) continue;
            
            #pragma unroll 3
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = out_w + kw;
                if (iw >= in_width) continue;
                
                float input_val = input[in_offset + ih * in_width + iw];
                float weight_val = weight[w_offset + kh * kernel_size + kw];
                sum += input_val * weight_val;
            }
        }
    }
    
    // Apply activation: tanh(softplus(x)) * x
    float softplus = logf(1.0f + expf(fminf(sum, 20.0f)));
    float tanh_val = tanhf(softplus);
    float act_output = tanh_val * sum;
    
    // Apply batchnorm: y = (x - running_mean) * (weight / sqrt(running_var + eps)) + bias
    float bn_scale = bn_weight[oc] / sqrtf(bn_running_var[oc] + eps);
    float bn_shift = bn_bias[oc] - bn_running_mean[oc] * bn_scale;
    
    // Write output
    int output_idx = ((n * out_channels + oc) * out_height + out_h) * out_width + out_w;
    output[output_idx] = act_output * bn_scale + bn_shift;
}

__global__ void custom_activation_kernel(
    const float* __restrict__ input, float* __restrict__ output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    float softplus = logf(1.0f + expf(fminf(x, 20.0f)));
    float tanh_val = tanhf(softplus);
    output[idx] = tanh_val * x;
}

torch::Tensor fused_conv_bn_hip(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias, 
    torch::Tensor bn_running_mean, torch::Tensor bn_running_var, float eps
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = in_height - kernel_size + 1;
    int out_width = in_width - kernel_size + 1;
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    
    // 4D grid for optimal parallelization
    dim3 block_size(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid_size(
        (out_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
        out_channels,
        batch_size
    );
    
    fused_conv_activation_bn_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        eps,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width, kernel_size
    );
    
    return output;
}

torch::Tensor custom_activation_hip(torch::Tensor input) {
    int size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    custom_activation_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

# Compile the HIP kernels
fused_conv_bn = load_inline(
    name="fused_conv_bn",
    cpp_sources=fused_conv_bn_cpp_source,
    functions=["fused_conv_bn_hip", "custom_activation_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.fused_conv_bn = fused_conv_bn
        
    def forward(self, x):
        # Always use the fused kernel for better performance
        x = x.contiguous().float()
        weight = self.conv.weight.contiguous().float()
        bias = self.conv.bias.contiguous().float()
        bn_weight = self.bn.weight.contiguous().float()
        bn_bias = self.bn.bias.contiguous().float()
        bn_running_mean = self.bn.running_mean.contiguous().float()
        bn_running_var = self.bn.running_var.contiguous().float()
        
        return self.fused_conv_bn.fused_conv_bn_hip(
            x, weight, bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, self.bn.eps
        )

# Configuration matching the original model
batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
