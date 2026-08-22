import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: tanh(softplus(x)) * x + batchnorm
# Combines all element-wise operations into a single kernel
# Uses batch statistics for training mode
fused_activation_bn_cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 16
#define MAX_BATCH_SIZE 64
#define MAX_CHANNELS 256

__device__ __forceinline__ float softplus(float x) {
    return logf(1.0f + expf(x));
}

__global__ void fused_activation_bn_kernel(
    const float* __restrict__ input,
    float* __restrict__ out,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    int n, int c, int h, int w)  // n=batch, c=channels, h=height, w=width
{
    int batch = blockIdx.z;
    int ch = blockIdx.y;
    
    int y = blockIdx.x * TILE_SIZE + threadIdx.y;
    int x_coord = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (y < h && x_coord < w) {
        int idx = batch * c * h * w + ch * h * w + y * w + x_coord;
        
        float val = input[idx];
        // Softplus: log(1 + exp(x))
        float sp = softplus(val);
        // Tanh: tanh(val)
        float th = tanhf(sp);
        // Element-wise multiply
        float activated = th * val;
        
        // BatchNorm: (x - mean) * invstd * gamma + beta
        float m = mean[ch];
        float inv = invstd[ch];
        float gm = gamma[ch];
        float bt = beta[ch];
        
        out[idx] = gm * (activated - m) * inv + bt;
    }
}

torch::Tensor fused_activation_bn_hip(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor invstd) {
    
    int n = input.size(0);  // batch
    int c = input.size(1);  // channels
    int h = input.size(2);  // height
    int w = input.size(3);  // width
    
    auto out = torch::empty_like(input);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((w + TILE_SIZE - 1) / TILE_SIZE, c, n);
    
    fused_activation_bn_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        n, c, h, w
    );
    
    return out;
}
"""

fused_activation_bn = load_inline(
    name="fused_activation_bn",
    cpp_sources=fused_activation_bn_cpp_source,
    functions=["fused_activation_bn_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep conv2d as is (already optimized)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Use PyTorch.BatchNorm2d for storing running stats and computing batch stats
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.fused_activation_bn = fused_activation_bn
    
    def forward(self, x):
        x = self.conv(x)
        # Apply activation using PyTorch ops (easier to manage)
        act_x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
        # Compute batch statistics (training mode)
        mean = act_x.mean(dim=(0, 2, 3))
        var = act_x.var(dim=(0, 2, 3), unbiased=False)
        invstd = 1.0 / torch.sqrt(var + self.bn.eps)
        # Use fused kernel for batchnorm with pre-computed batch stats
        x = self.fused_activation_bn.fused_activation_bn_hip(
            act_x, 
            self.bn.weight, 
            self.bn.bias, 
            mean, 
            invstd
        )
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]