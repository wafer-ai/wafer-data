import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

instnorm_divide_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void instnorm_divide_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    float divide_by,
    float eps) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    bool is_valid = (x < width && y < height);
    int num_pixels = height * width;
    
    // Process each channel
    for (int c = 0; c < channels; c++) {
        int pixel_idx = y * width + x;
        int input_idx = ((batch_idx * channels + c) * height + y) * width + x;
        
        float val = is_valid ? input[input_idx] : 0.0f;
        
        // Compute sum using warp reduction
        float sum = val;
        sum = __shfl_down(sum, 16); sum = __shfl(sum, 0) + (tid % 32 < 16 ? sum : 0);
        sum = __shfl(sum, 0);
        
        // Better approach: use simple atomic in global memory
        __shared__ float s_sum[256];
        __shared__ float s_sq[256];
        
        s_sum[tid] = is_valid ? val : 0.0f;
        s_sq[tid] = is_valid ? val * val : 0.0f;
        
        __syncthreads();
        
        // Block reduction
        for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum[tid] += s_sum[tid + s];
                s_sq[tid] += s_sq[tid + s];
            }
            __syncthreads();
        }
        
        float mean = s_sum[0] / num_pixels;
        float variance = s_sq[0] / num_pixels - mean * mean;
        variance = fmaxf(variance, 0.0f);
        float std = sqrtf(variance + eps);
        
        // Normalize and divide
        if (is_valid) {
            float norm_val = (val - mean) / std;
            norm_val /= divide_by;
            output[input_idx] = norm_val;
        }
        __syncthreads();
    }
}

torch::Tensor instnorm_divide_hip(
    torch::Tensor input,
    float divide_by) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    float eps = 1e-5f;
    
    auto output = torch::zeros_like(input);
    
    const int threads_x = 16;
    const int threads_y = 16;
    int num_blocks_x = (width + threads_x - 1) / threads_x;
    int num_blocks_y = (height + threads_y - 1) / threads_y;
    
    dim3 blocks(num_blocks_x, num_blocks_y, batch_size);
    dim3 threads(threads_x, threads_y);
    
    instnorm_divide_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        divide_by,
        eps);
    
    return output;
}
"""

instnorm_divide_module = load_inline(
    name="instnorm_divide_mod_v2",
    cpp_sources=instnorm_divide_cpp_source,
    functions=["instnorm_divide_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using fused InstanceNorm + Division kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divide_by = divide_by
        
        # Keep original Conv2d (highly optimized in PyTorch)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Custom fused InstanceNorm + Division kernel
        self.instnorm_divide = instnorm_divide_module
        
    def forward(self, x):
        x = self.conv(x)
        x = self.instnorm_divide.instnorm_divide_hip(x, self.divide_by)
        return x