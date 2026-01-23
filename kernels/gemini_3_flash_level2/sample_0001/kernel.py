
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

instance_norm_divide_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    static __shared__ T shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // The number of warps is blockDim.x / warpSize. For 1024 threads and warpSize 64, it's 16.
    T res = (threadIdx.x < (blockDim.x / (float)warpSize)) ? shared[lane] : (T)0.0;
    if (wid == 0) res = warp_reduce_sum(res);
    return res;
}

__global__ void instance_norm_divide_kernel_vec(
    float* __restrict__ input,
    int N, int C, int H, int W,
    float eps, float divide_by) {

    int nc = blockIdx.x;
    int hw_size = H * W;
    float* input_ptr = input + nc * hw_size;

    double sum = 0.0;
    double sum_sq = 0.0;

    int vec_size = hw_size / 4;
    float4* input_ptr4 = reinterpret_cast<float4*>(input_ptr);

    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val4 = input_ptr4[i];
        sum += (double)val4.x + (double)val4.y + (double)val4.z + (double)val4.w;
        sum_sq += (double)val4.x * (double)val4.x + (double)val4.y * (double)val4.y + 
                  (double)val4.z * (double)val4.z + (double)val4.w * (double)val4.w;
    }

    for (int i = vec_size * 4 + threadIdx.x; i < hw_size; i += blockDim.x) {
        float val = input_ptr[i];
        sum += (double)val;
        sum_sq += (double)val * (double)val;
    }

    double final_sum = block_reduce_sum(sum);
    double final_sum_sq = block_reduce_sum(sum_sq);

    __shared__ float mean_shared;
    __shared__ float inv_std_shared;

    if (threadIdx.x == 0) {
        float mean = (float)(final_sum / hw_size);
        float var = (float)((final_sum_sq / hw_size) - (double)mean * (double)mean);
        if (var < 0.0f) var = 0.0f;
        mean_shared = mean;
        inv_std_shared = 1.0f / (sqrtf(var + eps) * divide_by);
    }
    __syncthreads();

    float mean = mean_shared;
    float inv_std = inv_std_shared;

    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val4 = input_ptr4[i];
        val4.x = (val4.x - mean) * inv_std;
        val4.y = (val4.y - mean) * inv_std;
        val4.z = (val4.z - mean) * inv_std;
        val4.w = (val4.w - mean) * inv_std;
        input_ptr4[i] = val4;
    }

    for (int i = vec_size * 4 + threadIdx.x; i < hw_size; i += blockDim.x) {
        input_ptr[i] = (input_ptr[i] - mean) * inv_std;
    }
}

torch::Tensor instance_norm_divide_hip(torch::Tensor input, float eps, float divide_by) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    const int block_size = 512;
    int num_blocks = N * C;

    instance_norm_divide_kernel_vec<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), N, C, H, W, eps, divide_by);

    return input;
}
"""

instance_norm_divide_lib = load_inline(
    name="instance_norm_divide",
    cpp_sources=instance_norm_divide_source,
    functions=["instance_norm_divide_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = float(divide_by)
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        # Using F.conv2d would require weight and bias. Let's just use self.conv.
        # Ensure it's contiguous for our kernel.
        x = x.contiguous()
        return instance_norm_divide_lib.instance_norm_divide_hip(x, self.eps, self.divide_by)

def get_inputs():
    batch_size = 128
    in_channels  = 64  
    out_channels = 128  
    height = width = 128  
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    in_channels  = 64  
    out_channels = 128  
    kernel_size = 3
    divide_by = 2.0
    return [in_channels, out_channels, kernel_size, divide_by]
