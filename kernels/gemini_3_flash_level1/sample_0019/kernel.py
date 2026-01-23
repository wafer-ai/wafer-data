
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

instance_norm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WAVEFRONT_SIZE 64

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[WAVEFRONT_SIZE];
    int lane = threadIdx.x % WAVEFRONT_SIZE;
    int wid = threadIdx.x / WAVEFRONT_SIZE;

    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / WAVEFRONT_SIZE)) ? shared[lane] : 0.0f;
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down(val, offset);
        }
    }
    return val;
}

__global__ void __launch_bounds__(256) instance_norm_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int N, int C, int HW, float eps) 
{
    int nc = blockIdx.x;
    if (nc >= N * C) return;

    int offset = nc * HW;

    double sum = 0.0;
    double sum_sq = 0.0;

    int HW4 = HW / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input + offset);

    #pragma unroll 4
    for (int i = threadIdx.x; i < HW4; i += blockDim.x) {
        float4 val4 = input4[i];
        sum += (double)val4.x + (double)val4.y + (double)val4.z + (double)val4.w;
        sum_sq += (double)val4.x * (double)val4.x + (double)val4.y * (double)val4.y + 
                  (double)val4.z * (double)val4.z + (double)val4.w * (double)val4.w;
    }
    
    for (int i = HW4 * 4 + threadIdx.x; i < HW; i += blockDim.x) {
        float val = input[offset + i];
        sum += (double)val;
        sum_sq += (double)val * (double)val;
    }

    float final_sum = blockReduceSum((float)sum);
    float final_sum_sq = blockReduceSum((float)sum_sq);

    __shared__ float shared_mean;
    __shared__ float shared_inv_std;

    if (threadIdx.x == 0) {
        float mean = final_sum / HW;
        float var = (final_sum_sq / HW) - (mean * mean);
        shared_mean = mean;
        shared_inv_std = rsqrtf(max(0.0f, var) + eps);
    }
    __syncthreads();

    float mean = shared_mean;
    float inv_std = shared_inv_std;

    float4* output4 = reinterpret_cast<float4*>(output + offset);
    #pragma unroll 4
    for (int i = threadIdx.x; i < HW4; i += blockDim.x) {
        float4 val4 = input4[i];
        float4 res4;
        res4.x = (val4.x - mean) * inv_std;
        res4.y = (val4.y - mean) * inv_std;
        res4.z = (val4.z - mean) * inv_std;
        res4.w = (val4.w - mean) * inv_std;
        output4[i] = res4;
    }

    for (int i = HW4 * 4 + threadIdx.x; i < HW; i += blockDim.x) {
        output[offset + i] = (input[offset + i] - mean) * inv_std;
    }
}

torch::Tensor instance_norm_hip(torch::Tensor x, float eps) {
    int N = x.size(0);
    int C = x.size(1);
    int HW = x.size(2) * x.size(3);
    auto output = torch::empty_like(x);
    instance_norm_kernel<<<N * C, 256>>>(x.data_ptr<float>(), output.data_ptr<float>(), N, C, HW, eps);
    return output;
}
"""

instance_norm_lib = load_inline(
    name="instance_norm_lib",
    cpp_sources=instance_norm_cpp_source,
    functions=["instance_norm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_lib.instance_norm_hip(x, self.eps)

