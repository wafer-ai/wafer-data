import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set hipcc as the compiler
os.environ["CXX"] = "hipcc"

softmax_source = """
#include <hip/hip_runtime.h>

#define WARP_SIZE 64

__device__ __forceinline__ void warp_reduce(float& max_val, float& sum_val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down(max_val, offset);
        float other_sum = __shfl_down(sum_val, offset);
        
        float new_max = fmaxf(max_val, other_max);
        
        if (new_max == -INFINITY) {
            sum_val = 0.0f;
        } else {
            sum_val = sum_val * expf(max_val - new_max) + other_sum * expf(other_max - new_max);
        }
        
        max_val = new_max;
    }
}

__global__ void __launch_bounds__(512) softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    size_t offset = (size_t)row * cols;
    const float* row_input = input + offset;
    float* row_output = output + offset;

    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    int tid = threadIdx.x;
    const float4* row_input_f4 = reinterpret_cast<const float4*>(row_input);
    int cols_f4 = cols / 4;
    
    // Grid Stride Loop
    #pragma unroll 4
    for (int i = tid; i < cols_f4; i += blockDim.x) {
        float4 val4 = row_input_f4[i];
        
        float v0 = val4.x;
        float v1 = val4.y;
        float v2 = val4.z;
        float v3 = val4.w;
        
        float local_max = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
        float new_max = fmaxf(thread_max, local_max);
        
        float term = expf(v0 - new_max) + expf(v1 - new_max) + expf(v2 - new_max) + expf(v3 - new_max);
        thread_sum = thread_sum * expf(thread_max - new_max) + term;
        thread_max = new_max;
    }

    // Warp Reduction
    warp_reduce(thread_max, thread_sum);

    static __shared__ float shared_max[32]; 
    static __shared__ float shared_sum[32];
    
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;
    
    if (lane == 0) {
        shared_max[warp] = thread_max;
        shared_sum[warp] = thread_sum;
    }
    
    __syncthreads();
    
    if (warp == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        float w_max = (lane < num_warps) ? shared_max[lane] : -INFINITY;
        float w_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        
        warp_reduce(w_max, w_sum);
        
        if (lane == 0) {
            shared_max[0] = w_max;
            shared_sum[0] = w_sum;
        }
    }
    
    __syncthreads();
    
    float row_max = shared_max[0];
    float row_sum = shared_sum[0];
    
    // Second Pass
    float4* row_output_f4 = reinterpret_cast<float4*>(row_output);
    float inv_sum = 1.0f / row_sum;
    
    #pragma unroll 4
    for (int i = tid; i < cols_f4; i += blockDim.x) {
        float4 val4 = row_input_f4[i];
        float4 out4;
        
        out4.x = expf(val4.x - row_max) * inv_sum;
        out4.y = expf(val4.y - row_max) * inv_sum;
        out4.z = expf(val4.z - row_max) * inv_sum;
        out4.w = expf(val4.w - row_max) * inv_sum;
        
        row_output_f4[i] = out4;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 512;
    const int grid_size = rows;

    softmax_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    
    return output;
}
"""

softmax_module = load_inline(
    name="softmax_module_v4",
    cpp_sources=softmax_source,
    functions=["softmax_hip"],
    extra_cflags=["-O3", "--gpu-max-threads-per-block=1024"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_hip = softmax_module.softmax_hip
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_hip(x)
