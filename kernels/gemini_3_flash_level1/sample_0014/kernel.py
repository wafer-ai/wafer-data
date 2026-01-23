
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

softmax_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

struct MaxSum {
    float max_val;
    float sum_val;
};

__device__ __forceinline__ MaxSum combine(MaxSum a, MaxSum b) {
    if (a.max_val > b.max_val) {
        return {a.max_val, a.sum_val + b.sum_val * expf(b.max_val - a.max_val)};
    } else {
        return {b.max_val, b.sum_val + a.sum_val * expf(a.max_val - b.max_val)};
    }
}

__global__ void softmax_online_kernel_v5(const float4* __restrict__ input, float4* __restrict__ output, int batch_size, int dim_v4) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float4* row_input = input + row * dim_v4;
    float4* row_output = output + row * dim_v4;

    MaxSum thread_res = {-1e38f, 0.0f};

    // First pass: find max and sum
    for (int i = threadIdx.x; i < dim_v4; i += blockDim.x) {
        float4 val4 = row_input[i];
        thread_res = combine(thread_res, {val4.x, 1.0f});
        thread_res = combine(thread_res, {val4.y, 1.0f});
        thread_res = combine(thread_res, {val4.z, 1.0f});
        thread_res = combine(thread_res, {val4.w, 1.0f});
    }

    static __shared__ MaxSum shared[64];
    int warp_size = 64; // AMD MI300X
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    
    // Warp-level reduction
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down(thread_res.max_val, offset);
        float other_sum = __shfl_down(thread_res.sum_val, offset);
        thread_res = combine(thread_res, {other_max, other_sum});
    }

    if (lane_id == 0) shared[warp_id] = thread_res;
    __syncthreads();

    // Final reduction across warps
    if (warp_id == 0) {
        MaxSum warp_res = (threadIdx.x < (blockDim.x / warp_size)) ? shared[threadIdx.x] : (MaxSum){-1e38f, 0.0f};
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            float other_max = __shfl_down(warp_res.max_val, offset);
            float other_sum = __shfl_down(warp_res.sum_val, offset);
            warp_res = combine(warp_res, {other_max, other_sum});
        }
        if (lane_id == 0) shared[0] = warp_res;
    }
    __syncthreads();

    float row_max = shared[0].max_val;
    float row_sum = shared[0].sum_val;
    float inv_row_sum = 1.0f / row_sum;

    // Second pass: compute output
    for (int i = threadIdx.x; i < dim_v4; i += blockDim.x) {
        float4 val4 = row_input[i];
        float4 res4;
        res4.x = expf(val4.x - row_max) * inv_row_sum;
        res4.y = expf(val4.y - row_max) * inv_row_sum;
        res4.z = expf(val4.z - row_max) * inv_row_sum;
        res4.w = expf(val4.w - row_max) * inv_row_sum;
        row_output[i] = res4;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = batch_size;
    const int dim_v4 = dim / 4;

    softmax_online_kernel_v5<<<num_blocks, block_size>>>(
        (const float4*)input.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        batch_size,
        dim_v4
    );

    return output;
}
"""

softmax_module = load_inline(
    name="softmax_online_v5",
    cpp_sources=softmax_cpp_source,
    functions=["softmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax_hip = softmax_module.softmax_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_hip(x)
