
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

softmax_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 1024
#define WARP_SIZE 64

struct MaxSum {
    float max_val;
    float sum_val;
};

__device__ __forceinline__ MaxSum combine_max_sum(MaxSum a, MaxSum b) {
    if (a.max_val >= b.max_val) {
        return {a.max_val, a.sum_val + __expf(b.max_val - a.max_val) * b.sum_val};
    } else {
        return {b.max_val, b.sum_val + __expf(a.max_val - b.max_val) * a.sum_val};
    }
}

__device__ __forceinline__ MaxSum warp_reduce_max_sum(MaxSum val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        MaxSum other;
        other.max_val = __shfl_xor(val.max_val, offset, WARP_SIZE);
        other.sum_val = __shfl_xor(val.sum_val, offset, WARP_SIZE);
        val = combine_max_sum(val, other);
    }
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * dim;
    float* row_output = output + row * dim;

    MaxSum thread_res = {-INFINITY, 0.0f};

    const float4* row_input_v4 = reinterpret_cast<const float4*>(row_input);
    int dim_v4 = dim / 4;

    // Unroll first pass
    for (int i = threadIdx.x; i < dim_v4; i += BLOCK_SIZE * 2) {
        float4 vals = row_input_v4[i];
        float v[4] = {vals.x, vals.y, vals.z, vals.w};
        for (int j = 0; j < 4; j++) {
            if (v[j] > thread_res.max_val) {
                thread_res.sum_val = thread_res.sum_val * __expf(thread_res.max_val - v[j]) + 1.0f;
                thread_res.max_val = v[j];
            } else {
                thread_res.sum_val += __expf(v[j] - thread_res.max_val);
            }
        }
        if (i + BLOCK_SIZE < dim_v4) {
            float4 vals2 = row_input_v4[i + BLOCK_SIZE];
            float v2[4] = {vals2.x, vals2.y, vals2.z, vals2.w};
            for (int j = 0; j < 4; j++) {
                if (v2[j] > thread_res.max_val) {
                    thread_res.sum_val = thread_res.sum_val * __expf(thread_res.max_val - v2[j]) + 1.0f;
                    thread_res.max_val = v2[j];
                } else {
                    thread_res.sum_val += __expf(v2[j] - thread_res.max_val);
                }
            }
        }
    }

    MaxSum warp_res = warp_reduce_max_sum(thread_res);

    __shared__ float shared_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        shared_max[warp_id] = warp_res.max_val;
        shared_sum[warp_id] = warp_res.sum_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        MaxSum block_res = {-INFINITY, 0.0f};
        if (lane_id < (BLOCK_SIZE / WARP_SIZE)) {
            block_res.max_val = shared_max[lane_id];
            block_res.sum_val = shared_sum[lane_id];
        }
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            MaxSum other;
            other.max_val = __shfl_xor(block_res.max_val, offset, WARP_SIZE);
            other.sum_val = __shfl_xor(block_res.sum_val, offset, WARP_SIZE);
            block_res = combine_max_sum(block_res, other);
        }
        if (lane_id == 0) {
            shared_max[0] = block_res.max_val;
            shared_sum[0] = block_res.sum_val;
        }
    }
    __syncthreads();

    float row_max = shared_max[0];
    float row_sum = shared_sum[0];
    float inv_row_sum = 1.0f / row_sum;

    float4* row_output_v4 = reinterpret_cast<float4*>(row_output);
    for (int i = threadIdx.x; i < dim_v4; i += BLOCK_SIZE) {
        float4 vals = row_input_v4[i];
        vals.x = __expf(vals.x - row_max) * inv_row_sum;
        vals.y = __expf(vals.y - row_max) * inv_row_sum;
        vals.z = __expf(vals.z - row_max) * inv_row_sum;
        vals.w = __expf(vals.w - row_max) * inv_row_sum;
        row_output_v4[i] = vals;
    }
}

torch::Tensor softmax_hip(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto y = torch::empty_like(x);

    const int num_blocks = batch_size;
    const int threads_per_block = BLOCK_SIZE;

    softmax_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        dim
    );

    return y;
}
"""

softmax_module = load_inline(
    name="softmax_hip",
    cpp_sources=softmax_cpp_source,
    functions=["softmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_hip = softmax_module.softmax_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_hip(x)
