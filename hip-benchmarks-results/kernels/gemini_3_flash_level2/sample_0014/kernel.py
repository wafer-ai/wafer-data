
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_bias_gelu_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define SOFTMAX_BLOCK_SIZE 256
#define WAVEFRONT_SIZE 64
#define ELEMENTS_PER_THREAD 32

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
}

__device__ inline void online_softmax_step(float& m, float& s, float x) {
    if (x > m) {
        s = s * expf(m - x) + 1.0f;
        m = x;
    } else {
        s = s + expf(x - m);
    }
}

__device__ inline void warp_reduce_online_softmax(float& m, float& s) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        float other_m = __shfl_xor(m, offset, WAVEFRONT_SIZE);
        float other_s = __shfl_xor(s, offset, WAVEFRONT_SIZE);
        if (other_m > m) {
            s = s * expf(m - other_m) + other_s;
            m = other_m;
        } else {
            s = s + other_s * expf(other_m - m);
        }
    }
}

__global__ void bias_gelu_softmax_kernel(const float4* __restrict__ input, const float4* __restrict__ bias, float4* __restrict__ output, int rows, int cols_v4) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float4* row_input = input + row * cols_v4;
    float4* row_output = output + row * cols_v4;

    float local_vals[ELEMENTS_PER_THREAD];
    float m = -1e20f;
    float s = 0.0f;

    for (int i = 0; i < ELEMENTS_PER_THREAD / 4; ++i) {
        int idx = threadIdx.x + i * SOFTMAX_BLOCK_SIZE;
        if (idx < cols_v4) {
            float4 v4 = row_input[idx];
            float4 b4 = bias[idx];
            float g1 = gelu(v4.x + b4.x);
            float g2 = gelu(v4.y + b4.y);
            float g3 = gelu(v4.z + b4.z);
            float g4 = gelu(v4.w + b4.w);
            local_vals[i*4 + 0] = g1;
            local_vals[i*4 + 1] = g2;
            local_vals[i*4 + 2] = g3;
            local_vals[i*4 + 3] = g4;
            online_softmax_step(m, s, g1);
            online_softmax_step(m, s, g2);
            online_softmax_step(m, s, g3);
            online_softmax_step(m, s, g4);
        }
    }

    __shared__ float shared_m[SOFTMAX_BLOCK_SIZE / WAVEFRONT_SIZE];
    __shared__ float shared_s[SOFTMAX_BLOCK_SIZE / WAVEFRONT_SIZE];
    
    warp_reduce_online_softmax(m, s);
    
    if ((threadIdx.x % WAVEFRONT_SIZE) == 0) {
        shared_m[threadIdx.x / WAVEFRONT_SIZE] = m;
        shared_s[threadIdx.x / WAVEFRONT_SIZE] = s;
    }
    __syncthreads();

    float final_m = shared_m[0];
    float final_s = shared_s[0];
    for (int i = 1; i < (SOFTMAX_BLOCK_SIZE / WAVEFRONT_SIZE); ++i) {
        float other_m = shared_m[i];
        float other_s = shared_s[i];
        if (other_m > final_m) {
            final_s = final_s * expf(final_m - other_m) + other_s;
            final_m = other_m;
        } else {
            final_s = final_s + other_s * expf(other_m - final_m);
        }
    }

    float inv_s = 1.0f / final_s;

    for (int i = 0; i < ELEMENTS_PER_THREAD / 4; ++i) {
        int idx = threadIdx.x + i * SOFTMAX_BLOCK_SIZE;
        if (idx < cols_v4) {
            float4 res;
            res.x = expf(local_vals[i*4 + 0] - final_m) * inv_s;
            res.y = expf(local_vals[i*4 + 1] - final_m) * inv_s;
            res.z = expf(local_vals[i*4 + 2] - final_m) * inv_s;
            res.w = expf(local_vals[i*4 + 3] - final_m) * inv_s;
            row_output[idx] = res;
        }
    }
}

torch::Tensor bias_gelu_softmax_hip(torch::Tensor input, torch::Tensor bias) {
    int rows = input.size(0);
    int cols = input.size(1);
    int cols_v4 = cols / 4;
    auto output = torch::empty_like(input);

    dim3 grid(rows);
    dim3 block(SOFTMAX_BLOCK_SIZE);

    hipLaunchKernelGGL(bias_gelu_softmax_kernel, grid, block, 0, 0, (const float4*)input.data_ptr<float>(), (const float4*)bias.data_ptr<float>(), (float4*)output.data_ptr<float>(), rows, cols_v4);

    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bias_gelu_softmax_source,
    functions=["bias_gelu_softmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features).cuda()

    def forward(self, x):
        # Using torch.mm instead of self.linear(x) to avoid double bias addition
        z = torch.mm(x, self.linear.weight.t())
        x = fused_ops.bias_gelu_softmax_hip(z, self.linear.bias)
        return x
