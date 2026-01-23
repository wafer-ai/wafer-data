
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CXX to hipcc for ROCm
os.environ["CXX"] = "hipcc"

# HIP source for highly optimized Bias + Softmax
fused_ops_src = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void bias_softmax_kernel_vectorized(const float4* __restrict__ input, const float* __restrict__ bias, float4* __restrict__ output, int rows, int cols_v4) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float4* input_row = input + row * cols_v4;
    float4* output_row = output + row * cols_v4;

    const int block_size = 256;
    const int elements_per_thread = 16; // 16384 / (256 * 4)
    float4 data[elements_per_thread];

    float max_val = -1e38f;
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = threadIdx.x + i * block_size;
        data[i] = input_row[idx];
        
        // Load bias for each element in float4
        float4 b;
        b.x = bias[idx * 4 + 0];
        b.y = bias[idx * 4 + 1];
        b.z = bias[idx * 4 + 2];
        b.w = bias[idx * 4 + 3];
        
        data[i].x += b.x;
        data[i].y += b.y;
        data[i].z += b.z;
        data[i].w += b.w;
        
        max_val = fmaxf(max_val, data[i].x);
        max_val = fmaxf(max_val, data[i].y);
        max_val = fmaxf(max_val, data[i].z);
        max_val = fmaxf(max_val, data[i].w);
    }

    // Warp reduction
    for (int offset = 32; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_xor(max_val, offset, 64));
    }
    
    __shared__ float shared_val[32]; 
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;
    if (lane == 0) shared_val[wid] = max_val;
    __syncthreads();
    
    if (wid == 0) {
        max_val = (threadIdx.x < 4) ? shared_val[lane] : -1e38f;
        for (int offset = 32; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_xor(max_val, offset, 64));
        }
        shared_val[0] = max_val;
    }
    __syncthreads();
    max_val = shared_val[0];

    float sum_exp = 0.0f;
    for (int i = 0; i < elements_per_thread; ++i) {
        data[i].x = expf(data[i].x - max_val);
        data[i].y = expf(data[i].y - max_val);
        data[i].z = expf(data[i].z - max_val);
        data[i].w = expf(data[i].w - max_val);
        sum_exp += data[i].x + data[i].y + data[i].z + data[i].w;
    }

    for (int offset = 32; offset > 0; offset /= 2) {
        sum_exp += __shfl_xor(sum_exp, offset, 64);
    }
    if (lane == 0) shared_val[wid] = sum_exp;
    __syncthreads();
    
    if (wid == 0) {
        sum_exp = (threadIdx.x < 4) ? shared_val[lane] : 0.0f;
        for (int offset = 32; offset > 0; offset /= 2) {
            sum_exp += __shfl_xor(sum_exp, offset, 64);
        }
        shared_val[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = shared_val[0];

    float inv_sum_exp = 1.0f / sum_exp;
    for (int i = 0; i < elements_per_thread; ++i) {
        data[i].x *= inv_sum_exp;
        data[i].y *= inv_sum_exp;
        data[i].z *= inv_sum_exp;
        data[i].w *= inv_sum_exp;
        output_row[threadIdx.x + i * block_size] = data[i];
    }
}

torch::Tensor fused_bias_softmax_hip(torch::Tensor input, torch::Tensor bias) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    dim3 grid(rows);
    dim3 block(block_size);

    bias_softmax_kernel_vectorized<<<grid, block>>>(
        (const float4*)input.data_ptr<float>(),
        bias.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        rows, cols / 4
    );

    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_src,
    functions=["fused_bias_softmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p
        self.fused_ops = fused_ops

    def forward(self, x):
        # mm is slightly faster than Linear if we don't need bias addition
        x = torch.matmul(x, self.matmul.weight.t())
        
        if not self.training:
            # Fuse Bias and Softmax
            x = self.fused_ops.fused_bias_softmax_hip(x, self.matmul.bias)
        else:
            x = x + self.matmul.bias
            x = torch.nn.functional.dropout(x, p=self.dropout_p, training=True)
            x = torch.softmax(x, dim=1)
        return x

def get_inputs():
    batch_size = 128
    in_features = 16384
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 16384
    out_features = 16384
    dropout_p = 0.2
    return [in_features, out_features, dropout_p]
