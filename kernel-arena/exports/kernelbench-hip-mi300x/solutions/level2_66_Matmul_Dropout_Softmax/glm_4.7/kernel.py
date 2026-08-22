import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized Fused Matmul + Dropout + Softmax
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void matmul_dropout_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float dropout_p,
    float scale,
    bool training,
    unsigned long long seed
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= batch_size || col >= out_features) return;
    
    float sum = 0.0f;
    
    // Unroll loop for better performance
    int k = 0;
    for (; k + 3 < in_features; k += 4) {
        sum += x[row * in_features + k] * weight[col * in_features + k];
        sum += x[row * in_features + k + 1] * weight[col * in_features + k + 1];
        sum += x[row * in_features + k + 2] * weight[col * in_features + k + 2];
        sum += x[row * in_features + k + 3] * weight[col * in_features + k + 3];
    }
    for (; k < in_features; ++k) {
        sum += x[row * in_features + k] * weight[col * in_features + k];
    }
    
    if (bias != nullptr) {
        sum += bias[col];
    }
    
    if (training) {
        unsigned long long idx = (unsigned long long)row * out_features + col;
        unsigned long long hash = (idx * 0x9e3779b97f4a7c15ULL) ^ seed;
        float rand_val = (float)((hash >> 32) ^ (hash & 0xFFFFFFFF)) / (float)0xFFFFFFFF;
        
        if (rand_val < dropout_p) {
            sum = 0.0f;
        } else {
            sum *= scale;
        }
    }
    
    output[row * out_features + col] = sum;
}

__global__ void softmax_atomic_kernel(
    float* __restrict__ output,
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    int batch_size,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= batch_size || col >= out_features) return;
    
    int idx = row * out_features + col;
    float val = output[idx];
    
    // Atomic max for row maximum
    atomicMax(&row_max[row], val);
}

__global__ void softmax_normalize_kernel(
    float* __restrict__ output,
    const float* __restrict__ row_max_in,
    float* __restrict__ row_sum,
    int batch_size,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= batch_size || col >= out_features) return;
    
    int idx = row * out_features + col;
    float max_val = row_max_in[row];
    float exp_val = expf(fmaxf(output[idx] - max_val, -50.0f));
    
    output[idx] = exp_val;
    atomicAdd(&row_sum[row], exp_val);
}

__global__ void softmax_divide_kernel(
    float* __restrict__ output,
    const float* __restrict__ row_sum_in,
    int batch_size,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= batch_size || col >= out_features) return;
    
    int idx = row * out_features + col;
    float sum = row_sum_in[row];
    output[idx] = output[idx] / fmaxf(sum, 1e-10f);
}

torch::Tensor fused_matmul_dropout_softmax_hip(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_p,
    bool training
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, x.options());
    auto row_max = torch::full({batch_size}, -1e30f, x.options());
    auto row_sum = torch::zeros({batch_size}, x.options());
    float scale = 1.0f / (1.0f - dropout_p + 1e-6f);
    unsigned long long seed = (unsigned long long)x.data_ptr<float>();
    
    // Matmul + Dropout
    dim3 blockDim(32, 8);
    dim3 gridDim((out_features + 31) / 32, (batch_size + 7) / 8);
    
    matmul_dropout_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        dropout_p,
        scale,
        training,
        seed
    );
    
    hipStreamSynchronize(0);
    
    // Softmax - find max
    dim3 softBlock(32, 8);
    dim3 softGrid((out_features + 31) / 32, (batch_size + 7) / 8);
    
    softmax_atomic_kernel<<<softGrid, softBlock>>>(
        output.data_ptr<float>(),
        row_max.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    hipStreamSynchronize(0);
    
    // Normalize and sum
    softmax_normalize_kernel<<<softGrid, softBlock>>>(
        output.data_ptr<float>(),
        row_max.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    hipStreamSynchronize(0);
    
    // Divide
    softmax_divide_kernel<<<softGrid, softBlock>>>(
        output.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_module = load_inline(
    name="fused_matmul_dropout_softmax",
    cpp_sources=fused_kernel_source,
    functions=["fused_matmul_dropout_softmax_hip"],
    verbose=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.dropout_p = dropout_p
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_module

    def forward(self, x):
        output = self.fused_op.fused_matmul_dropout_softmax_hip(
            x, self.weight, self.bias, self.dropout_p, self.training
        )
        return output

import math

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]