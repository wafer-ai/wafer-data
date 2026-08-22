import os
# Set compiler to hipcc before importing torch
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_decl = """
#include <torch/extension.h>

torch::Tensor fused_swish_bias_gn(
    torch::Tensor input,
    torch::Tensor custom_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups,
    float eps);
"""

cuda_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64

__device__ __forceinline__ float warp_reduce_sum_broadcast(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__global__ void fused_swish_bias_gn_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ custom_bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    int N,
    int C,
    int G,
    float eps) 
{
    int tid = threadIdx.x;
    int wf_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int global_wf_id = blockIdx.x * (blockDim.x / WARP_SIZE) + wf_id;
    
    if (global_wf_id >= N * G) return;
    
    int n = global_wf_id / G;
    int g = global_wf_id % G;
    
    int c = g * WARP_SIZE + lane_id;
    int idx = n * C + c;
    
    float val = input[idx];
    
    // 1. Swish
    float sig = 1.0f / (1.0f + __expf(-val));
    val = val * sig;
    
    // 2. Add Custom Bias
    val = val + custom_bias[c];
    
    // 3. GroupNorm
    float sum = warp_reduce_sum_broadcast(val);
    float mean = sum / (float)WARP_SIZE;
    
    float diff = val - mean;
    float sum_sq = warp_reduce_sum_broadcast(diff * diff);
    float var = sum_sq / (float)WARP_SIZE;
    
    float inv_std = rsqrtf(var + eps);
    float norm = (val - mean) * inv_std;
    
    float out = norm * gn_weight[c] + gn_bias[c];
    
    output[idx] = out;
}

torch::Tensor fused_swish_bias_gn(
    torch::Tensor input,
    torch::Tensor custom_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups,
    float eps)
{
    int N = input.size(0);
    int C = input.size(1);
    
    if (!input.is_contiguous()) input = input.contiguous();
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int warps_per_block = block_size / WARP_SIZE;
    int total_tasks = N * num_groups;
    int grid_size = (total_tasks + warps_per_block - 1) / warps_per_block;
    
    fused_swish_bias_gn_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        custom_bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        N, C, num_groups, eps
    );
    
    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_decl,
    cuda_sources=cuda_source,
    functions=["fused_swish_bias_gn"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups

    def forward(self, x):
        x = self.matmul(x)
        if not x.is_contiguous(): x = x.contiguous()
        return fused_ops.fused_swish_bias_gn(
            x, 
            self.bias, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.num_groups, 
            1e-5
        )

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
