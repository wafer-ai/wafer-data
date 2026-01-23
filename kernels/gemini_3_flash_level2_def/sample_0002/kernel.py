
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

// Warp-level reduction for sum, only across 'width' threads
__device__ __forceinline__ float segment_reduce_sum(float val, int width) {
    for (int offset = width / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset, width);
    }
    return val;
}

__global__ void fused_swish_bias_gn_kernel_v3(
    const float4* __restrict__ input,      // [batch_size, out_features / 4]
    const float4* __restrict__ bias_param, // [out_features / 4]
    const float4* __restrict__ gn_weight,  // [out_features / 4]
    const float4* __restrict__ gn_bias,    // [out_features / 4]
    float4* __restrict__ output,           // [batch_size, out_features / 4]
    int batch_size,
    int out_features_v4,
    int num_groups,
    int elements_per_group_v4,
    int num_total_groups,
    float eps) {

    int total_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int group_idx = total_thread_idx / 16;
    int group_thread_idx = total_thread_idx % 16; // 0 to 15
    
    if (group_idx >= num_total_groups) return;

    int sample_idx = group_idx / num_groups;
    int group_in_sample = group_idx % num_groups;

    int channel_idx_v4 = group_in_sample * elements_per_group_v4 + group_thread_idx;
    int input_idx_v4 = sample_idx * out_features_v4 + channel_idx_v4;

    float4 in_v4 = input[input_idx_v4];
    float4 bp_v4 = bias_param[channel_idx_v4];

    float vals[4];
    vals[0] = swish(in_v4.x) + bp_v4.x;
    vals[1] = swish(in_v4.y) + bp_v4.y;
    vals[2] = swish(in_v4.z) + bp_v4.z;
    vals[3] = swish(in_v4.w) + bp_v4.w;

    float local_sum = vals[0] + vals[1] + vals[2] + vals[3];
    float local_sum_sq = vals[0]*vals[0] + vals[1]*vals[1] + vals[2]*vals[2] + vals[3]*vals[3];

    // Reduction within the 16 threads of the group
    float group_sum = segment_reduce_sum(local_sum, 16);
    float group_sum_sq = segment_reduce_sum(local_sum_sq, 16);

    float mean = group_sum / 64.0f;
    float var = (group_sum_sq / 64.0f) - (mean * mean);
    float inv_std = 1.0f / sqrtf(fmaxf(var, 0.0f) + eps);

    float4 gw_v4 = gn_weight[channel_idx_v4];
    float4 gb_v4 = gn_bias[channel_idx_v4];

    float4 out_v4;
    out_v4.x = (vals[0] - mean) * inv_std * gw_v4.x + gb_v4.x;
    out_v4.y = (vals[1] - mean) * inv_std * gw_v4.y + gb_v4.y;
    out_v4.z = (vals[2] - mean) * inv_std * gw_v4.z + gb_v4.z;
    out_v4.w = (vals[3] - mean) * inv_std * gw_v4.w + gb_v4.w;

    output[input_idx_v4] = out_v4;
}

torch::Tensor fused_swish_bias_gn_hip(
    torch::Tensor input, 
    torch::Tensor bias_param, 
    torch::Tensor gn_weight, 
    torch::Tensor gn_bias, 
    int num_groups, 
    float eps) {
    
    int batch_size = input.size(0);
    int out_features = input.size(1);
    int elements_per_group = out_features / num_groups;

    auto output = torch::empty_like(input);

    int out_features_v4 = out_features / 4;
    int elements_per_group_v4 = elements_per_group / 4;
    int num_total_groups = batch_size * num_groups;
    
    int threads_per_block = 256;
    int num_blocks = (num_total_groups * 16 + threads_per_block - 1) / threads_per_block;

    fused_swish_bias_gn_kernel_v3<<<num_blocks, threads_per_block>>>(
        (const float4*)input.data_ptr<float>(),
        (const float4*)bias_param.data_ptr<float>(),
        (const float4*)gn_weight.data_ptr<float>(),
        (const float4*)gn_bias.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        batch_size,
        out_features_v4,
        num_groups,
        elements_per_group_v4,
        num_total_groups,
        eps
    );

    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops_v3",
    cpp_sources=fused_kernel_source,
    functions=["fused_swish_bias_gn_hip"],
    verbose=True,
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
        x = fused_ops.fused_swish_bias_gn_hip(
            x, 
            self.bias, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.num_groups, 
            self.group_norm.eps
        )
        return x

def get_inputs():
    batch_size = 32768
    in_features = 1024
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 1024
    out_features = 4096
    num_groups = 64
    bias_shape = (out_features,)
    return [in_features, out_features, num_groups, bias_shape]
