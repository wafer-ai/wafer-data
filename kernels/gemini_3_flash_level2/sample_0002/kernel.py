
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# HIP kernel for fused Swish + Bias + GroupNorm
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__device__ inline float swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_swish_bias_gn_vectorized_kernel(
    const float4* __restrict__ input,      // [batch_size, out_features / 4]
    const float* __restrict__ bias,       // [out_features]
    const float* __restrict__ gn_weight,  // [out_features]
    const float* __restrict__ gn_bias,    // [out_features]
    float4* __restrict__ output,           // [batch_size, out_features / 4]
    int batch_size,
    int out_features_div4,
    int num_groups,
    int elements_per_group,
    float eps) 
{
    int tid = threadIdx.x;
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    
    // Each warp handles 4 groups of 16 threads each (16 * 4 = 64)
    int group_in_warp = lane_id / 16;
    int lane_in_group = lane_id % 16;
    
    int total_groups = batch_size * num_groups;
    int group_idx_in_batch = (blockIdx.x * (blockDim.x / 64) + warp_id) * 4 + group_in_warp;
    
    if (group_idx_in_batch >= total_groups) return;
    
    int batch_idx = group_idx_in_batch / num_groups;
    int group_idx = group_idx_in_batch % num_groups;
    
    int offset = batch_idx * out_features_div4 + group_idx * (elements_per_group / 4) + lane_in_group;
    int weight_offset_base = group_idx * elements_per_group + lane_in_group * 4;
    
    float4 val4 = input[offset];
    
    // Load bias (4 floats)
    float b0 = bias[weight_offset_base + 0];
    float b1 = bias[weight_offset_base + 1];
    float b2 = bias[weight_offset_base + 2];
    float b3 = bias[weight_offset_base + 3];
    
    // Apply Swish and bias addition
    val4.x = swish(val4.x) + b0;
    val4.y = swish(val4.y) + b1;
    val4.z = swish(val4.z) + b2;
    val4.w = swish(val4.w) + b3;
    
    float sum = val4.x + val4.y + val4.z + val4.w;
    float sq_sum = val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
    
    // Warp shuffle reduction within 16-thread group
    for (int i = 8; i > 0; i /= 2) {
        sum += __shfl_xor(sum, i, 64);
        sq_sum += __shfl_xor(sq_sum, i, 64);
    }
    
    // Note: __shfl_xor with 64 will still work here, but we only care about the first 16 threads' result
    // Actually, we need each of the 16 threads to have the same mean and var.
    // The shuffle above for i=8,4,2,1 within a 16-thread group will make all 16 threads have the same sum.
    
    float mean = sum / (float)elements_per_group;
    float var = sq_sum / (float)elements_per_group - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Load GN weight and bias (4 floats each)
    float gw0 = gn_weight[weight_offset_base + 0];
    float gw1 = gn_weight[weight_offset_base + 1];
    float gw2 = gn_weight[weight_offset_base + 2];
    float gw3 = gn_weight[weight_offset_base + 3];
    
    float gb0 = gn_bias[weight_offset_base + 0];
    float gb1 = gn_bias[weight_offset_base + 1];
    float gb2 = gn_bias[weight_offset_base + 2];
    float gb3 = gn_bias[weight_offset_base + 3];
    
    val4.x = (val4.x - mean) * inv_std * gw0 + gb0;
    val4.y = (val4.y - mean) * inv_std * gw1 + gb1;
    val4.z = (val4.z - mean) * inv_std * gw2 + gb2;
    val4.w = (val4.w - mean) * inv_std * gw3 + gb3;
    
    output[offset] = val4;
}

torch::Tensor fused_swish_bias_gn_hip(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups,
    float eps) 
{
    int batch_size = input.size(0);
    int out_features = input.size(1);
    int elements_per_group = out_features / num_groups;
    
    auto output = torch::empty_like(input);
    
    int threads_per_block = 256;
    int groups_per_block = (threads_per_block / 64) * 4;
    int total_groups = batch_size * num_groups;
    int num_blocks = (total_groups + groups_per_block - 1) / groups_per_block;
    
    fused_swish_bias_gn_vectorized_kernel<<<num_blocks, threads_per_block>>>(
        (const float4*)input.data_ptr<float>(),
        bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        batch_size,
        out_features / 4,
        num_groups,
        elements_per_group,
        eps
    );
    
    return output;
}
"""

fused_op = load_inline(
    name="fused_swish_bias_gn_v2",
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
        self.fused_op = fused_op

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_op.fused_swish_bias_gn_hip(
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
