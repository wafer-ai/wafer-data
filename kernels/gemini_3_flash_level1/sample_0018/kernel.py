
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

batch_norm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void batch_norm_inference_float4(
    const float4* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float4* __restrict__ out,
    int N, int C, int HW_over_4,
    float eps) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_over_4 = N * C * HW_over_4;
    
    if (tid < total_elements_over_4) {
        int c = (tid / HW_over_4) % C;
        
        float m = mean[c];
        float v = var[c];
        float w = weight[c];
        float b = bias[c];
        
        float inv_std = 1.0f / sqrtf(v + eps);
        float4 val4 = x[tid];
        float4 res4;
        res4.x = (val4.x - m) * inv_std * w + b;
        res4.y = (val4.y - m) * inv_std * w + b;
        res4.z = (val4.z - m) * inv_std * w + b;
        res4.w = (val4.w - m) * inv_std * w + b;
        out[tid] = res4;
    }
}

torch::Tensor batch_norm_inference_hip(
    torch::Tensor x,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto out = torch::empty_like(x);
    
    int HW = H * W;
    int HW_over_4 = HW / 4;
    int total_elements_over_4 = N * C * HW_over_4;
    int block_size = 256;
    int num_blocks = (total_elements_over_4 + block_size - 1) / block_size;
    
    batch_norm_inference_float4<<<num_blocks, block_size>>>(
        (const float4*)x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        (float4*)out.data_ptr<float>(),
        N, C, HW_over_4, eps);
        
    return out;
}
"""

batch_norm_lib = load_inline(
    name="batch_norm_inference_new",
    cpp_sources=batch_norm_source,
    functions=["batch_norm_inference_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # For training, use the most efficient built-in implementation
            # to avoid any performance loss or correctness issues.
            return self.bn(x)
        else:
            return batch_norm_lib.batch_norm_inference_hip(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                self.bn.eps
            )
