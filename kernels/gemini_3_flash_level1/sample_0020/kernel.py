
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

rms_norm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void rms_norm_kernel(const float* __restrict__ x, float* __restrict__ out, int N, int C, int HW, float eps) {
    int i = blockIdx.x * (int)blockDim.x + threadIdx.x;
    int num_nhw = N * HW;
    if (i < num_nhw) {
        int n = i / HW;
        int hw = i % HW;
        
        long long base_idx = (long long)n * C * HW + hw;
        const float* x_ptr = x + base_idx;
        float* out_ptr = out + base_idx;

        float sum_sq = 0.0f;
        long long hw_long = HW;
        for (int c = 0; c < C; ++c) {
            float val = x_ptr[c * hw_long];
            sum_sq += val * val;
        }
        
        float inv_rms = rsqrtf(sum_sq / (float)C + eps);
        
        for (int c = 0; c < C; ++c) {
            out_ptr[c * hw_long] = x_ptr[c * hw_long] * inv_rms;
        }
    }
}

torch::Tensor rms_norm_hip(torch::Tensor x, float eps) {
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int HW = 1;
    for (int i = 2; i < x.dim(); ++i) {
        HW *= sizes[i];
    }

    auto out = torch::empty_like(x);

    int num_nhw = N * HW;
    const int block_size = 256;
    const int num_blocks = (num_nhw + block_size - 1) / block_size;

    rms_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, C, HW, eps
    );

    return out;
}
"""

rms_norm_module = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_source,
    functions=["rms_norm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm = rms_norm_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm.rms_norm_hip(x, self.eps)
