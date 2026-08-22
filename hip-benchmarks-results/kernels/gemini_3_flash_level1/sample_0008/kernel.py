
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

rms_norm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* __restrict__ x, float* __restrict__ out, int N, int C, int H, int W, float eps) {
    int idx = blockIdx.x * (int)blockDim.x + threadIdx.x;
    int stride = H * W;
    int num_nhw = N * stride;

    if (idx < num_nhw) {
        int n = idx / stride;
        int hw_idx = idx % stride;
        int base_idx = n * (C * stride) + hw_idx;

        float sum_sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = x[base_idx + c * stride];
            sum_sq += val * val;
        }
        
        float inv_rms = rsqrtf(sum_sq / (float)C + eps);

        for (int c = 0; c < C; ++c) {
            out[base_idx + c * stride] = x[base_idx + c * stride] * inv_rms;
        }
    }
}

torch::Tensor rms_norm_hip(torch::Tensor x, float eps) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    
    auto out = torch::empty_like(x);
    
    int num_nhw = (int)(N * H * W);
    const int block_size = 256;
    const int num_blocks = (num_nhw + block_size - 1) / block_size;
    
    rms_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        (int)N, (int)C, (int)H, (int)W, 
        eps
    );
    
    return out;
}
"""

rms_norm_lib = load_inline(
    name="rms_norm_lib",
    cpp_sources=rms_norm_source,
    functions=["rms_norm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_lib = rms_norm_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm_lib.rms_norm_hip(x, self.eps)
