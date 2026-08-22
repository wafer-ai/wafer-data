import os
os.environ['CXX'] = 'hipcc'
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void swish_bias_kernel(const float* x, const float* bias, float* out, int C, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int c = idx % C;
        float val = x[idx];
        float sig = 1.0f / (1.0f + __expf(-val));
        out[idx] = sig * val + bias[c];
    }
}

torch::Tensor swish_bias_hip(torch::Tensor x, torch::Tensor bias) {
    auto total = x.numel();
    auto C = x.size(1);
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    swish_bias_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), (int)C, (int)total);
    return out;
}
"""

swish_module = load_inline(
    name="swish_bias",
    cpp_sources=cpp_source,
    functions=["swish_bias_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.swish_bias = swish_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.swish_bias.swish_bias_hip(x, self.bias)
        x = self.group_norm(x)
        return x
