import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_scale_residual_kernel(const float *a, const float *res, float scale, float *out, int64_t size) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scale + res[idx];
    }
}

torch::Tensor fused_scale_residual_hip(torch::Tensor a, torch::Tensor res, float scale) {
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Float, "Must be FP32");
    TORCH_CHECK(a.sizes() == res.sizes(), "a and res must have same shape");
    TORCH_CHECK(a.is_cuda(), "Must be on GPU");
    auto out = torch::empty_like(a);
    int64_t size = a.numel();
    const int block_size = 1024;
    int64_t num_blocks = (size + block_size - 1) / block_size;
    dim3 block(block_size);
    dim3 grid(num_blocks);
    fused_scale_residual_kernel<<<grid, block>>>(a.data_ptr<float>(), res.data_ptr<float>(), scale, out.data_ptr<float>(), size);
    return out;
}
"""

fused_module = load_inline(
    name="fused_scale_residual",
    cpp_sources=hip_source,
    functions=["fused_scale_residual_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.fused_op = fused_module

    def forward(self, x):
        matmul_out = self.linear(x)
        residual = matmul_out.detach()
        return self.fused_op.fused_scale_residual_hip(matmul_out, residual, self.scaling_factor)
