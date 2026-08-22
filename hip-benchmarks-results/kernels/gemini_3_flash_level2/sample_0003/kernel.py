
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Optimized HIP kernel with vectorized loads/stores
fused_bias_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_bias_scale_kernel_vec(float4* x, const float4* bias, float factor, int total_vecs, int cols_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vecs) {
        int col_idx = idx % cols_vec;
        float4 x_val = x[idx];
        float4 b_val = bias[col_idx];
        
        x_val.x = (x_val.x + b_val.x) * factor;
        x_val.y = (x_val.y + b_val.y) * factor;
        x_val.z = (x_val.z + b_val.z) * factor;
        x_val.w = (x_val.w + b_val.w) * factor;
        
        x[idx] = x_val;
    }
}

void fused_bias_scale_hip(torch::Tensor x, torch::Tensor bias, float factor) {
    int rows = x.size(0);
    int cols = x.size(1);
    int total_elements = rows * cols;
    int total_vecs = total_elements / 4;
    int cols_vec = cols / 4;

    const int block_size = 256;
    const int num_blocks = (total_vecs + block_size - 1) / block_size;

    fused_bias_scale_kernel_vec<<<num_blocks, block_size>>>(
        reinterpret_cast<float4*>(x.data_ptr<float>()),
        reinterpret_cast<const float4*>(bias.data_ptr<float>()),
        factor,
        total_vecs,
        cols_vec
    );
}
"""

fused_bias_scale_op = load_inline(
    name="fused_bias_scale_final",
    cpp_sources=fused_bias_scale_source,
    functions=["fused_bias_scale_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.factor = 1.0 + scaling_factor

    def forward(self, x):
        # res = x @ weight.T
        res = torch.mm(x, self.matmul.weight.t())
        
        # Then we use our custom HIP kernel to add bias and scale
        fused_bias_scale_op.fused_bias_scale_hip(res, self.matmul.bias, self.factor)
        
        return res

def get_inputs():
    return [torch.rand(16384, 4096).cuda()]

def get_init_inputs():
    return [4096, 4096, 0.5]
