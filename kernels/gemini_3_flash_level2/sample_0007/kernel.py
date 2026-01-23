
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Optimized HIP source for the fused swish + scaling kernel
fused_op_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float swish_scaling(float x, float scaling_factor) {
    return (x / (1.0f + __expf(-x))) * scaling_factor;
}

__global__ void fused_swish_scaling_kernel_vec(
    float* __restrict__ data,
    int size,
    float scaling_factor) {
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        float4 val_vec = reinterpret_cast<float4*>(&data[idx])[0];
        val_vec.x = swish_scaling(val_vec.x, scaling_factor);
        val_vec.y = swish_scaling(val_vec.y, scaling_factor);
        val_vec.z = swish_scaling(val_vec.z, scaling_factor);
        val_vec.w = swish_scaling(val_vec.w, scaling_factor);
        reinterpret_cast<float4*>(&data[idx])[0] = val_vec;
    } else {
        for (int i = idx; i < size; ++i) {
            data[i] = swish_scaling(data[i], scaling_factor);
        }
    }
}

torch::Tensor fused_swish_scaling_hip(torch::Tensor data, float scaling_factor) {
    int size = data.numel();
    const int block_size = 256;
    int num_blocks = (size / 4 + block_size - 1) / block_size;
    if (num_blocks > 0) {
        fused_swish_scaling_kernel_vec<<<num_blocks, block_size>>>(
            data.data_ptr<float>(),
            size,
            scaling_factor
        );
    }
    return data;
}
"""

fused_op = load_inline(
    name="fused_op_v4",
    cpp_sources=fused_op_source,
    functions=["fused_swish_scaling_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.matmul(x)
        # Apply fused swish and scaling in-place
        return self.fused_op.fused_swish_scaling_hip(x, float(self.scaling_factor))

def get_inputs():
    return [torch.rand(128, 32768).cuda()]

def get_init_inputs():
    return [32768, 32768, 2.0]
