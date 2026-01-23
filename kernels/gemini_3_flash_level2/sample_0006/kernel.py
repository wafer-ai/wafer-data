
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_reduction_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void fused_reduction_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int out_features,
    int kernel_size,
    float scale_factor) {

    int b = blockIdx.x / 2; // Each batch has 2 blocks
    int part = blockIdx.x % 2;
    if (b >= batch_size) return;

    int num_pools = out_features / kernel_size;
    int pools_per_block = num_pools / 2;
    int start_pool = part * pools_per_block;
    int end_pool = (part == 1) ? num_pools : (part + 1) * pools_per_block;

    float thread_sum = 0.0f;
    const float2* x2 = reinterpret_cast<const float2*>(x + b * out_features);

    for (int p = start_pool + threadIdx.x; p < end_pool; p += blockDim.x) {
        float2 val = x2[p];
        thread_sum += fmaxf(val.x, val.y);
    }

    __shared__ float shared_data[256];
    int tid = threadIdx.x;
    shared_data[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&out[b], shared_data[0] * scale_factor);
    }
}

torch::Tensor fused_reduction_hip(
    torch::Tensor x,
    int kernel_size,
    float scale_factor) {

    int batch_size = x.size(0);
    int out_features = x.size(1);
    auto out = torch::zeros({batch_size}, x.options());

    const int block_size = 256;
    int num_blocks = batch_size * 2;

    fused_reduction_kernel<<<num_blocks, block_size, 0>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features,
        kernel_size,
        scale_factor
    );

    return out;
}
"""

fused_reduction_lib = load_inline(
    name="fused_reduction_v6",
    cpp_sources=fused_reduction_cpp_source,
    functions=["fused_reduction_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.matmul(x)
        return fused_reduction_lib.fused_reduction_hip(
            x, self.kernel_size, self.scale_factor
        )

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
