import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized RMSNorm kernel - single pass through memory
// Each thread handles one spatial position
__global__ __launch_bounds__(256) void rmsnorm_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int dim1,
    const int dim2,
    const float eps,
    const float inv_num_features
) {
    const int spatial_size = dim1 * dim2;
    const int total_positions = batch_size * spatial_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_positions) return;
    
    const int batch_idx = idx / spatial_size;
    const int spatial_idx = idx % spatial_size;
    
    const int feature_stride = spatial_size;
    const int batch_offset = batch_idx * num_features * feature_stride + spatial_idx;
    
    // First pass: compute sum of squares
    float sum_sq = 0.0f;
    
    // Load values and compute sum of squares
    // Fully unrolled for 64 features
    #pragma unroll 16
    for (int f = 0; f < num_features; f++) {
        float val = input[batch_offset + f * feature_stride];
        sum_sq += val * val;
    }
    
    // Compute inverse RMS using fast rsqrt
    float inv_rms = rsqrtf(sum_sq * inv_num_features + eps);
    
    // Second pass: normalize and write
    #pragma unroll 16
    for (int f = 0; f < num_features; f++) {
        int addr = batch_offset + f * feature_stride;
        output[addr] = input[addr] * inv_rms;
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int dim1 = input.size(2);
    const int dim2 = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int total_positions = batch_size * dim1 * dim2;
    const int block_size = 256;
    const int num_blocks = (total_positions + block_size - 1) / block_size;
    const float inv_num_features = 1.0f / num_features;
    
    rmsnorm_fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        dim1,
        dim2,
        eps,
        inv_num_features
    );
    
    return output;
}
"""

rmsnorm_cpp_source = """
torch::Tensor rmsnorm_hip(torch::Tensor input, float eps);
"""

rmsnorm_module = load_inline(
    name="rmsnorm_hip_v5",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_hip_source,
    functions=["rmsnorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized RMS Normalization using fused HIP kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_module.rmsnorm_hip(x, self.eps)


def get_inputs():
    x = torch.rand(112, 64, 512, 512).cuda()
    return [x]


def get_init_inputs():
    return [64]
