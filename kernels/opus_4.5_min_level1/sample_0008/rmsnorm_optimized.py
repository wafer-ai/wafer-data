import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Kernel for RMSNorm: fuses square, mean, sqrt, and division
// Input shape: (batch_size, num_features, dim1, dim2)
// We reduce along dim=1 (num_features)

__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int num_features,
    int dim1,
    int dim2,
    float eps
) {
    // Each thread handles one spatial position (batch, d1, d2)
    // Grid: (dim2, dim1, batch_size)
    int d2 = blockIdx.x * blockDim.x + threadIdx.x;
    int d1 = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    
    if (d2 >= dim2 || d1 >= dim1 || b >= batch_size) return;
    
    int spatial_size = dim1 * dim2;
    int feature_stride = spatial_size;
    
    // Base index for this spatial position
    int base_idx = b * num_features * spatial_size + d1 * dim2 + d2;
    
    // Calculate sum of squares across features
    float sum_sq = 0.0f;
    for (int f = 0; f < num_features; f++) {
        float val = x[base_idx + f * feature_stride];
        sum_sq += val * val;
    }
    
    // Calculate RMS
    float rms = sqrtf(sum_sq / (float)num_features + eps);
    float inv_rms = 1.0f / rms;
    
    // Normalize
    for (int f = 0; f < num_features; f++) {
        int idx = base_idx + f * feature_stride;
        out[idx] = x[idx] * inv_rms;
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor x, float eps) {
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int num_features = sizes[1];
    int dim1 = sizes[2];
    int dim2 = sizes[3];
    
    auto out = torch::empty_like(x);
    
    // Use 2D blocks for spatial dimensions
    dim3 block(16, 16, 1);
    dim3 grid(
        (dim2 + block.x - 1) / block.x,
        (dim1 + block.y - 1) / block.y,
        batch_size
    );
    
    rmsnorm_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_features,
        dim1,
        dim2,
        eps
    );
    
    return out;
}
"""

rmsnorm_cpp_source = """
torch::Tensor rmsnorm_hip(torch::Tensor x, float eps);
"""

rmsnorm_module = load_inline(
    name="rmsnorm_hip",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_hip_source,
    functions=["rmsnorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using a custom HIP kernel.
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
