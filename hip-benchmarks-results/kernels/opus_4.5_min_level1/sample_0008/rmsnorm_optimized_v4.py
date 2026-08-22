import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Optimized kernel using warp-level parallelism for feature reduction
// Each warp handles one spatial position, threads in warp reduce across features
__global__ void rmsnorm_warp_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int num_features,
    int spatial_size,
    float eps,
    float inv_num_features
) {
    // Each warp handles one spatial position
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    int total_positions = batch_size * spatial_size;
    if (warp_id >= total_positions) return;
    
    int b = warp_id / spatial_size;
    int spatial_idx = warp_id % spatial_size;
    
    int base_idx = b * num_features * spatial_size + spatial_idx;
    int feature_stride = spatial_size;
    
    // Each thread sums a subset of features
    float local_sum_sq = 0.0f;
    for (int f = lane_id; f < num_features; f += WARP_SIZE) {
        float val = x[base_idx + f * feature_stride];
        local_sum_sq += val * val;
    }
    
    // Warp-level reduction
    float sum_sq = warp_reduce_sum(local_sum_sq);
    
    // Broadcast inv_rms from lane 0
    float inv_rms = rsqrtf(sum_sq * inv_num_features + eps);
    inv_rms = __shfl(inv_rms, 0);
    
    // Normalize - each thread handles its features
    for (int f = lane_id; f < num_features; f += WARP_SIZE) {
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
    
    int spatial_size = dim1 * dim2;
    int total_positions = batch_size * spatial_size;
    
    // Each warp handles one position, use 256 threads per block (4 warps)
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_blocks = (total_positions + warps_per_block - 1) / warps_per_block;
    
    float inv_num_features = 1.0f / (float)num_features;
    
    rmsnorm_warp_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_features,
        spatial_size,
        eps,
        inv_num_features
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
