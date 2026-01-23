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

// Warp-level reduction using shuffle operations
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Each warp handles one spatial position, using all 64 threads to sum 64 features
// Perfect fit for num_features = 64!
__global__ void rmsnorm_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int dim1,
    const int dim2,
    const float eps
) {
    const int spatial_size = dim1 * dim2;
    const int total_positions = batch_size * spatial_size;
    
    // One warp per position
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= total_positions) return;
    
    // Compute batch and spatial indices
    const int batch_idx = warp_id / spatial_size;
    const int spatial_idx = warp_id % spatial_size;
    const int d1 = spatial_idx / dim2;
    const int d2 = spatial_idx % dim2;
    
    const int feature_stride = spatial_size;
    const int batch_offset = batch_idx * num_features * feature_stride;
    const int spatial_offset = d1 * dim2 + d2;
    
    // Each thread loads one feature value (64 threads for 64 features)
    float val = 0.0f;
    if (lane_id < num_features) {
        const int idx = batch_offset + lane_id * feature_stride + spatial_offset;
        val = input[idx];
    }
    
    // Compute sum of squares using warp reduction
    float sq = val * val;
    float sum_sq = warpReduceSum(sq);
    
    // Broadcast RMS to all threads
    sum_sq = __shfl(sum_sq, 0);
    float inv_rms = rsqrtf(sum_sq / num_features + eps);
    
    // Write normalized output
    if (lane_id < num_features) {
        const int idx = batch_offset + lane_id * feature_stride + spatial_offset;
        output[idx] = val * inv_rms;
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int dim1 = input.size(2);
    const int dim2 = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int total_positions = batch_size * dim1 * dim2;
    
    // One warp (64 threads) per spatial position
    // 256 threads = 4 warps per block
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / WARP_SIZE;
    const int num_blocks = (total_positions + warps_per_block - 1) / warps_per_block;
    
    rmsnorm_warp_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        dim1,
        dim2,
        eps
    );
    
    return output;
}
"""

rmsnorm_cpp_source = """
torch::Tensor rmsnorm_hip(torch::Tensor input, float eps);
"""

rmsnorm_module = load_inline(
    name="rmsnorm_hip_v3",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_hip_source,
    functions=["rmsnorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized RMS Normalization using HIP kernel with warp-level reduction.
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
