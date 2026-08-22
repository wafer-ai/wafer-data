import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Optimized RMSNorm kernel using vectorized loads (float4)
// Input shape: (batch_size, num_features, dim1, dim2)
// Reduce along dim=1 (num_features)

__global__ void rmsnorm_kernel_vec4(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int num_features,
    int dim1,
    int dim2,
    float eps
) {
    // Each thread handles 4 consecutive spatial positions
    int spatial_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int b = blockIdx.y;
    
    int spatial_size = dim1 * dim2;
    
    if (spatial_idx >= spatial_size || b >= batch_size) return;
    
    int feature_stride = spatial_size;
    int base_idx = b * num_features * spatial_size + spatial_idx;
    
    // Handle edge case where spatial_idx + 4 > spatial_size
    int num_elements = min(4, spatial_size - spatial_idx);
    
    if (num_elements == 4) {
        // Full vector case
        float4 sum_sq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        // Calculate sum of squares across features
        for (int f = 0; f < num_features; f++) {
            int idx = base_idx + f * feature_stride;
            float4 val = *reinterpret_cast<const float4*>(&x[idx]);
            sum_sq.x += val.x * val.x;
            sum_sq.y += val.y * val.y;
            sum_sq.z += val.z * val.z;
            sum_sq.w += val.w * val.w;
        }
        
        // Calculate RMS for each position
        float inv_rms_x = rsqrtf(sum_sq.x / (float)num_features + eps);
        float inv_rms_y = rsqrtf(sum_sq.y / (float)num_features + eps);
        float inv_rms_z = rsqrtf(sum_sq.z / (float)num_features + eps);
        float inv_rms_w = rsqrtf(sum_sq.w / (float)num_features + eps);
        
        // Normalize
        for (int f = 0; f < num_features; f++) {
            int idx = base_idx + f * feature_stride;
            float4 val = *reinterpret_cast<const float4*>(&x[idx]);
            float4 result;
            result.x = val.x * inv_rms_x;
            result.y = val.y * inv_rms_y;
            result.z = val.z * inv_rms_z;
            result.w = val.w * inv_rms_w;
            *reinterpret_cast<float4*>(&out[idx]) = result;
        }
    } else {
        // Scalar fallback for edge cases
        for (int i = 0; i < num_elements; i++) {
            int pos = spatial_idx + i;
            int base = b * num_features * spatial_size + pos;
            
            float sum_sq = 0.0f;
            for (int f = 0; f < num_features; f++) {
                float val = x[base + f * feature_stride];
                sum_sq += val * val;
            }
            
            float inv_rms = rsqrtf(sum_sq / (float)num_features + eps);
            
            for (int f = 0; f < num_features; f++) {
                int idx = base + f * feature_stride;
                out[idx] = x[idx] * inv_rms;
            }
        }
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
    int threads = 256;
    int blocks_x = (spatial_size / 4 + threads - 1) / threads;
    
    dim3 grid(blocks_x, batch_size);
    dim3 block(threads);
    
    rmsnorm_kernel_vec4<<<grid, block>>>(
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
