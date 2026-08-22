import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Highly optimized RMSNorm kernel with:
// - float4 vectorization 
// - Fully unrolled feature loop (num_features=64)
// - Pre-computed constants
// Input shape: (batch_size, 64, dim1, dim2)

__global__ void rmsnorm_kernel_f64(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int spatial_size,
    float eps,
    float inv_64
) {
    // Each thread handles 4 consecutive spatial positions
    int spatial_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int b = blockIdx.y;
    
    if (spatial_idx + 3 >= spatial_size || b >= batch_size) {
        // Handle edge cases with scalar code
        if (spatial_idx < spatial_size) {
            const int num_features = 64;
            int feature_stride = spatial_size;
            for (int i = 0; i < 4 && spatial_idx + i < spatial_size; i++) {
                int pos = spatial_idx + i;
                int base = b * num_features * spatial_size + pos;
                
                float sum_sq = 0.0f;
                #pragma unroll
                for (int f = 0; f < 64; f++) {
                    float val = x[base + f * feature_stride];
                    sum_sq += val * val;
                }
                
                float inv_rms = rsqrtf(sum_sq * inv_64 + eps);
                
                #pragma unroll
                for (int f = 0; f < 64; f++) {
                    int idx = base + f * feature_stride;
                    out[idx] = x[idx] * inv_rms;
                }
            }
        }
        return;
    }
    
    const int num_features = 64;
    int feature_stride = spatial_size;
    int base_idx = b * num_features * spatial_size + spatial_idx;
    
    // Accumulate sum of squares using float4
    float4 sum_sq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Fully unroll the feature loop - 64 features, 8 iterations of 8
    #pragma unroll 8
    for (int f = 0; f < 64; f += 8) {
        float4 v0 = *reinterpret_cast<const float4*>(&x[base_idx + f * feature_stride]);
        float4 v1 = *reinterpret_cast<const float4*>(&x[base_idx + (f+1) * feature_stride]);
        float4 v2 = *reinterpret_cast<const float4*>(&x[base_idx + (f+2) * feature_stride]);
        float4 v3 = *reinterpret_cast<const float4*>(&x[base_idx + (f+3) * feature_stride]);
        float4 v4 = *reinterpret_cast<const float4*>(&x[base_idx + (f+4) * feature_stride]);
        float4 v5 = *reinterpret_cast<const float4*>(&x[base_idx + (f+5) * feature_stride]);
        float4 v6 = *reinterpret_cast<const float4*>(&x[base_idx + (f+6) * feature_stride]);
        float4 v7 = *reinterpret_cast<const float4*>(&x[base_idx + (f+7) * feature_stride]);
        
        sum_sq.x += v0.x*v0.x + v1.x*v1.x + v2.x*v2.x + v3.x*v3.x + v4.x*v4.x + v5.x*v5.x + v6.x*v6.x + v7.x*v7.x;
        sum_sq.y += v0.y*v0.y + v1.y*v1.y + v2.y*v2.y + v3.y*v3.y + v4.y*v4.y + v5.y*v5.y + v6.y*v6.y + v7.y*v7.y;
        sum_sq.z += v0.z*v0.z + v1.z*v1.z + v2.z*v2.z + v3.z*v3.z + v4.z*v4.z + v5.z*v5.z + v6.z*v6.z + v7.z*v7.z;
        sum_sq.w += v0.w*v0.w + v1.w*v1.w + v2.w*v2.w + v3.w*v3.w + v4.w*v4.w + v5.w*v5.w + v6.w*v6.w + v7.w*v7.w;
    }
    
    // Calculate inverse RMS
    float inv_rms_x = rsqrtf(sum_sq.x * inv_64 + eps);
    float inv_rms_y = rsqrtf(sum_sq.y * inv_64 + eps);
    float inv_rms_z = rsqrtf(sum_sq.z * inv_64 + eps);
    float inv_rms_w = rsqrtf(sum_sq.w * inv_64 + eps);
    
    // Normalize - fully unrolled
    #pragma unroll 8
    for (int f = 0; f < 64; f += 8) {
        float4 v0 = *reinterpret_cast<const float4*>(&x[base_idx + f * feature_stride]);
        float4 v1 = *reinterpret_cast<const float4*>(&x[base_idx + (f+1) * feature_stride]);
        float4 v2 = *reinterpret_cast<const float4*>(&x[base_idx + (f+2) * feature_stride]);
        float4 v3 = *reinterpret_cast<const float4*>(&x[base_idx + (f+3) * feature_stride]);
        float4 v4 = *reinterpret_cast<const float4*>(&x[base_idx + (f+4) * feature_stride]);
        float4 v5 = *reinterpret_cast<const float4*>(&x[base_idx + (f+5) * feature_stride]);
        float4 v6 = *reinterpret_cast<const float4*>(&x[base_idx + (f+6) * feature_stride]);
        float4 v7 = *reinterpret_cast<const float4*>(&x[base_idx + (f+7) * feature_stride]);
        
        *reinterpret_cast<float4*>(&out[base_idx + f * feature_stride]) = 
            make_float4(v0.x*inv_rms_x, v0.y*inv_rms_y, v0.z*inv_rms_z, v0.w*inv_rms_w);
        *reinterpret_cast<float4*>(&out[base_idx + (f+1) * feature_stride]) = 
            make_float4(v1.x*inv_rms_x, v1.y*inv_rms_y, v1.z*inv_rms_z, v1.w*inv_rms_w);
        *reinterpret_cast<float4*>(&out[base_idx + (f+2) * feature_stride]) = 
            make_float4(v2.x*inv_rms_x, v2.y*inv_rms_y, v2.z*inv_rms_z, v2.w*inv_rms_w);
        *reinterpret_cast<float4*>(&out[base_idx + (f+3) * feature_stride]) = 
            make_float4(v3.x*inv_rms_x, v3.y*inv_rms_y, v3.z*inv_rms_z, v3.w*inv_rms_w);
        *reinterpret_cast<float4*>(&out[base_idx + (f+4) * feature_stride]) = 
            make_float4(v4.x*inv_rms_x, v4.y*inv_rms_y, v4.z*inv_rms_z, v4.w*inv_rms_w);
        *reinterpret_cast<float4*>(&out[base_idx + (f+5) * feature_stride]) = 
            make_float4(v5.x*inv_rms_x, v5.y*inv_rms_y, v5.z*inv_rms_z, v5.w*inv_rms_w);
        *reinterpret_cast<float4*>(&out[base_idx + (f+6) * feature_stride]) = 
            make_float4(v6.x*inv_rms_x, v6.y*inv_rms_y, v6.z*inv_rms_z, v6.w*inv_rms_w);
        *reinterpret_cast<float4*>(&out[base_idx + (f+7) * feature_stride]) = 
            make_float4(v7.x*inv_rms_x, v7.y*inv_rms_y, v7.z*inv_rms_z, v7.w*inv_rms_w);
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
    
    float inv_64 = 1.0f / 64.0f;
    
    rmsnorm_kernel_f64<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        spatial_size,
        eps,
        inv_64
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
