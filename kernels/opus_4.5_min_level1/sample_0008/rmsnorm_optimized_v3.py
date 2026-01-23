import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Further optimized RMSNorm kernel with:
// - Loop unrolling for feature dimension
// - Better register usage
// - Vectorized loads/stores with float4

__global__ void rmsnorm_kernel_optimized(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int num_features,
    int spatial_size,
    float eps,
    float inv_num_features
) {
    // Each thread handles 4 consecutive spatial positions
    int spatial_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int b = blockIdx.y;
    
    if (spatial_idx + 3 >= spatial_size || b >= batch_size) {
        // Handle edge cases with scalar code
        if (spatial_idx < spatial_size) {
            int feature_stride = spatial_size;
            for (int i = 0; i < 4 && spatial_idx + i < spatial_size; i++) {
                int pos = spatial_idx + i;
                int base = b * num_features * spatial_size + pos;
                
                float sum_sq = 0.0f;
                for (int f = 0; f < num_features; f++) {
                    float val = x[base + f * feature_stride];
                    sum_sq += val * val;
                }
                
                float inv_rms = rsqrtf(sum_sq * inv_num_features + eps);
                
                for (int f = 0; f < num_features; f++) {
                    int idx = base + f * feature_stride;
                    out[idx] = x[idx] * inv_rms;
                }
            }
        }
        return;
    }
    
    int feature_stride = spatial_size;
    int base_idx = b * num_features * spatial_size + spatial_idx;
    
    // Accumulate sum of squares
    float4 sum_sq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Unroll the feature loop by 4
    int f = 0;
    for (; f + 3 < num_features; f += 4) {
        float4 val0 = *reinterpret_cast<const float4*>(&x[base_idx + f * feature_stride]);
        float4 val1 = *reinterpret_cast<const float4*>(&x[base_idx + (f+1) * feature_stride]);
        float4 val2 = *reinterpret_cast<const float4*>(&x[base_idx + (f+2) * feature_stride]);
        float4 val3 = *reinterpret_cast<const float4*>(&x[base_idx + (f+3) * feature_stride]);
        
        sum_sq.x += val0.x * val0.x + val1.x * val1.x + val2.x * val2.x + val3.x * val3.x;
        sum_sq.y += val0.y * val0.y + val1.y * val1.y + val2.y * val2.y + val3.y * val3.y;
        sum_sq.z += val0.z * val0.z + val1.z * val1.z + val2.z * val2.z + val3.z * val3.z;
        sum_sq.w += val0.w * val0.w + val1.w * val1.w + val2.w * val2.w + val3.w * val3.w;
    }
    
    // Handle remaining features
    for (; f < num_features; f++) {
        float4 val = *reinterpret_cast<const float4*>(&x[base_idx + f * feature_stride]);
        sum_sq.x += val.x * val.x;
        sum_sq.y += val.y * val.y;
        sum_sq.z += val.z * val.z;
        sum_sq.w += val.w * val.w;
    }
    
    // Calculate inverse RMS for each position
    float inv_rms_x = rsqrtf(sum_sq.x * inv_num_features + eps);
    float inv_rms_y = rsqrtf(sum_sq.y * inv_num_features + eps);
    float inv_rms_z = rsqrtf(sum_sq.z * inv_num_features + eps);
    float inv_rms_w = rsqrtf(sum_sq.w * inv_num_features + eps);
    
    // Normalize with loop unrolling
    f = 0;
    for (; f + 3 < num_features; f += 4) {
        float4 val0 = *reinterpret_cast<const float4*>(&x[base_idx + f * feature_stride]);
        float4 val1 = *reinterpret_cast<const float4*>(&x[base_idx + (f+1) * feature_stride]);
        float4 val2 = *reinterpret_cast<const float4*>(&x[base_idx + (f+2) * feature_stride]);
        float4 val3 = *reinterpret_cast<const float4*>(&x[base_idx + (f+3) * feature_stride]);
        
        float4 res0 = make_float4(val0.x * inv_rms_x, val0.y * inv_rms_y, val0.z * inv_rms_z, val0.w * inv_rms_w);
        float4 res1 = make_float4(val1.x * inv_rms_x, val1.y * inv_rms_y, val1.z * inv_rms_z, val1.w * inv_rms_w);
        float4 res2 = make_float4(val2.x * inv_rms_x, val2.y * inv_rms_y, val2.z * inv_rms_z, val2.w * inv_rms_w);
        float4 res3 = make_float4(val3.x * inv_rms_x, val3.y * inv_rms_y, val3.z * inv_rms_z, val3.w * inv_rms_w);
        
        *reinterpret_cast<float4*>(&out[base_idx + f * feature_stride]) = res0;
        *reinterpret_cast<float4*>(&out[base_idx + (f+1) * feature_stride]) = res1;
        *reinterpret_cast<float4*>(&out[base_idx + (f+2) * feature_stride]) = res2;
        *reinterpret_cast<float4*>(&out[base_idx + (f+3) * feature_stride]) = res3;
    }
    
    // Handle remaining features
    for (; f < num_features; f++) {
        float4 val = *reinterpret_cast<const float4*>(&x[base_idx + f * feature_stride]);
        float4 res = make_float4(val.x * inv_rms_x, val.y * inv_rms_y, val.z * inv_rms_z, val.w * inv_rms_w);
        *reinterpret_cast<float4*>(&out[base_idx + f * feature_stride]) = res;
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
    
    float inv_num_features = 1.0f / (float)num_features;
    
    rmsnorm_kernel_optimized<<<grid, block>>>(
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
