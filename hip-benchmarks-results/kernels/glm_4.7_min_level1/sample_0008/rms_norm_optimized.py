import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rms_norm_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void rms_norm_kernel(const float* __restrict__ x, 
                                float* __restrict__ out, 
                                int batch_size,
                                int features,
                                int dim1,
                                int dim2,
                                float eps) {
    // Process each spatial position (batch, d1, d2) in parallel
    int total_spatial = batch_size * dim1 * dim2;
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (spatial_idx >= total_spatial) return;
    
    // Convert to spatial coordinates
    int batch_idx = spatial_idx / (dim1 * dim2);
    int d1_d2_idx = spatial_idx % (dim1 * dim2);
    int d1_idx = d1_d2_idx / dim2;
    int d2_idx = d1_d2_idx % dim2;
    
    // Compute sum of squares across features
    float sum_sq = 0.0f;
    const int spatial_stride = dim1 * dim2;
    const int feature_stride = spatial_stride;
    const int batch_stride = features * feature_stride;
    
    int base_offset = batch_idx * batch_stride + d1_idx * dim2 + d2_idx;
    
    // Use vectorized loads for better memory bandwidth (load 4 floats at a time)
    const int vec_size = 4;
    const int features_vec = features / vec_size;
    
    for (int fv = 0; fv < features_vec; fv++) {
        int idx = base_offset + fv * vec_size * feature_stride;
        // Load 4 floats using reinterpret_cast
        const float4* x_vec = reinterpret_cast<const float4*>(&x[idx]);
        float4 vals = __ldg(x_vec);
        
        sum_sq += vals.x * vals.x;
        sum_sq += vals.y * vals.y;
        sum_sq += vals.z * vals.z;
        sum_sq += vals.w * vals.w;
    }
    
    // Handle remaining elements
    for (int f = features_vec * vec_size; f < features; f++) {
        int idx = base_offset + f * feature_stride;
        float val = x[idx];
        sum_sq += val * val;
    }
    
    // Compute RMS
    float mean_sq = sum_sq / (float)features;
    float rms_val = sqrtf(mean_sq + eps);
    float inv_rms = 1.0f / rms_val;
    
    // Write normalized output with vectorized stores
    for (int fv = 0; fv < features_vec; fv++) {
        int idx = base_offset + fv * vec_size * feature_stride;
        // Load, multiply by inv_rms, and store
        const float4* x_vec = reinterpret_cast<const float4*>(&x[idx]);
        float4 vals = __ldg(x_vec);
        
        vals.x *= inv_rms;
        vals.y *= inv_rms;
        vals.z *= inv_rms;
        vals.w *= inv_rms;
        
        float4* out_vec = reinterpret_cast<float4*>(&out[idx]);
        *out_vec = vals;
    }
    
    // Handle remaining elements
    for (int f = features_vec * vec_size; f < features; f++) {
        int idx = base_offset + f * feature_stride;
        out[idx] = x[idx] * inv_rms;
    }
}

torch::Tensor rms_norm_hip(torch::Tensor x, float eps) {
    auto batch_size = x.size(0);
    auto features = x.size(1);
    auto dim1 = x.size(2);
    auto dim2 = x.size(3);
    
    auto out = torch::zeros_like(x);
    
    int total_spatial = batch_size * dim1 * dim2;
    const int block_size = 256;
    const int num_blocks = (total_spatial + block_size - 1) / block_size;
    
    rms_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        features,
        dim1,
        dim2,
        eps
    );
    
    return out;
}
"""

rms_norm_module = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    functions=["rms_norm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization with optimized HIP kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_module = rms_norm_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using optimized HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return self.rms_norm_module.rms_norm_hip(x, self.eps)