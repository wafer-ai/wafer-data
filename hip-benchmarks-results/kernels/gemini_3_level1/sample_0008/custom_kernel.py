
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set compiler to hipcc
os.environ["CXX"] = "hipcc"

cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>

__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int C,
    const int H,
    const int W,
    const float eps) {

    // We process the tensor as a grid of spatial locations (N, H, W).
    // Vectorize along W by 4 (float4).
    // Grid size corresponds to N * H * (W / 4).
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_w = W / 4;
    int spatial_size = N * H * stride_w;

    if (idx >= spatial_size) return;

    // Mapping idx to (n, h, w_vec)
    // idx = n * (H * stride_w) + h * stride_w + w_vec
    
    int w_vec = idx % stride_w;
    int tmp = idx / stride_w;
    int h = tmp % H;
    int n = tmp / H;
    
    int w = w_vec * 4;

    // Strides
    // x shape (N, C, H, W). Layout is contiguous N, C, H, W.
    // plane_stride (stride for C) = H * W
    // row_stride (stride for H) = W
    
    size_t plane_stride = (size_t)H * W;
    
    // Base offset for c=0 at (n, h, w)
    size_t base_offset = (size_t)n * C * plane_stride + (size_t)h * W + w;

    float4 sum_sq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Pass 1: Accumulate sum of squares over C
    // Memory access pattern: All threads in a warp access contiguous w, 
    // then jump by plane_stride together. This preserves coalescing.
    for (int c = 0; c < C; ++c) {
        float4 val = *reinterpret_cast<const float4*>(x + base_offset + c * plane_stride);
        sum_sq.x += val.x * val.x;
        sum_sq.y += val.y * val.y;
        sum_sq.z += val.z * val.z;
        sum_sq.w += val.w * val.w;
    }

    float4 rms;
    rms.x = rsqrtf(sum_sq.x / (float)C + eps);
    rms.y = rsqrtf(sum_sq.y / (float)C + eps);
    rms.z = rsqrtf(sum_sq.z / (float)C + eps);
    rms.w = rsqrtf(sum_sq.w / (float)C + eps);

    // Pass 2: Normalize and write output
    for (int c = 0; c < C; ++c) {
        float4 val = *reinterpret_cast<const float4*>(x + base_offset + c * plane_stride);
        float4 res;
        res.x = val.x * rms.x;
        res.y = val.y * rms.y;
        res.z = val.z * rms.z;
        res.w = val.w * rms.w;
        
        *reinterpret_cast<float4*>(out + base_offset + c * plane_stride) = res;
    }
}

torch::Tensor rms_norm_hip(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    // Ensure W is multiple of 4 for float4 optimization
    // We also require contiguous input for pointer arithmetic validity
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(W % 4 == 0, "Dimension W must be divisible by 4");

    int stride_w = W / 4;
    int spatial_size = N * H * stride_w;
    
    const int block_size = 256;
    const int num_blocks = (spatial_size + block_size - 1) / block_size;
    
    rms_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, eps
    );
    
    return out;
}
"""

rms_norm_module = load_inline(
    name="rms_norm_v1",
    cpp_sources=cpp_source,
    functions=["rms_norm_hip"],
    extra_cflags=['-O3'],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_op = rms_norm_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return self.rms_norm_op.rms_norm_hip(x, self.eps)

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
