import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

combined_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void gelu_scale_kernel(const float* x, float* out, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float cube = val * val * val;
        float tanh_arg = 0.7978845608028654f * (val + 0.044715f * cube);
        float gelu_val = val * 0.5f * (1.0f + tanhf(tanh_arg));
        out[idx] = gelu_val * scale;
    }
}

torch::Tensor gelu_scale_hip(torch::Tensor x, float scale) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    gelu_scale_kernel<<<num_blocks, block_size>>>(
        reinterpret_cast<const float*>(x.data_ptr<float>()),
        reinterpret_cast<float*>(out.data_ptr<float>()),
        size,
        scale
    );
    
    return out;
}

__global__ void max_reduce_kernel(const float* x, float* out, int batch_size, int num_features) {
    int batch_idx = blockIdx.x;
    __shared__ float shared_max[256];
    int local_idx = threadIdx.x;
    
    float max_val = -FLT_MAX;
    for (int i = local_idx; i < num_features; i += blockDim.x) {
        int idx = batch_idx * num_features + i;
        max_val = fmaxf(max_val, x[idx]);
    }
    shared_max[local_idx] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            shared_max[local_idx] = fmaxf(shared_max[local_idx], shared_max[local_idx + s]);
        }
        __syncthreads();
    }
    
    if (local_idx == 0) {
        out[batch_idx] = shared_max[0];
    }
}

torch::Tensor max_reduce_hip(torch::Tensor x) {
    int batch_size = x.size(0);
    int num_features = x.size(1);
    auto out = torch::zeros({batch_size}, x.options());
    
    max_reduce_kernel<<<batch_size, 256>>>(
        reinterpret_cast<const float*>(x.data_ptr<float>()),
        reinterpret_cast<float*>(out.data_ptr<float>()),
        batch_size,
        num_features
    );
    
    return out;
}
"""

kernels = load_inline(
    name="kernels",
    cpp_sources=combined_cpp_source,
    functions=["gelu_scale_hip", "max_reduce_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        
    def forward(self, x):
        x = self.matmul(x)
        x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        x = kernels.gelu_scale_hip(x, self.scale_factor)
        x = kernels.max_reduce_hip(x)
        return x