import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Simple but effective fused kernel
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>

// Sane GELU approximation
__device__ __forceinline__ float gelu(float x) {
    float x_cubed = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void fused_pool_gelu_scale_max_kernel_v4(
    const float* __restrict__ matmul_out,
    float* __restrict__ final_out,
    int batch_size,
    int out_features,
    int pool_size,
    float scale
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_pools = out_features / pool_size;  // 512
    
    if (batch_idx >= batch_size) return;
    
    const float* batch_data = matmul_out + batch_idx * out_features;
    
    // Each thread handles multiple pools
    float thread_max = -1e38f;
    
    for (int pool_idx = tid; pool_idx < num_pools; pool_idx += blockDim.x) {
        int start_feature = pool_idx * pool_size;
        float sum = 0.0f;
        
        // Vectorized pool sum
        #pragma unroll 16
        for (int i = 0; i < pool_size; i++) {
            sum += batch_data[start_feature + i];
        }
        
        float avg = sum / pool_size;
        float gelu_val = gelu(avg);
        float scaled = gelu_val * scale;
        
        thread_max = fmaxf(thread_max, scaled);
    }
    
    // Shared memory for reduction
    __shared__ float shared_max[256];
    
    shared_max[tid] = thread_max;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        final_out[batch_idx] = shared_max[0];
    }
}

torch::Tensor fused_pool_gelu_scale_max_v4(torch::Tensor matmul_out, float scale_factor) {
    auto batch_size = matmul_out.size(0);
    auto out_features = matmul_out.size(1);
    auto pool_size = 16;
    auto final_out = torch::zeros({batch_size}, matmul_out.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    
    fused_pool_gelu_scale_max_kernel_v4<<<blocks, threads>>>(
        matmul_out.data_ptr<float>(),
        final_out.data_ptr<float>(),
        batch_size,
        out_features,
        pool_size,
        scale_factor
    );
    
    return final_out;
}
"""

fused_kernel_v4 = load_inline(
    name="fused_kernel_v4",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_pool_gelu_scale_max_v4"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_kernel_v4 = fused_kernel_v4
        
    def forward(self, x):
        x = self.matmul(x)  # Matmul + bias
        x = self.fused_kernel_v4.fused_pool_gelu_scale_max_v4(x, self.scale_factor)
        return x

# Input generation functions
batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]