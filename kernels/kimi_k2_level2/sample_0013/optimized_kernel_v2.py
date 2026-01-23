import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Custom fused kernel: avg_pool + gelu + scale + max - redesigned
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_pool_gelu_scale_max_kernel_v2(
    const float* matmul_out,
    float* final_out,
    int batch_size,
    int out_features,
    int pool_size,
    float scale
) {
    int batch_idx = blockIdx.x;
    int pool_idx = threadIdx.x;
    int num_pools = out_features / pool_size;  // 8192 / 16 = 512
    
    if (batch_idx >= batch_size || pool_idx >= num_pools) return;
    
    // Compute sum for this pool (each thread handles one pool)
    float sum = 0.0f;
    int start_idx = batch_idx * out_features + pool_idx * pool_size;
    
    #pragma unroll
    for (int i = 0; i < pool_size; i++) {
        sum += matmul_out[start_idx + i];
    }
    
    float avg = sum / pool_size;
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x_cubed = avg * avg * avg;
    float inner = 0.7978845608f * (avg + 0.044715f * x_cubed);
    float tanh_val = tanhf(inner);
    float gelu_val = 0.5f * avg * (1.0f + tanh_val);
    
    float scaled = gelu_val * scale;
    
    // Store in shared memory for max reduction
    extern __shared__ float shared[];
    shared[pool_idx] = scaled;
    
    __syncthreads();
    
    // Parallel reduction to find max (using all 512 threads)
    for (int s = num_pools / 2; s > 0; s >>= 1) {
        if (pool_idx < s) {
            shared[pool_idx] = fmaxf(shared[pool_idx], shared[pool_idx + s]);
        }
        __syncthreads();
    }
    
    if (pool_idx == 0) {
        final_out[batch_idx] = shared[0];
    }
}

torch::Tensor fused_pool_gelu_scale_max_v2(torch::Tensor matmul_out, float scale_factor) {
    auto batch_size = matmul_out.size(0);
    auto out_features = matmul_out.size(1);
    auto pool_size = 16;
    auto final_out = torch::zeros({batch_size}, matmul_out.options());
    
    const int threads = 512;  // Number of pools (8192 / 16 = 512)
    const int blocks = batch_size;
    size_t shared_mem = threads * sizeof(float);
    
    fused_pool_gelu_scale_max_kernel_v2<<<blocks, threads, shared_mem>>>(
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

fused_kernel_v2 = load_inline(
    name="fused_kernel_v2",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_pool_gelu_scale_max_v2"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_kernel_v2 = fused_kernel_v2
        
    def forward(self, x):
        x = self.matmul(x)  # Matmul + bias
        x = self.fused_kernel_v2.fused_pool_gelu_scale_max_v2(x, self.scale_factor)
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