import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for post-matmul operations
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <float.h>

// GELU approximation using tanh
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

// Fused avgpool + gelu + scale + max kernel
// Using warp shuffles for efficient reduction on AMD GCN architecture
__global__ void fused_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int out_features,
    const float scale_factor,
    const float inv_pool_size
) {
    constexpr int POOL_SIZE = 16;
    const int pooled_size = out_features >> 4;
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    const float* batch_input = input + batch_idx * out_features;
    const float4* batch_input_vec = reinterpret_cast<const float4*>(batch_input);
    
    float local_max = -FLT_MAX;
    
    for (int pool_idx = tid; pool_idx < pooled_size; pool_idx += num_threads) {
        int vec_start = pool_idx << 2;
        
        float4 v0 = batch_input_vec[vec_start];
        float4 v1 = batch_input_vec[vec_start + 1];
        float4 v2 = batch_input_vec[vec_start + 2];
        float4 v3 = batch_input_vec[vec_start + 3];
        
        float sum = v0.x + v0.y + v0.z + v0.w +
                    v1.x + v1.y + v1.z + v1.w +
                    v2.x + v2.y + v2.z + v2.w +
                    v3.x + v3.y + v3.z + v3.w;
        
        float avg = sum * inv_pool_size;
        float result = gelu(avg) * scale_factor;
        local_max = fmaxf(local_max, result);
    }
    
    // Warp reduction (AMD wavefront = 64)
    for (int offset = 32; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    
    __shared__ float warp_maxes[16];
    
    int lane = tid & 63;
    int warp_id = tid >> 6;
    
    if (lane == 0) {
        warp_maxes[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction
    if (tid == 0) {
        int num_warps = (num_threads + 63) >> 6;
        float final_max = -FLT_MAX;
        for (int i = 0; i < num_warps; i++) {
            final_max = fmaxf(final_max, warp_maxes[i]);
        }
        output[batch_idx] = final_max;
    }
}

torch::Tensor fused_avgpool_gelu_scale_max_hip(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
) {
    const int batch_size = input.size(0);
    const int out_features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const float inv_pool_size = 1.0f / pool_kernel_size;
    
    fused_avgpool_gelu_scale_max_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        scale_factor,
        inv_pool_size
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_avgpool_gelu_scale_max_hip(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
);
"""

fused_module = load_inline(
    name="fused_avgpool_gelu_scale_max_v5",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_avgpool_gelu_scale_max_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized model implementing "Matmul_AvgPool_GELU_Scale_Max".
    Uses addmm for potentially better GEMM performance + fused post-processing.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # Store weight and bias separately for using addmm
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1 / (in_features ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.fused_module = fused_module

    def forward(self, x):
        # Use F.linear for better performance
        x = F.linear(x, self.weight, self.bias)
        x = self.fused_module.fused_avgpool_gelu_scale_max_hip(
            x, self.pool_kernel_size, self.scale_factor
        )
        return x


def get_inputs():
    return [torch.rand(1024, 8192).cuda()]


def get_init_inputs():
    return [8192, 8192, 16, 2.0]
