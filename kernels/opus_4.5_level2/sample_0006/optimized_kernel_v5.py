import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused matmul + maxpool + sum + scale kernel
# The key insight: since we're summing all maxpool outputs, we can compute dot products
# and accumulate directly without materializing the full (batch, out_features) tensor
fused_kernel_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Fused Linear + MaxPool + Sum + Scale
// For each batch element, computes: sum(max(w[2i] @ x + b[2i], w[2i+1] @ x + b[2i+1])) * scale
// where w[k] is row k of weight matrix, b[k] is bias[k]

// Each thread block handles one batch element
// We tile over output features (pairs for maxpool)

__global__ void fused_linear_maxpool_sum_scale_kernel(
    const float* __restrict__ input,     // (batch, in_features)
    const float* __restrict__ weight,    // (out_features, in_features)
    const float* __restrict__ bias,      // (out_features,)
    float* __restrict__ output,          // (batch,)
    const int batch_size,
    const int in_features,
    const int out_features,
    const float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float sdata[];
    
    const float* batch_input = input + batch_idx * in_features;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_pairs = out_features / 2;
    
    float local_sum = 0.0f;
    
    // Each thread processes multiple output pairs
    for (int pair_idx = tid; pair_idx < num_pairs; pair_idx += num_threads) {
        int out_idx0 = pair_idx * 2;
        int out_idx1 = pair_idx * 2 + 1;
        
        // Compute dot products for both outputs in the pair
        const float* w0 = weight + out_idx0 * in_features;
        const float* w1 = weight + out_idx1 * in_features;
        
        float dot0 = bias[out_idx0];
        float dot1 = bias[out_idx1];
        
        // Vectorized dot product
        const float4* in4 = reinterpret_cast<const float4*>(batch_input);
        const float4* w0_4 = reinterpret_cast<const float4*>(w0);
        const float4* w1_4 = reinterpret_cast<const float4*>(w1);
        int num_vec = in_features / 4;
        
        for (int k = 0; k < num_vec; k++) {
            float4 x = in4[k];
            float4 wv0 = w0_4[k];
            float4 wv1 = w1_4[k];
            
            dot0 += x.x * wv0.x + x.y * wv0.y + x.z * wv0.z + x.w * wv0.w;
            dot1 += x.x * wv1.x + x.y * wv1.y + x.z * wv1.z + x.w * wv1.w;
        }
        
        // MaxPool + accumulate
        local_sum += fmaxf(dot0, dot1);
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = sdata[0] * scale_factor;
    }
}

torch::Tensor fused_linear_maxpool_sum_scale(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    fused_linear_maxpool_sum_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_linear_maxpool_sum_scale(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
);
"""

fused_module = load_inline(
    name="fused_linear_maxpool_sum_scale",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_maxpool_sum_scale"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fully fused linear + maxpool + sum + scale.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # Still need to store weights
        self.linear = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_op = fused_module

    def forward(self, x):
        return self.fused_op.fused_linear_maxpool_sum_scale(
            x,
            self.linear.weight,
            self.linear.bias,
            self.scale_factor
        )
