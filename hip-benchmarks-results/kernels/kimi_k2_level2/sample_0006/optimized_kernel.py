import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: matmul + maxpool + sum + scale
cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// Fused kernel: linear + maxpool + sum + scale
// Performs: output = scale * sum(maxpool(linear(x)))
__global__ void fused_kernel(
    const float* __restrict__ x,      // input: (batch_size, in_features)
    const float* __restrict__ W,      // weight: (out_features, in_features)
    const float* __restrict__ bias,   // bias: (out_features,)
    float* __restrict__ output,       // output: (batch_size,)
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,                  // maxpool kernel size (2)
    float scale_factor
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ float shared_sum[BLOCK_SIZE];
    float thread_sum = 0.0f;
    
    // Number of maxpool groups = out_features / kernel_size
    int num_groups = out_features / kernel_size;
    
    // Each thread processes multiple maxpool groups
    for (int group = tid; group < num_groups; group += blockDim.x) {
        float max_val = -1e30f;  // Very low initial value
        
        // Process each element in the maxpool group
        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            int out_idx = group * kernel_size + k;
            
            // Compute linear output: out = dot(x, W[out_idx]) + bias[out_idx]
            float dot = (bias != nullptr) ? bias[out_idx] : 0.0f;
            
            // Vectorized dot product: process 4 elements at a time
            int i = 0;
            for (; i <= in_features - 4; i += 4) {
                float4 x_vec = *reinterpret_cast<const float4*>(&x[batch_idx * in_features + i]);
                float4 w_vec = *reinterpret_cast<const float4*>(&W[out_idx * in_features + i]);
                
                dot += x_vec.x * w_vec.x;
                dot += x_vec.y * w_vec.y;
                dot += x_vec.z * w_vec.z;
                dot += x_vec.w * w_vec.w;
            }
            
            // Handle remaining elements
            for (; i < in_features; i++) {
                dot += x[batch_idx * in_features + i] * W[out_idx * in_features + i];
            }
            
            // Update max value for this group
            max_val = fmaxf(max_val, dot);
        }
        
        // Accumulate the max value for this group
        thread_sum += max_val;
    }
    
    // Store thread contribution in shared memory
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write final result for this batch element
    if (tid == 0) {
        output[batch_idx] = shared_sum[0] * scale_factor;
    }
}

torch::Tensor fused_forward(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor bias,
    float scale_factor,
    int kernel_size
) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = W.size(0);
    
    // Allocate output tensor of shape (batch_size,)
    auto output = torch::zeros({batch_size}, x.options());
    
    // Launch one block per batch element
    dim3 blocks(batch_size);
    dim3 threads(BLOCK_SIZE);
    
    fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_op = load_inline(
    name="fused_kernel",
    cpp_sources=cpp_source,
    functions=["fused_forward"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_op = fused_op
        # Use PyTorch default initialization (same as reference)
    
    def forward(self, x):
        # Fused forward: linear + maxpool + sum + scale
        return self.fused_op.fused_forward(
            x, self.linear.weight, self.linear.bias,
            self.scale_factor, self.kernel_size
        )


batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
