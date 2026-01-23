
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_ops_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
}

__global__ void post_matmul_fused_kernel(
    const float* __restrict__ input, // shape: (batch_size, num_pooled)
    float* __restrict__ output,       // shape: (batch_size,)
    int batch_size,
    int num_pooled,
    float scale_factor) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared_data[];

    float max_val = -1e20f;
    for (int i = threadIdx.x; i < num_pooled; i += blockDim.x) {
        float val = gelu(input[row * num_pooled + i]) * scale_factor;
        if (val > max_val) max_val = val;
    }

    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = shared_data[0];
    }
}

torch::Tensor post_matmul_fused_hip(torch::Tensor input, float scale_factor) {
    int batch_size = input.size(0);
    int num_pooled = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());

    int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    int shared_mem_size = block_size * sizeof(float);

    post_matmul_fused_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_pooled,
        scale_factor
    );

    return output;
}
"""

post_matmul_fused_lib = load_inline(
    name="post_matmul_fused_lib",
    cpp_sources=fused_ops_cpp_source,
    functions=["post_matmul_fused_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        
        self.matmul = nn.Linear(in_features, out_features)
        self.post_matmul_fused_lib = post_matmul_fused_lib
        
        # We'll use these to cache the pooled weight/bias
        self.register_buffer('weight_pooled', None)
        self.register_buffer('bias_pooled', None)

    def _initialize_pooled_weights(self):
        # pool the weights and bias
        with torch.no_grad():
            w = self.matmul.weight
            b = self.matmul.bias
            num_pooled = self.out_features // self.pool_kernel_size
            
            # W_pooled: (num_pooled, in_features)
            wp = w[:num_pooled * self.pool_kernel_size, :].view(
                num_pooled, self.pool_kernel_size, self.in_features
            ).mean(dim=1)
            self.weight_pooled = wp
            
            if b is not None:
                bp = b[:num_pooled * self.pool_kernel_size].view(
                    num_pooled, self.pool_kernel_size
                ).mean(dim=1)
                self.bias_pooled = bp
            else:
                self.bias_pooled = None

    def forward(self, x):
        if self.weight_pooled is None:
            self._initialize_pooled_weights()
        
        # New matmul: (batch_size, in_features) @ (in_features, num_pooled)
        # weight_pooled is (num_pooled, in_features), so we transpose it.
        # Ensure x and weights are on the same device and use same dtype
        x = torch.matmul(x, self.weight_pooled.t())
        if self.bias_pooled is not None:
            x = x + self.bias_pooled
            
        return self.post_matmul_fused_lib.post_matmul_fused_hip(x, self.scale_factor)
