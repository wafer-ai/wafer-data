
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gelu_scale_max_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <float.h>

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}

__global__ void gelu_scale_max_kernel(const float* input, float* output, int batch_size, int seq_len, float scale_factor) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    float max_val = -FLT_MAX;

    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = gelu(input[row * seq_len + i]) * scale_factor;
        if (val > max_val) max_val = val;
    }

    shared_data[tid] = max_val;
    __syncthreads();

    // Standard reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_data[tid + s] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = shared_data[0];
    }
}

torch::Tensor gelu_scale_max_hip(torch::Tensor input, float scale_factor) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    auto output = torch::empty({batch_size}, input.options());

    int block_size = 256;
    int shared_mem_size = block_size * sizeof(float);
    gelu_scale_max_kernel<<<batch_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, seq_len, scale_factor);

    return output;
}
"""

gelu_scale_max_lib = load_inline(
    name="gelu_scale_max",
    cpp_sources=gelu_scale_max_source,
    functions=["gelu_scale_max_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        self.register_buffer('W_new', None)
        self.register_buffer('b_new', None)
        self.precalculated = False

    def _precalculate(self, device):
        # Average weights and biases
        weight = self.matmul.weight
        bias = self.matmul.bias
        out_features, in_features = weight.shape
        
        W_new = weight.view(out_features // self.pool_kernel_size, self.pool_kernel_size, in_features).mean(dim=1)
        b_new = bias.view(out_features // self.pool_kernel_size, self.pool_kernel_size).mean(dim=1)
        
        self.W_new = W_new.to(device)
        self.b_new = b_new.to(device)
        self.precalculated = True

    def forward(self, x):
        if not self.precalculated:
            self._precalculate(x.device)
        
        # Optimized matmul
        x = F.linear(x, self.W_new, self.b_new)
        
        # Apply the fused GELU + Scale + Max kernel
        return gelu_scale_max_lib.gelu_scale_max_hip(x, float(self.scale_factor))

