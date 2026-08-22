import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Ensure HIP compiler is used
os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.79788456f; 
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x3)));
}

__global__ void __launch_bounds__(256) gelu_vec_inplace_kernel(
    float* __restrict__ in_out,
    int total_vecs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vecs) {
        float4* io_vec = reinterpret_cast<float4*>(in_out);
        float4 v = io_vec[idx];
        
        v.x = gelu_tanh(v.x);
        v.y = gelu_tanh(v.y);
        v.z = gelu_tanh(v.z);
        v.w = gelu_tanh(v.w);
        
        io_vec[idx] = v;
    }
}

__global__ void gelu_inplace_kernel(
    float* __restrict__ in_out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        in_out[idx] = gelu_tanh(in_out[idx]);
    }
}

void gelu_hip_inplace(torch::Tensor input) {
    int rows = input.size(0);
    int cols = input.size(1);
    int total_elements = rows * cols;

    bool aligned = (cols % 4 == 0) &&
                   (reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0);

    if (aligned) {
        int total_vecs = total_elements / 4;
        const int block_size = 256;
        const int num_blocks = (total_vecs + block_size - 1) / block_size;
        
        gelu_vec_inplace_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            total_vecs
        );
    } else {
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        gelu_inplace_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            total_elements
        );
    }
}
"""

gelu_op = load_inline(
    name="gelu_op_tanh",
    cpp_sources=cpp_source,
    functions=["gelu_hip_inplace"],
    verbose=False,
    extra_cflags=['-O3', '-ffast-math']
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        
        # Optimization 1: Fold division into weights and bias
        with torch.no_grad():
            self.linear.weight.div_(divisor)
            self.linear.bias.div_(divisor)
            
        self.gelu_op = gelu_op
        self.graphed_forward = None

    def _forward_impl(self, x):
        # Optimization 2: Linear + In-place Custom GELU
        y = self.linear(x)
        self.gelu_op.gelu_hip_inplace(y)
        return y

    def forward(self, x):
        # Optimization 3: CUDA Graph
        if self.graphed_forward is None:
             self.graphed_forward = torch.cuda.make_graphed_callables(
                 self._forward_impl, (x,)
             )
        
        return self.graphed_forward(x)

def get_inputs():
    batch_size = 1024
    input_size = 8192
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    input_size = 8192
    output_size = 8192
    divisor = 10.0
    return [input_size, output_size, divisor]
