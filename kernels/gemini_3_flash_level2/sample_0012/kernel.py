
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_ops_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ inline float gelu_func(float x) {
    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
}

__global__ void vectorized_bias_div_gelu_kernel(float* x, const float* bias, float inv_divisor, int rows, int cols) {
    int row = blockIdx.y;
    int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (col_base < cols) {
        int idx = row * cols + col_base;
        float4 x_vec = reinterpret_cast<float4*>(&x[idx])[0];
        float4 bias_vec = reinterpret_cast<const float4*>(&bias[col_base])[0];
        
        x_vec.x = gelu_func((x_vec.x + bias_vec.x) * inv_divisor);
        x_vec.y = gelu_func((x_vec.y + bias_vec.y) * inv_divisor);
        x_vec.z = gelu_func((x_vec.z + bias_vec.z) * inv_divisor);
        x_vec.w = gelu_func((x_vec.w + bias_vec.w) * inv_divisor);
        
        reinterpret_cast<float4*>(&x[idx])[0] = x_vec;
    }
}

void bias_div_gelu_hip(torch::Tensor x, torch::Tensor bias, float inv_divisor) {
    int rows = x.size(0);
    int cols = x.size(1);
    const int threads_per_block = 256;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim((cols / 4 + threads_per_block - 1) / threads_per_block, rows);
    
    vectorized_bias_div_gelu_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        inv_divisor, 
        rows, 
        cols
    );
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    functions=["bias_div_gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # Pre-transpose and make contiguous to optimize the gemm call
        self.weight_t = nn.Parameter(self.linear.weight.t().contiguous())
        self.bias = self.linear.bias
        self.divisor = divisor
        self.inv_divisor = 1.0 / divisor
        self.fused_ops = fused_ops

    def forward(self, x):
        # We found that using torch.mm with a pre-transposed contiguous weight 
        # and then fusing the rest into a custom kernel is very close to the reference performance.
        # Given that GEMM is the primary bottleneck, this fusion is the most we can do
        # without writing a highly optimized custom GEMM kernel.
        res = torch.mm(x, self.weight_t)
        self.fused_ops.bias_div_gelu_hip(res, self.bias, self.inv_divisor)
        return res

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, output_size, divisor]
