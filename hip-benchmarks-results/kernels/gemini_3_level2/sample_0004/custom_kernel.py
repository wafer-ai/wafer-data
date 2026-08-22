
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set compiler to hipcc
os.environ["CXX"] = "hipcc"

cuda_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void fused_ops_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int C,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int pool_k,
    const float sub1,
    const float sub2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;

    if (idx < total_elements) {
        int w_out = idx % W_out;
        int tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp = tmp / H_out;
        int c = tmp % C;
        int n = tmp / C;

        float sum = 0.0f;
        int input_base = n * (C * H_in * W_in) + c * (H_in * W_in);

        // Optimization for pool_k=2 and even width (alignment)
        // We ensure 8-byte alignment for float2 loads.
        // H_in*W_in is even -> channel offset is aligned.
        // w_start is even -> pixel offset is aligned.
        if (pool_k == 2 && (W_in & 1) == 0) {
             int h_start = h_out * 2;
             int w_start = w_out * 2;
             
             // Row 0
             int offset0 = input_base + h_start * W_in + w_start;
             // Use float2 to load 2 elements at once
             float2 v0 = *reinterpret_cast<const float2*>(&input[offset0]);
             
             sum += tanhf(v0.x - sub1) - sub2;
             sum += tanhf(v0.y - sub1) - sub2;
             
             // Row 1
             int offset1 = input_base + (h_start + 1) * W_in + w_start;
             float2 v1 = *reinterpret_cast<const float2*>(&input[offset1]);
             
             sum += tanhf(v1.x - sub1) - sub2;
             sum += tanhf(v1.y - sub1) - sub2;
             
             output[idx] = sum * 0.25f;
        } else {
            // Generic path
            int h_start = h_out * pool_k;
            int w_start = w_out * pool_k;

            for (int i = 0; i < pool_k; ++i) {
                for (int j = 0; j < pool_k; ++j) {
                    int h_in = h_start + i;
                    int w_in = w_start + j;

                    if (h_in < H_in && w_in < W_in) {
                        int input_idx = input_base + h_in * W_in + w_in;
                        float val = input[input_idx];
                        val = tanhf(val - sub1) - sub2;
                        sum += val;
                    }
                }
            }
            float count = (float)(pool_k * pool_k);
            output[idx] = sum / count;
        }
    }
}

torch::Tensor fused_ops_hip(torch::Tensor input, float sub1, float sub2, int pool_k) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int H_out = H_in / pool_k;
    const int W_out = W_in / pool_k;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    int total_elements = N * C * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out, pool_k, sub1, sub2
    );

    return output;
}
"""

cpp_source = "torch::Tensor fused_ops_hip(torch::Tensor input, float sub1, float sub2, int pool_k);"

fused_ops = load_inline(
    name="fused_ops_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_ops_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_hip(x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 64
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    subtract1_value = 0.5
    subtract2_value = 0.2
    kernel_size_pool = 2
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
