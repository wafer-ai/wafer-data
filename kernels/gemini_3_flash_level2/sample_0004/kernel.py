
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void __launch_bounds__(256) fused_post_conv_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_h, int in_w,
    int out_h, int out_w,
    int pool_k,
    float s2,
    int total_output_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output_elements) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int nc = idx / (out_w * out_h);

    float sum = 0.0f;
    int n_c_offset = nc * in_h * in_w;
    int in_row_start = oh * pool_k;
    int in_col_start = ow * pool_k;

    if (pool_k == 2) {
        int offset1 = n_c_offset + in_row_start * in_w + in_col_start;
        int offset2 = offset1 + in_w;
        float2 row1 = reinterpret_cast<const float2*>(input + offset1)[0];
        float2 row2 = reinterpret_cast<const float2*>(input + offset2)[0];
        sum += tanhf(row1.x) - s2;
        sum += tanhf(row1.y) - s2;
        sum += tanhf(row2.x) - s2;
        sum += tanhf(row2.y) - s2;
    } else {
        for (int i = 0; i < pool_k; ++i) {
            int ih = in_row_start + i;
            for (int j = 0; j < pool_k; ++j) {
                int iw = in_col_start + j;
                float val = input[n_c_offset + ih * in_w + iw];
                val = tanhf(val) - s2;
                sum += val;
            }
        }
    }
    output[idx] = sum / (pool_k * pool_k);
}

torch::Tensor fused_post_conv_hip(torch::Tensor input, float s2, int pool_k) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = in_h / pool_k;
    int out_w = in_w / pool_k;
    int total_output_elements = batch_size * channels * out_h * out_w;
    auto output = torch::empty({batch_size, channels, out_h, out_w}, input.options());
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;
    fused_post_conv_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        in_h, in_w,
        out_h, out_w,
        pool_k,
        s2,
        total_output_elements
    );
    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops_v5",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_post_conv_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        # Fuse subtract1_value into conv bias
        with torch.no_grad():
            self.conv.bias.sub_(subtract1_value)

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops.fused_post_conv_hip(x, self.subtract2_value, self.kernel_size_pool)
        return x
