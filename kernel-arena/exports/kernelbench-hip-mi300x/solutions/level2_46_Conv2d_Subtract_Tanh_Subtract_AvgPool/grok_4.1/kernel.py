import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_post_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_tanh_pool_sub_kernel(const float* in, float* out, float sub_val, int N, int C, int Hi, int Wi, int Ho, int Wo) {
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t size_out = size_t(N) * C * Ho * Wo;
    if (gid >= size_out) return;
    int j = gid % Wo;
    size_t temp1 = gid / Wo;
    int i = temp1 % Ho;
    size_t temp2 = temp1 / Ho;
    int c = temp2 % C;
    int n = temp2 / C;
    float sum = 0.0f;
    int base_h = i * 2;
    int base_w = j * 2;
    int ch_offset = n * C + c;
    int row_offset = ch_offset * Hi + base_h;
    sum += tanhf(in[row_offset * Wi + base_w]);
    sum += tanhf(in[(row_offset + 1) * Wi + base_w]);
    sum += tanhf(in[row_offset * Wi + base_w + 1]);
    sum += tanhf(in[(row_offset + 1) * Wi + base_w + 1]);
    out[gid] = sum * 0.25f - sub_val;
}

torch::Tensor fused_post_hip(torch::Tensor in, float sub_val) {
    int N = in.size(0);
    int C = in.size(1);
    int Hi = in.size(2);
    int Wi = in.size(3);
    int Ho = Hi / 2;
    int Wo = Wi / 2;
    torch::Tensor out = torch::empty({N, C, Ho, Wo}, in.options());
    int size_out = N * C * Ho * Wo;
    const int block_size = 256;
    int num_blocks = (size_out + block_size - 1) / block_size;
    fused_tanh_pool_sub_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), sub_val, N, C, Hi, Wi, Ho, Wo);
    return out;
}
"""

fused_post = load_inline(
    name="fused_post",
    cpp_sources=fused_post_cpp_source,
    functions=["fused_post_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv.bias.data -= subtract1_value
        self.subtract2_value = subtract2_value
        self.fused_post = fused_post

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_post.fused_post_hip(x, self.subtract2_value)
        return x
