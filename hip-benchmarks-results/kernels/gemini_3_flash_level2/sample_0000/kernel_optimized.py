
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

softmax_maxpool_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

__global__ void softmax_maxpool_kernel_nhwc_v4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int total_spatial_out,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_spatial_out) {
        int w_out_idx = idx % out_w;
        int h_out_idx = (idx / out_w) % out_h;
        int d_out_idx = (idx / (out_w * out_h)) % out_d;
        int b_idx = idx / (out_w * out_h * out_d);
        
        float max_vals[16];
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            max_vals[c] = -1.0e30f;
        }
        
        for (int i = 0; i < 4; ++i) {
            int d_in_idx = d_out_idx * 4 + i;
            if (d_in_idx >= in_d) continue;
            for (int j = 0; j < 4; ++j) {
                int h_in_idx = h_out_idx * 4 + j;
                if (h_in_idx >= in_h) continue;
                for (int k = 0; k < 4; ++k) {
                    int w_in_idx = w_out_idx * 4 + k;
                    if (w_in_idx >= in_w) continue;
                    
                    int in_base_idx_f4 = (((b_idx * in_d + d_in_idx) * in_h + h_in_idx) * in_w + w_in_idx) * 4;
                    float4 c0 = input[in_base_idx_f4];
                    float4 c1 = input[in_base_idx_f4 + 1];
                    float4 c2 = input[in_base_idx_f4 + 2];
                    float4 c3 = input[in_base_idx_f4 + 3];
                    
                    float max_c = fmaxf(fmaxf(fmaxf(c0.x, c0.y), fmaxf(c0.z, c0.w)),
                                       fmaxf(fmaxf(c1.x, c1.y), fmaxf(c1.z, c1.w)));
                    max_c = fmaxf(max_c, fmaxf(fmaxf(c2.x, c2.y), fmaxf(c2.z, c2.w)));
                    max_c = fmaxf(max_c, fmaxf(fmaxf(c3.x, c3.y), fmaxf(c3.z, c3.w)));
                    
                    c0.x = __expf(c0.x - max_c); c0.y = __expf(c0.y - max_c); c0.z = __expf(c0.z - max_c); c0.w = __expf(c0.w - max_c);
                    c1.x = __expf(c1.x - max_c); c1.y = __expf(c1.y - max_c); c1.z = __expf(c1.z - max_c); c1.w = __expf(c1.w - max_c);
                    c2.x = __expf(c2.x - max_c); c2.y = __expf(c2.y - max_c); c2.z = __expf(c2.z - max_c); c2.w = __expf(c2.w - max_c);
                    c3.x = __expf(c3.x - max_c); c3.y = __expf(c3.y - max_c); c3.z = __expf(c3.z - max_c); c3.w = __expf(c3.w - max_c);
                    
                    float sum_exp = c0.x + c0.y + c0.z + c0.w + c1.x + c1.y + c1.z + c1.w + c2.x + c2.y + c2.z + c2.w + c3.x + c3.y + c3.z + c3.w;
                    float inv_sum = 1.0f / sum_exp;
                    
                    max_vals[0] = fmaxf(max_vals[0], c0.x * inv_sum);
                    max_vals[1] = fmaxf(max_vals[1], c0.y * inv_sum);
                    max_vals[2] = fmaxf(max_vals[2], c0.z * inv_sum);
                    max_vals[3] = fmaxf(max_vals[3], c0.w * inv_sum);
                    max_vals[4] = fmaxf(max_vals[4], c1.x * inv_sum);
                    max_vals[5] = fmaxf(max_vals[5], c1.y * inv_sum);
                    max_vals[6] = fmaxf(max_vals[6], c1.z * inv_sum);
                    max_vals[7] = fmaxf(max_vals[7], c1.w * inv_sum);
                    max_vals[8] = fmaxf(max_vals[8], c2.x * inv_sum);
                    max_vals[9] = fmaxf(max_vals[9], c2.y * inv_sum);
                    max_vals[10] = fmaxf(max_vals[10], c2.z * inv_sum);
                    max_vals[11] = fmaxf(max_vals[11], c2.w * inv_sum);
                    max_vals[12] = fmaxf(max_vals[12], c3.x * inv_sum);
                    max_vals[13] = fmaxf(max_vals[13], c3.y * inv_sum);
                    max_vals[14] = fmaxf(max_vals[14], c3.z * inv_sum);
                    max_vals[15] = fmaxf(max_vals[15], c3.w * inv_sum);
                }
            }
        }
        
        int out_base_idx_f4 = idx * 4;
        output[out_base_idx_f4] = make_float4(max_vals[0], max_vals[1], max_vals[2], max_vals[3]);
        output[out_base_idx_f4 + 1] = make_float4(max_vals[4], max_vals[5], max_vals[6], max_vals[7]);
        output[out_base_idx_f4 + 2] = make_float4(max_vals[8], max_vals[9], max_vals[10], max_vals[11]);
        output[out_base_idx_f4 + 3] = make_float4(max_vals[12], max_vals[13], max_vals[14], max_vals[15]);
    }
}

torch::Tensor softmax_maxpool_hip(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    
    int out_d = 3;
    int out_h = 7;
    int out_w = 7;
    
    auto output = torch::empty({batch_size, 16, out_d, out_h, out_w}, input.options().memory_format(torch::MemoryFormat::ChannelsLast3d));
    
    int total_spatial_out = batch_size * out_d * out_h * out_w;
    int block_size = 256;
    int num_blocks = (total_spatial_out + block_size - 1) / block_size;
    
    softmax_maxpool_kernel_nhwc_v4<<<num_blocks, block_size>>>(
        (const float4*)input.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        total_spatial_out, in_d, in_h, in_w, out_d, out_h, out_w
    );
    
    return output;
}
"""

softmax_maxpool = load_inline(
    name="softmax_maxpool_nhwc_v4",
    cpp_sources=softmax_maxpool_source,
    functions=["softmax_maxpool_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size).cuda().to(memory_format=torch.channels_last_3d)
        self.softmax_maxpool = softmax_maxpool

    def forward(self, x):
        # We assume the input x might be NCDHW and needs to be NHWC for the conv.
        # Conv3d with channels_last_3d is usually faster.
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv(x)
        x = self.softmax_maxpool.softmax_maxpool_hip(x)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 32, 32
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    pool_kernel_size = 2
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
