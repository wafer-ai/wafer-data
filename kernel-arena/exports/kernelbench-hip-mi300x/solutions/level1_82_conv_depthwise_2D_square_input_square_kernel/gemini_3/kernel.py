import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void depthwise_conv2d_shm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int K, int S, int P) {
    
    extern __shared__ float smem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int n = bz / C;
    int c = bz % C;

    // Dimensions of the input tile needed for this block
    int tile_w = blockDim.x * S + K - 1;
    int tile_h = blockDim.y * S + K - 1;
    
    // Top-left of the input window for this block (without padding offset)
    int h_out_start = by * blockDim.y;
    int w_out_start = bx * blockDim.x;
    
    int h_in_start = h_out_start * S - P;
    int w_in_start = w_out_start * S - P;
    
    // Input pointer for this batch/channel
    long long input_offset_base = (long long)(n * C + c) * H * W;
    const float* input_ptr = input + input_offset_base;
    
    // Load to shared memory
    // Loop over the tile dimensions
    for (int i = ty; i < tile_h; i += blockDim.y) {
        int h_global = h_in_start + i;
        bool h_valid = (h_global >= 0 && h_global < H);
        
        for (int j = tx; j < tile_w; j += blockDim.x) {
            int w_global = w_in_start + j;
            float val = 0.0f;
            if (h_valid && w_global >= 0 && w_global < W) {
                val = input_ptr[h_global * W + w_global];
            }
            smem[i * tile_w + j] = val;
        }
    }
    
    __syncthreads();
    
    // Compute Output
    int h_out = h_out_start + ty;
    int w_out = w_out_start + tx;
    
    if (h_out < H_out && w_out < W_out) {
        float sum = 0.0f;
        
        // Weight pointer
        long long weight_offset = (long long)c * K * K;
        const float* weight_ptr = weight + weight_offset;
        
        // Input tile offset in shared memory
        int smem_row_start = ty * S;
        int smem_col_start = tx * S;
        
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                int r = smem_row_start + i;
                int c_smem = smem_col_start + j;
                
                sum += smem[r * tile_w + c_smem] * weight_ptr[i * K + j];
            }
        }
        
        // Add bias
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        long long output_idx = (long long)(n * C + c) * H_out * W_out + h_out * W_out + w_out;
        output[output_idx] = sum;
    }
}

torch::Tensor depthwise_conv2d_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    int K = weight.size(2);
    
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    
    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    
    const float* bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }
    
    dim3 block(32, 32);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C);
    
    int shared_mem_size = (block.x * stride + K - 1) * (block.y * stride + K - 1) * 4; // 4 bytes per float
    
    depthwise_conv2d_shm_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        K, stride, padding
    );
    
    return output;
}
"""

conv_module = load_inline(
    name="depthwise_conv2d",
    cpp_sources=cpp_source,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.conv_func = conv_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.conv2d.bias
        if bias is None:
            bias = torch.empty(0, device=x.device)
        return self.conv_func.depthwise_conv2d_hip(x, self.conv2d.weight, bias, self.stride, self.padding)
