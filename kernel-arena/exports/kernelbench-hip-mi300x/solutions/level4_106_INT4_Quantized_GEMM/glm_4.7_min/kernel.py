import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

int4_gemm_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/torch.h>

__global__ void int4_quantized_gemm_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_packed,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int M,
    int N,
    int K,
    int group_size
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (n >= N || m >= M) return;
    
    int num_groups = K / group_size;
    
    double acc = 0.0;  // Double precision for accumulation
    
    for (int k = 0; k < K; k++) {
        // x is (M, K) row-major
        float x_val = __half2float(x[m * K + k]);
        
        // weight_packed is (N, K/2) row-major
        int packed_idx = n * (K / 2) + (k / 2);
        uint8_t packed = weight_packed[packed_idx];
        
        int w_int;
        if (k % 2 == 0) {
            w_int = packed & 0x0F;
        } else {
            w_int = (packed >> 4) & 0x0F;
        }
        
        // scales is (N, num_groups) row-major
        int group_idx = k / group_size;
        float scale = __half2float(scales[n * num_groups + group_idx]);
        
        // Dequantize: scale * (w_int - 8)
        double w_dequant = scale * ((double)w_int - 8.0);
        
        // Accumulate
        acc += x_val * w_dequant;
    }
    
    output[m * N + n] = __float2half((float)acc);
}

torch::Tensor int4_quantized_gemm_hip(
    torch::Tensor x,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    int K,
    int N,
    int group_size
) {
    auto M = x.size(0);
    
    auto output = torch::zeros({M, N}, x.options());
    
    const int block_size_x = 16;
    const int block_size_y = 16;
    const int num_blocks_x = (N + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (M + block_size_y - 1) / block_size_y;
    
    dim3 blockDim(block_size_x, block_size_y);
    dim3 gridDim(num_blocks_x, num_blocks_y);
    
    auto x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    auto scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
    auto output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    
    hipLaunchKernelGGL(
        int4_quantized_gemm_kernel,
        gridDim, blockDim, 0, 0,
        x_ptr,
        weight_packed.data_ptr<uint8_t>(),
        scales_ptr,
        output_ptr,
        M, N, K, group_size
    );
    
    return output;
}
"""

int4_gemm = load_inline(
    name="int4_gemm",
    cpp_sources=int4_gemm_source,
    functions=["int4_quantized_gemm_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, K: int, N: int, group_size: int = 128):
        super().__init__()
        self.K = K
        self.N = N
        self.group_size = group_size
        self.num_groups = K // group_size

        assert K % group_size == 0, "K must be divisible by group_size"
        assert K % 2 == 0, "K must be even for INT4 packing"

        self.register_buffer(
            "weight_packed",
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        )

        self.register_buffer(
            "scales",
            torch.randn(N, self.num_groups, dtype=torch.float16).abs() * 0.1
        )

        self.int4_gemm = int4_gemm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_2d = x.view(-1, self.K)

        out = self.int4_gemm.int4_quantized_gemm_hip(
            x_2d, 
            self.weight_packed, 
            self.scales, 
            self.K, 
            self.N, 
            self.group_size
        )

        return out.view(batch_size, seq_len, self.N)