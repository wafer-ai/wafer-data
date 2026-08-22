import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

int4_gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void int4_gemm_fused_kernel(
    const half* __restrict__ x,           // (M, K)
    const uint8_t* __restrict__ w_packed, // (N, K//2) packed INT4 weights
    const half* __restrict__ scales,      // (N, num_groups) per-group scales
    half* __restrict__ output,            // (M, N)
    int M, int N, int K, int group_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    // Accumulator with float for better precision
    float acc = 0.0f;
    int num_groups = K / group_size;
    int zero_point = 8;
    
    // Unroll by 8 to improve instruction-level parallelism
    for (int k = 0; k < K; k++) {
        // Load input
        half x_val = x[row * K + k];
        float x_f = __half2float(x_val);
        
        // Get packed weight index
        int k_pair = k / 2;
        int packed_idx = col * (K / 2) + k_pair;
        
        uint8_t packed_byte = w_packed[packed_idx];
        
        // Unpack INT4: low nibble for even k, high nibble for odd k
        int w_int;
        if (k % 2 == 0) {
            w_int = packed_byte & 0x0F;
        } else {
            w_int = (packed_byte >> 4) & 0x0F;
        }
        
        // Get scale for this group
        int group_idx = k / group_size;
        int scale_idx = col * num_groups + group_idx;
        half scale_h = scales[scale_idx];
        float scale_f = __half2float(scale_h);
        
        // Dequantize and accumulate
        float w_deq = scale_f * ((float)w_int - (float)zero_point);
        acc += x_f * w_deq;
    }
    
    output[row * N + col] = __float2half(acc);
}

torch::Tensor int4_gemm_fused_hip(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales, int K, int N, int group_size) {
    auto M = x.size(0);
    auto output = torch::zeros({M, N}, x.options());
    
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 7) / 8);
    
    int4_gemm_fused_kernel<<<grid, block>>>(
        reinterpret_cast<half*>(x.data_ptr<c10::Half>()),
        w_packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<c10::Half>()),
        reinterpret_cast<half*>(output.data_ptr<c10::Half>()),
        M, N, K, group_size
    );
    
    return output;
}
"""

int4_gemm = load_inline(
    name="int4_gemm",
    cpp_sources=int4_gemm_cpp_source,
    functions=["int4_gemm_fused_hip"],
    verbose=True,
    with_cuda=True,
)


class ModelNew(nn.Module):
    """
    INT4 Weight-Only Quantized Linear Layer - Optimized with fused kernel.
    """
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
        
        self.int4_gemm_kernel = int4_gemm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_2d = x.view(-1, self.K)
        
        out = self.int4_gemm_kernel.int4_gemm_fused_hip(
            x_2d, self.weight_packed, self.scales, self.K, self.N, self.group_size
        )
        
        return out.view(batch_size, seq_len, self.N)


def get_inputs():
    batch_size = 4
    seq_len = 2048
    K = 4096
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16).cuda()]


def get_init_inputs():
    K = 4096
    N = 11008
    group_size = 128
    return [K, N, group_size]