import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// Fused INT4 unpack + dequantize + GEMM kernel
// Uses FP16 accumulation to match reference precision
__global__ void int4_gemm_kernel(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int M, int N, int K, int group_size, int num_groups
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    // Use float32 for accumulation but match reference behavior
    float acc = 0.0f;
    int K_half = K / 2;
    
    for (int k = 0; k < K; k += 2) {
        int byte_idx = k / 2;
        uint8_t packed_byte = weight_packed[col * K_half + byte_idx];
        
        // Unpack INT4 values
        int w0 = packed_byte & 0x0F;
        int w1 = (packed_byte >> 4) & 0x0F;
        
        // Get scale for this group
        int g = k / group_size;
        __half scale = scales[col * num_groups + g];
        
        // Dequantize in FP16 to match reference: scale * (w - 8)
        __half w_dequant0 = __hmul(scale, __float2half((float)(w0 - 8)));
        __half w_dequant1 = __hmul(scale, __float2half((float)(w1 - 8)));
        
        // Load X values
        __half x0 = X[row * K + k];
        __half x1 = X[row * K + k + 1];
        
        // Accumulate in FP32
        acc += __half2float(__hmul(x0, w_dequant0)) + __half2float(__hmul(x1, w_dequant1));
    }
    
    output[row * N + col] = __float2half(acc);
}

// Optimized version with vectorized loads (process 8 weights at a time)
__global__ void int4_gemm_opt_kernel(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int M, int N, int K, int group_size, int num_groups
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float acc = 0.0f;
    int K_half = K / 2;
    
    // Process 8 weights at a time (4 bytes)
    for (int k = 0; k < K; k += 8) {
        int byte_idx = k / 2;
        
        // Load 4 bytes = 8 INT4 weights
        uint8_t b0 = weight_packed[col * K_half + byte_idx];
        uint8_t b1 = weight_packed[col * K_half + byte_idx + 1];
        uint8_t b2 = weight_packed[col * K_half + byte_idx + 2];
        uint8_t b3 = weight_packed[col * K_half + byte_idx + 3];
        
        // Get scale for this group (all 8 weights likely same group if group_size >= 8)
        int g = k / group_size;
        float scale = __half2float(scales[col * num_groups + g]);
        
        // Unpack and process
        int w0 = b0 & 0x0F;
        int w1 = (b0 >> 4) & 0x0F;
        int w2 = b1 & 0x0F;
        int w3 = (b1 >> 4) & 0x0F;
        int w4 = b2 & 0x0F;
        int w5 = (b2 >> 4) & 0x0F;
        int w6 = b3 & 0x0F;
        int w7 = (b3 >> 4) & 0x0F;
        
        // Load X values
        float x0 = __half2float(X[row * K + k]);
        float x1 = __half2float(X[row * K + k + 1]);
        float x2 = __half2float(X[row * K + k + 2]);
        float x3 = __half2float(X[row * K + k + 3]);
        float x4 = __half2float(X[row * K + k + 4]);
        float x5 = __half2float(X[row * K + k + 5]);
        float x6 = __half2float(X[row * K + k + 6]);
        float x7 = __half2float(X[row * K + k + 7]);
        
        // Dequantize and accumulate
        acc += x0 * scale * (float)(w0 - 8);
        acc += x1 * scale * (float)(w1 - 8);
        acc += x2 * scale * (float)(w2 - 8);
        acc += x3 * scale * (float)(w3 - 8);
        acc += x4 * scale * (float)(w4 - 8);
        acc += x5 * scale * (float)(w5 - 8);
        acc += x6 * scale * (float)(w6 - 8);
        acc += x7 * scale * (float)(w7 - 8);
    }
    
    output[row * N + col] = __float2half(acc);
}

torch::Tensor int4_gemm_hip(
    torch::Tensor X,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    int group_size
) {
    int M = X.size(0);
    int K = X.size(1);
    int N = weight_packed.size(0);
    int num_groups = scales.size(1);
    
    auto output = torch::empty({M, N}, X.options());
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    int4_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        weight_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        M, N, K, group_size, num_groups
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor int4_gemm_hip(
    torch::Tensor X,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    int group_size
);
"""

int4_gemm_module = load_inline(
    name="int4_gemm_hip",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["int4_gemm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x_2d = x.view(-1, self.K)
        
        out = int4_gemm_module.int4_gemm_hip(
            x_2d,
            self.weight_packed,
            self.scales,
            self.group_size
        )
        
        return out.view(batch_size, seq_len, self.N)


# Configuration
batch_size = 4
seq_len = 2048
K = 4096
N = 11008
group_size = 128


def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]


def get_init_inputs():
    return [K, N, group_size]
