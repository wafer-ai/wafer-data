import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Hybrid approach: highly optimized dequantization + rocBLAS GEMM
# This ensures correctness while being faster than naive implementation

hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// Optimized INT4 dequantization kernel with vectorized loads
// Processes 8 weights (4 packed bytes) per iteration
__global__ void int4_dequant_vec_kernel(
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int N, int K, int group_size, int num_groups
) {
    // Each thread processes multiple elements
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int K_half = K / 2;
    int total_packed = N * K_half;
    
    // Each thread handles 4 packed bytes (8 weights)
    int packed_start = thread_id * 4;
    
    if (packed_start >= total_packed) return;
    
    // Calculate n and k2 indices
    int n = packed_start / K_half;
    int k2 = packed_start % K_half;
    
    // Make sure we stay within same row
    if (k2 + 4 > K_half) {
        // Fall back to single element processing
        for (int i = 0; i < 4 && packed_start + i < total_packed; i++) {
            int curr_n = (packed_start + i) / K_half;
            int curr_k2 = (packed_start + i) % K_half;
            int k = curr_k2 * 2;
            
            uint8_t packed_byte = weight_packed[curr_n * K_half + curr_k2];
            
            int w0_int = packed_byte & 0x0F;
            int w1_int = (packed_byte >> 4) & 0x0F;
            
            int group_idx = k / group_size;
            float scale = __half2float(scales[curr_n * num_groups + group_idx]);
            
            output[curr_n * K + k] = __float2half(scale * (float)(w0_int - 8));
            output[curr_n * K + k + 1] = __float2half(scale * (float)(w1_int - 8));
        }
        return;
    }
    
    // Load 4 packed bytes (can use uint32 for coalescing)
    const uint8_t* row_weights = weight_packed + n * K_half;
    uint8_t b0 = row_weights[k2];
    uint8_t b1 = row_weights[k2 + 1];
    uint8_t b2 = row_weights[k2 + 2];
    uint8_t b3 = row_weights[k2 + 3];
    
    int k = k2 * 2;
    
    // Get scales
    const __half* row_scales = scales + n * num_groups;
    int group_idx = k / group_size;
    float scale = __half2float(row_scales[group_idx]);
    
    // Check if we cross group boundary within these 8 elements
    int group_end = (group_idx + 1) * group_size;
    
    __half* row_out = output + n * K + k;
    
    if (k + 8 <= group_end) {
        // All 8 weights in same group - use same scale
        row_out[0] = __float2half(scale * (float)((b0 & 0x0F) - 8));
        row_out[1] = __float2half(scale * (float)(((b0 >> 4) & 0x0F) - 8));
        row_out[2] = __float2half(scale * (float)((b1 & 0x0F) - 8));
        row_out[3] = __float2half(scale * (float)(((b1 >> 4) & 0x0F) - 8));
        row_out[4] = __float2half(scale * (float)((b2 & 0x0F) - 8));
        row_out[5] = __float2half(scale * (float)(((b2 >> 4) & 0x0F) - 8));
        row_out[6] = __float2half(scale * (float)((b3 & 0x0F) - 8));
        row_out[7] = __float2half(scale * (float)(((b3 >> 4) & 0x0F) - 8));
    } else {
        // Crossing group boundary - handle each element
        float scale_next = __half2float(row_scales[group_idx + 1]);
        row_out[0] = __float2half(((k < group_end) ? scale : scale_next) * (float)((b0 & 0x0F) - 8));
        row_out[1] = __float2half(((k+1 < group_end) ? scale : scale_next) * (float)(((b0 >> 4) & 0x0F) - 8));
        row_out[2] = __float2half(((k+2 < group_end) ? scale : scale_next) * (float)((b1 & 0x0F) - 8));
        row_out[3] = __float2half(((k+3 < group_end) ? scale : scale_next) * (float)(((b1 >> 4) & 0x0F) - 8));
        row_out[4] = __float2half(((k+4 < group_end) ? scale : scale_next) * (float)((b2 & 0x0F) - 8));
        row_out[5] = __float2half(((k+5 < group_end) ? scale : scale_next) * (float)(((b2 >> 4) & 0x0F) - 8));
        row_out[6] = __float2half(((k+6 < group_end) ? scale : scale_next) * (float)((b3 & 0x0F) - 8));
        row_out[7] = __float2half(((k+7 < group_end) ? scale : scale_next) * (float)(((b3 >> 4) & 0x0F) - 8));
    }
}

// Simpler but highly parallelized version
__global__ void int4_dequant_simple_kernel(
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int N, int K, int group_size, int num_groups
) {
    // Each thread handles 2 consecutive weights (one packed byte)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_half = K / 2;
    int total_packed = N * K_half;
    
    if (idx >= total_packed) return;
    
    int n = idx / K_half;
    int k2 = idx % K_half;
    int k = k2 * 2;
    
    uint8_t packed_byte = weight_packed[idx];
    
    // Unpack
    int w0_int = packed_byte & 0x0F;
    int w1_int = (packed_byte >> 4) & 0x0F;
    
    // Get scale
    int group_idx = k / group_size;
    float scale = __half2float(scales[n * num_groups + group_idx]);
    
    // Write output
    output[n * K + k] = __float2half(scale * (float)(w0_int - 8));
    
    // Check if second weight is in same group (usually yes)
    if ((k + 1) / group_size == group_idx) {
        output[n * K + k + 1] = __float2half(scale * (float)(w1_int - 8));
    } else {
        float scale1 = __half2float(scales[n * num_groups + group_idx + 1]);
        output[n * K + k + 1] = __float2half(scale1 * (float)(w1_int - 8));
    }
}

torch::Tensor int4_dequant_hip(
    torch::Tensor weight_packed,
    torch::Tensor scales,
    int64_t N,
    int64_t K,
    int64_t group_size
) {
    auto output = torch::empty({N, K}, torch::TensorOptions().dtype(torch::kFloat16).device(weight_packed.device()));
    
    int num_groups = K / group_size;
    int K_half = K / 2;
    int total_packed = N * K_half;
    
    // Use the simple kernel with high parallelism
    int block_size = 256;
    int num_blocks = (total_packed + block_size - 1) / block_size;
    
    int4_dequant_simple_kernel<<<num_blocks, block_size>>>(
        weight_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, K, group_size, num_groups
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor int4_dequant_hip(
    torch::Tensor weight_packed,
    torch::Tensor scales,
    int64_t N,
    int64_t K,
    int64_t group_size
);
"""

int4_dequant_module = load_inline(
    name="int4_dequant_v3",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["int4_dequant_hip"],
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

        # Packed INT4 weights: 2 weights per byte
        self.register_buffer(
            "weight_packed",
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        )

        # Per-group scales
        self.register_buffer(
            "scales",
            torch.randn(N, self.num_groups, dtype=torch.float16).abs() * 0.1
        )
        
        # Cache for dequantized weights
        self._weight_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Dequantize weights using optimized kernel
        w_dequant = int4_dequant_module.int4_dequant_hip(
            self.weight_packed, self.scales,
            self.N, self.K, self.group_size
        )
        
        # Use PyTorch's matmul for the GEMM
        x_2d = x.view(-1, self.K)
        output = torch.matmul(x_2d, w_dequant.T)
        
        return output.view(batch_size, seq_len, self.N)


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
