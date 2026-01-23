import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Heavily optimized dequantization kernel using vector types for memory coalescing

hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// Use vector types for coalesced memory access
typedef unsigned int uint32_t;

// Process 8 weights at a time with vectorized loads/stores
__global__ void int4_dequant_vec4_kernel(
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int N, int K_half, int group_size, int num_groups
) {
    // Each thread processes 4 packed bytes = 8 weights
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of 4-byte chunks
    int total_chunks = N * (K_half / 4);
    
    if (thread_idx >= total_chunks) return;
    
    // Calculate position
    int n = thread_idx / (K_half / 4);
    int chunk_in_row = thread_idx % (K_half / 4);
    int k2_base = chunk_in_row * 4;  // packed byte index
    int k_base = k2_base * 2;         // weight index
    
    // Load 4 packed bytes as uint32_t for coalescing
    const uint32_t* packed_ptr = reinterpret_cast<const uint32_t*>(weight_packed + n * K_half + k2_base);
    uint32_t packed = *packed_ptr;
    
    // Unpack bytes
    uint8_t b0 = packed & 0xFF;
    uint8_t b1 = (packed >> 8) & 0xFF;
    uint8_t b2 = (packed >> 16) & 0xFF;
    uint8_t b3 = (packed >> 24) & 0xFF;
    
    // Get scale (check group boundaries)
    int group_idx = k_base / group_size;
    const __half* row_scales = scales + n * num_groups;
    float scale = __half2float(row_scales[group_idx]);
    
    int group_end = (group_idx + 1) * group_size;
    int K = K_half * 2;
    
    // Calculate output pointer
    __half* out_ptr = output + n * K + k_base;
    
    // Check if all 8 weights are in same group (common case)
    if (k_base + 8 <= group_end) {
        // All in same group - fast path
        out_ptr[0] = __float2half(scale * (float)((b0 & 0x0F) - 8));
        out_ptr[1] = __float2half(scale * (float)(((b0 >> 4) & 0x0F) - 8));
        out_ptr[2] = __float2half(scale * (float)((b1 & 0x0F) - 8));
        out_ptr[3] = __float2half(scale * (float)(((b1 >> 4) & 0x0F) - 8));
        out_ptr[4] = __float2half(scale * (float)((b2 & 0x0F) - 8));
        out_ptr[5] = __float2half(scale * (float)(((b2 >> 4) & 0x0F) - 8));
        out_ptr[6] = __float2half(scale * (float)((b3 & 0x0F) - 8));
        out_ptr[7] = __float2half(scale * (float)(((b3 >> 4) & 0x0F) - 8));
    } else {
        // Crossing boundary - slow path
        float scale_next = __half2float(row_scales[group_idx + 1]);
        out_ptr[0] = __float2half(((k_base + 0 < group_end) ? scale : scale_next) * (float)((b0 & 0x0F) - 8));
        out_ptr[1] = __float2half(((k_base + 1 < group_end) ? scale : scale_next) * (float)(((b0 >> 4) & 0x0F) - 8));
        out_ptr[2] = __float2half(((k_base + 2 < group_end) ? scale : scale_next) * (float)((b1 & 0x0F) - 8));
        out_ptr[3] = __float2half(((k_base + 3 < group_end) ? scale : scale_next) * (float)(((b1 >> 4) & 0x0F) - 8));
        out_ptr[4] = __float2half(((k_base + 4 < group_end) ? scale : scale_next) * (float)((b2 & 0x0F) - 8));
        out_ptr[5] = __float2half(((k_base + 5 < group_end) ? scale : scale_next) * (float)(((b2 >> 4) & 0x0F) - 8));
        out_ptr[6] = __float2half(((k_base + 6 < group_end) ? scale : scale_next) * (float)((b3 & 0x0F) - 8));
        out_ptr[7] = __float2half(((k_base + 7 < group_end) ? scale : scale_next) * (float)(((b3 >> 4) & 0x0F) - 8));
    }
}

// Even more optimized: process 16 weights per thread
__global__ void int4_dequant_vec8_kernel(
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int N, int K_half, int group_size, int num_groups
) {
    // Each thread processes 8 packed bytes = 16 weights
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of 8-byte chunks
    int chunks_per_row = K_half / 8;
    int total_chunks = N * chunks_per_row;
    
    if (thread_idx >= total_chunks) return;
    
    int n = thread_idx / chunks_per_row;
    int chunk_in_row = thread_idx % chunks_per_row;
    int k2_base = chunk_in_row * 8;
    int k_base = k2_base * 2;
    int K = K_half * 2;
    
    // Load 8 packed bytes as 2x uint32_t
    const uint32_t* packed_ptr = reinterpret_cast<const uint32_t*>(weight_packed + n * K_half + k2_base);
    uint32_t packed0 = packed_ptr[0];
    uint32_t packed1 = packed_ptr[1];
    
    // Get scale
    int group_idx = k_base / group_size;
    float scale = __half2float(scales[n * num_groups + group_idx]);
    
    int group_end = (group_idx + 1) * group_size;
    __half* out_ptr = output + n * K + k_base;
    
    // Unpack and write first 8 weights
    #define DEQUANT(packed_val, shift) (__float2half(scale * (float)((((packed_val) >> (shift)) & 0x0F) - 8)))
    
    if (k_base + 16 <= group_end) {
        // All 16 weights in same group
        out_ptr[0] = DEQUANT(packed0, 0);
        out_ptr[1] = DEQUANT(packed0, 4);
        out_ptr[2] = DEQUANT(packed0, 8);
        out_ptr[3] = DEQUANT(packed0, 12);
        out_ptr[4] = DEQUANT(packed0, 16);
        out_ptr[5] = DEQUANT(packed0, 20);
        out_ptr[6] = DEQUANT(packed0, 24);
        out_ptr[7] = DEQUANT(packed0, 28);
        out_ptr[8] = DEQUANT(packed1, 0);
        out_ptr[9] = DEQUANT(packed1, 4);
        out_ptr[10] = DEQUANT(packed1, 8);
        out_ptr[11] = DEQUANT(packed1, 12);
        out_ptr[12] = DEQUANT(packed1, 16);
        out_ptr[13] = DEQUANT(packed1, 20);
        out_ptr[14] = DEQUANT(packed1, 24);
        out_ptr[15] = DEQUANT(packed1, 28);
    } else {
        // Handle group boundary crossing
        float scale_next = __half2float(scales[n * num_groups + group_idx + 1]);
        
        #define DEQUANT_CHECK(packed_val, shift, idx) \
            (__float2half(((k_base + (idx) < group_end) ? scale : scale_next) * (float)((((packed_val) >> (shift)) & 0x0F) - 8)))
        
        out_ptr[0] = DEQUANT_CHECK(packed0, 0, 0);
        out_ptr[1] = DEQUANT_CHECK(packed0, 4, 1);
        out_ptr[2] = DEQUANT_CHECK(packed0, 8, 2);
        out_ptr[3] = DEQUANT_CHECK(packed0, 12, 3);
        out_ptr[4] = DEQUANT_CHECK(packed0, 16, 4);
        out_ptr[5] = DEQUANT_CHECK(packed0, 20, 5);
        out_ptr[6] = DEQUANT_CHECK(packed0, 24, 6);
        out_ptr[7] = DEQUANT_CHECK(packed0, 28, 7);
        out_ptr[8] = DEQUANT_CHECK(packed1, 0, 8);
        out_ptr[9] = DEQUANT_CHECK(packed1, 4, 9);
        out_ptr[10] = DEQUANT_CHECK(packed1, 8, 10);
        out_ptr[11] = DEQUANT_CHECK(packed1, 12, 11);
        out_ptr[12] = DEQUANT_CHECK(packed1, 16, 12);
        out_ptr[13] = DEQUANT_CHECK(packed1, 20, 13);
        out_ptr[14] = DEQUANT_CHECK(packed1, 24, 14);
        out_ptr[15] = DEQUANT_CHECK(packed1, 28, 15);
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
    
    // Use vec4 kernel (processes 8 weights = 4 bytes per thread)
    int total_chunks = N * (K_half / 4);
    int block_size = 256;
    int num_blocks = (total_chunks + block_size - 1) / block_size;
    
    int4_dequant_vec4_kernel<<<num_blocks, block_size>>>(
        weight_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, K_half, group_size, num_groups
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
    name="int4_dequant_v4",
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Dequantize weights
        w_dequant = int4_dequant_module.int4_dequant_hip(
            self.weight_packed, self.scales,
            self.N, self.K, self.group_size
        )
        
        # Use PyTorch's matmul
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
