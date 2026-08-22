import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fully fused INT4 GEMM kernel with tiling
# This avoids materializing the full dequantized weight matrix

hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 64

// Fused INT4 unpack + dequant + GEMM kernel with tiling
__global__ void int4_gemm_fused_kernel(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int M, int N, int K, int group_size, int num_groups
) {
    // Block indices
    int bm = blockIdx.y;
    int bn = blockIdx.x;
    
    // Thread indices
    int tx = threadIdx.x;  // 0-31
    int ty = threadIdx.y;  // 0-31
    
    // Global row/col this thread will compute
    int row = bm * TILE_M + ty;
    int col = bn * TILE_N + tx;
    
    // Shared memory for input tile
    __shared__ float X_tile[TILE_M][TILE_K + 1];  // +1 to avoid bank conflicts
    
    float acc = 0.0f;
    
    int K_half = K / 2;
    
    // Loop over K dimension in tiles
    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        // Cooperatively load X tile into shared memory
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < TILE_K / 32; i++) {
            int k_idx = k_base + tx + i * 32;
            if (row < M && k_idx < K) {
                X_tile[ty][tx + i * 32] = __half2float(X[row * K + k_idx]);
            } else {
                X_tile[ty][tx + i * 32] = 0.0f;
            }
        }
        
        __syncthreads();
        
        if (col < N) {
            // Process TILE_K elements
            #pragma unroll 8
            for (int k_off = 0; k_off < TILE_K && (k_base + k_off) < K; k_off++) {
                int k = k_base + k_off;
                
                // Get input value from shared memory
                float x_val = X_tile[ty][k_off];
                
                // Unpack INT4 weight
                int packed_idx = col * K_half + (k / 2);
                uint8_t packed_byte = weight_packed[packed_idx];
                
                int w_int;
                if (k % 2 == 0) {
                    w_int = packed_byte & 0x0F;
                } else {
                    w_int = (packed_byte >> 4) & 0x0F;
                }
                
                // Get scale for this group
                int group_idx = k / group_size;
                float scale = __half2float(scales[col * num_groups + group_idx]);
                
                // Dequantize and accumulate
                float w_dequant = scale * (float)(w_int - 8);
                acc += x_val * w_dequant;
            }
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        output[row * N + col] = __float2half(acc);
    }
}

// Optimized version processing 2 K elements per inner loop iteration
__global__ void int4_gemm_fused_vec2_kernel(
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
    
    // Process 2 K elements per iteration (one packed byte)
    for (int k2 = 0; k2 < K_half; k2++) {
        int k = k2 * 2;
        
        // Load 2 input values
        float x0 = __half2float(X[row * K + k]);
        float x1 = __half2float(X[row * K + k + 1]);
        
        // Load packed byte
        uint8_t packed_byte = weight_packed[col * K_half + k2];
        
        // Unpack both INT4 values
        int w0_int = packed_byte & 0x0F;
        int w1_int = (packed_byte >> 4) & 0x0F;
        
        // Get scales (check if both are in same group for optimization)
        int group_idx = k / group_size;
        float scale = __half2float(scales[col * num_groups + group_idx]);
        
        // Since consecutive K values are usually in the same group
        // (group_size is typically 128), we can optimize
        if ((k + 1) / group_size == group_idx) {
            // Both in same group - single scale lookup
            float w0 = scale * (float)(w0_int - 8);
            float w1 = scale * (float)(w1_int - 8);
            acc += x0 * w0 + x1 * w1;
        } else {
            // Different groups - need second scale lookup
            float scale1 = __half2float(scales[col * num_groups + group_idx + 1]);
            float w0 = scale * (float)(w0_int - 8);
            float w1 = scale1 * (float)(w1_int - 8);
            acc += x0 * w0 + x1 * w1;
        }
    }
    
    output[row * N + col] = __float2half(acc);
}

// Version with larger thread blocks and vectorized loads
__global__ void int4_gemm_fused_opt_kernel(
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
    
    // Precompute scale pointer for this output column
    const __half* col_scales = scales + col * num_groups;
    const uint8_t* col_weights = weight_packed + col * K_half;
    const __half* row_X = X + row * K;
    
    // Process in chunks of 8 packed bytes (16 weights) when possible
    int k2 = 0;
    
    // Main loop: process 4 packed bytes (8 weights) per iteration
    for (; k2 + 3 < K_half; k2 += 4) {
        int k = k2 * 2;
        
        // Load 4 packed bytes
        uint8_t b0 = col_weights[k2];
        uint8_t b1 = col_weights[k2 + 1];
        uint8_t b2 = col_weights[k2 + 2];
        uint8_t b3 = col_weights[k2 + 3];
        
        // Load 8 input values
        float x0 = __half2float(row_X[k]);
        float x1 = __half2float(row_X[k + 1]);
        float x2 = __half2float(row_X[k + 2]);
        float x3 = __half2float(row_X[k + 3]);
        float x4 = __half2float(row_X[k + 4]);
        float x5 = __half2float(row_X[k + 5]);
        float x6 = __half2float(row_X[k + 6]);
        float x7 = __half2float(row_X[k + 7]);
        
        // Get scale (assuming all 8 weights in same group - true for group_size=128 and k aligned)
        int group_idx = k / group_size;
        float scale = __half2float(col_scales[group_idx]);
        
        // Check if we cross group boundary
        int group_end = (group_idx + 1) * group_size;
        
        if (k + 8 <= group_end) {
            // All 8 weights in same group
            acc += x0 * scale * (float)((b0 & 0x0F) - 8);
            acc += x1 * scale * (float)(((b0 >> 4) & 0x0F) - 8);
            acc += x2 * scale * (float)((b1 & 0x0F) - 8);
            acc += x3 * scale * (float)(((b1 >> 4) & 0x0F) - 8);
            acc += x4 * scale * (float)((b2 & 0x0F) - 8);
            acc += x5 * scale * (float)(((b2 >> 4) & 0x0F) - 8);
            acc += x6 * scale * (float)((b3 & 0x0F) - 8);
            acc += x7 * scale * (float)(((b3 >> 4) & 0x0F) - 8);
        } else {
            // Crossing group boundary - handle each weight individually
            float scale_next = __half2float(col_scales[group_idx + 1]);
            float s0 = (k < group_end) ? scale : scale_next;
            float s1 = (k + 1 < group_end) ? scale : scale_next;
            float s2 = (k + 2 < group_end) ? scale : scale_next;
            float s3 = (k + 3 < group_end) ? scale : scale_next;
            float s4 = (k + 4 < group_end) ? scale : scale_next;
            float s5 = (k + 5 < group_end) ? scale : scale_next;
            float s6 = (k + 6 < group_end) ? scale : scale_next;
            float s7 = (k + 7 < group_end) ? scale : scale_next;
            
            acc += x0 * s0 * (float)((b0 & 0x0F) - 8);
            acc += x1 * s1 * (float)(((b0 >> 4) & 0x0F) - 8);
            acc += x2 * s2 * (float)((b1 & 0x0F) - 8);
            acc += x3 * s3 * (float)(((b1 >> 4) & 0x0F) - 8);
            acc += x4 * s4 * (float)((b2 & 0x0F) - 8);
            acc += x5 * s5 * (float)(((b2 >> 4) & 0x0F) - 8);
            acc += x6 * s6 * (float)((b3 & 0x0F) - 8);
            acc += x7 * s7 * (float)(((b3 >> 4) & 0x0F) - 8);
        }
    }
    
    // Handle remaining weights
    for (; k2 < K_half; k2++) {
        int k = k2 * 2;
        uint8_t packed_byte = col_weights[k2];
        
        float x0 = __half2float(row_X[k]);
        float x1 = __half2float(row_X[k + 1]);
        
        int group_idx = k / group_size;
        float scale = __half2float(col_scales[group_idx]);
        
        float w0 = scale * (float)((packed_byte & 0x0F) - 8);
        float w1 = scale * (float)(((packed_byte >> 4) & 0x0F) - 8);
        
        acc += x0 * w0 + x1 * w1;
    }
    
    output[row * N + col] = __float2half(acc);
}

torch::Tensor int4_gemm_hip(
    torch::Tensor x,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    int64_t K,
    int64_t N,
    int64_t group_size
) {
    int64_t M = x.numel() / K;
    
    auto x_2d = x.view({M, K}).contiguous();
    auto output = torch::empty({M, N}, x.options());
    
    int num_groups = K / group_size;
    
    // Use 16x16 thread blocks
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    int4_gemm_fused_opt_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(x_2d.data_ptr<at::Half>()),
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
    torch::Tensor x,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    int64_t K,
    int64_t N,
    int64_t group_size
);
"""

int4_gemm_module = load_inline(
    name="int4_gemm_fused",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["int4_gemm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
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
        
        output = int4_gemm_module.int4_gemm_hip(
            x, self.weight_packed, self.scales,
            self.K, self.N, self.group_size
        )
        
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
