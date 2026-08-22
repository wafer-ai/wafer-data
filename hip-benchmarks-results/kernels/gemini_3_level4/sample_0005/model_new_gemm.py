import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set compiler to hipcc
os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

#define GROUP_SIZE 128

__global__ void unpack_dequant_kernel(
    const uint8_t* __restrict__ packed,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int N, int K, int num_groups)
{
    int n = blockIdx.y;
    int k_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes 8 weights (4 bytes of packed)
    int k_packed_offset = k_block_idx * 4;
    
    // Check if within bounds of packed array (row-wise)
    if (k_packed_offset >= K/2) return;
    
    // Load 4 bytes
    const uint32_t* packed_ptr = (const uint32_t*)(packed + n * (K/2) + k_packed_offset);
    uint32_t packed_val = *packed_ptr;
    
    // Calculate start index in output
    int k_start = k_packed_offset * 2;
    
    // Get scale
    // 8 weights are guaranteed to be in same group if K%8==0 and GroupSize%8==0
    int g = k_start / GROUP_SIZE;
    
    float s = __half2float(scales[n * num_groups + g]);
    float bias = s * 8.0f;
    
    uchar4 bytes = *reinterpret_cast<uchar4*>(&packed_val);
    
    // Unpack and dequantize
    float w0 = s * (float)(bytes.x & 0xF) - bias;
    float w1 = s * (float)(bytes.x >> 4) - bias;
    float w2 = s * (float)(bytes.y & 0xF) - bias;
    float w3 = s * (float)(bytes.y >> 4) - bias;
    float w4 = s * (float)(bytes.z & 0xF) - bias;
    float w5 = s * (float)(bytes.z >> 4) - bias;
    float w6 = s * (float)(bytes.w & 0xF) - bias;
    float w7 = s * (float)(bytes.w >> 4) - bias;
    
    // Store as float4 (8 halves)
    half result[8];
    result[0] = __float2half(w0);
    result[1] = __float2half(w1);
    result[2] = __float2half(w2);
    result[3] = __float2half(w3);
    result[4] = __float2half(w4);
    result[5] = __float2half(w5);
    result[6] = __float2half(w6);
    result[7] = __float2half(w7);
    
    float4* out_ptr = (float4*)(output + n * K + k_start);
    *out_ptr = *reinterpret_cast<float4*>(result);
}

void launch_unpack_dequant(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor output,
    int N, int K, int num_groups)
{
    const uint8_t* packed_ptr = packed.data_ptr<uint8_t>();
    const half* scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    
    // Grid calculation
    // X dimension covers K/2 bytes. Each thread does 4 bytes.
    // Threads per row = (K/2) / 4 = K/8.
    int threads_per_row = K / 8;
    int block_size = 256;
    int grid_x = (threads_per_row + block_size - 1) / block_size;
    int grid_y = N;
    
    dim3 grid(grid_x, grid_y);
    dim3 block(block_size);
    
    unpack_dequant_kernel<<<grid, block>>>(packed_ptr, scales_ptr, output_ptr, N, K, num_groups);
}
"""

class ModelNew(nn.Module):
    def __init__(self, K: int, N: int, group_size: int = 128):
        super().__init__()
        self.K = K
        self.N = N
        self.group_size = group_size
        self.num_groups = K // group_size
        
        assert group_size == 128, "Kernel optimized for group_size=128"
        assert K % 32 == 0, "K must be multiple of 32"

        self.register_buffer(
            "weight_packed",
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        )

        self.register_buffer(
            "scales",
            torch.randn(N, self.num_groups, dtype=torch.float16).abs() * 0.1
        )
        
        self.int4_gemm = load_inline(
            name="int4_gemm",
            cpp_sources=cpp_source,
            functions=["launch_unpack_dequant"],
            verbose=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        M = batch_size * seq_len
        
        # Dequantize
        w_dequant = torch.empty((self.N, self.K), device=x.device, dtype=torch.float16)
        
        self.int4_gemm.launch_unpack_dequant(
            self.weight_packed,
            self.scales,
            w_dequant,
            self.N, self.K, self.num_groups
        )
        
        # GEMM
        # x: (M, K)
        # w_dequant: (N, K)
        # out = x @ w.T
        x_view = x.view(-1, self.K)
        out = torch.matmul(x_view, w_dequant.T)
        
        return out.view(batch_size, seq_len, self.N)

# Configuration sized for LLM inference workloads
batch_size = 4
seq_len = 2048
K = 4096         # Input features (hidden dim)
N = 11008        # Output features (MLP intermediate, typical for 7B models)
group_size = 128 # Standard group size for GPTQ

def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16).cuda()]

def get_init_inputs():
    return [K, N, group_size]
