import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use a hybrid approach: dequantize in a custom kernel, then use torch.matmul
# This should be faster than the naive approach while maintaining correctness

hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// Fused INT4 unpack + dequantization kernel
// weight_packed: (N, K/2) uint8
// scales: (N, num_groups) fp16
// output: (N, K) fp16
__global__ void int4_dequant_kernel(
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int N, int K, int group_size, int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    
    if (idx >= total) return;
    
    int n = idx / K;
    int k = idx % K;
    
    // Get packed byte index
    int k2 = k / 2;
    uint8_t packed_byte = weight_packed[n * (K / 2) + k2];
    
    // Unpack INT4: low nibble = even k, high nibble = odd k
    int w_int;
    if (k % 2 == 0) {
        w_int = packed_byte & 0x0F;
    } else {
        w_int = (packed_byte >> 4) & 0x0F;
    }
    
    // Get scale for this group
    int group_idx = k / group_size;
    float scale = __half2float(scales[n * num_groups + group_idx]);
    
    // Dequantize: scale * (w_int - 8)
    float w_dequant = scale * (float)(w_int - 8);
    
    output[n * K + k] = __float2half(w_dequant);
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
    int total = N * K;
    
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    int4_dequant_kernel<<<num_blocks, block_size>>>(
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
    name="int4_dequant",
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
        
        # Dequantize weights using our fast kernel
        w_dequant = int4_dequant_module.int4_dequant_hip(
            self.weight_packed, self.scales,
            self.N, self.K, self.group_size
        )
        
        # Use PyTorch's matmul for the GEMM (uses rocBLAS)
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
