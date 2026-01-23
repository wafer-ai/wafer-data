
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

dequant_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hip/hip_fp16.h>

__global__ void fast_dequant_kernel_v3(
    const uint8_t* __restrict__ weight_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ w_dequant,
    int N, int K, int group_size
) {
    int n = blockIdx.y;
    int k_8 = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N && k_8 * 8 < K) {
        int k_base = k_8 * 8;
        uint32_t packed_val = *((uint32_t*)&weight_packed[n * (K / 2) + k_base / 2]);
        int num_groups = K / group_size;
        float s = (float)scales[n * num_groups + (k_base / group_size)];
        __half out[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t b = (uint8_t)(packed_val >> (i * 8));
            out[i*2] = (__half)(s * ((float)(b & 0x0F) - 8.0f));
            out[i*2+1] = (__half)(s * ((float)(b >> 4) - 8.0f));
        }
        *((float4*)&w_dequant[n * K + k_base]) = *((float4*)&out[0]);
    }
}

void dequantize_hip_inplace(torch::Tensor weight_packed, torch::Tensor scales, torch::Tensor w_dequant, int group_size) {
    int N = weight_packed.size(0);
    int K = weight_packed.size(1) * 2;
    dim3 block(256, 1);
    dim3 grid((K / 8 + 255) / 256, N);
    fast_dequant_kernel_v3<<<grid, block>>>(
        (uint8_t*)weight_packed.data_ptr<uint8_t>(),
        (__half*)scales.data_ptr<at::Half>(),
        (__half*)w_dequant.data_ptr<at::Half>(),
        N, K, group_size
    );
}
"""

dequant_module = load_inline(
    name="dequant_module_v3",
    cpp_sources=dequant_cpp_source,
    functions=["dequantize_hip_inplace"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, K: int, N: int, group_size: int = 128):
        super().__init__()
        self.K = K
        self.N = N
        self.group_size = group_size
        self.num_groups = K // group_size
        self.register_buffer("weight_packed", torch.randint(0, 256, (N, K // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.randn(N, self.num_groups, dtype=torch.float16).abs() * 0.1)
        self.register_buffer("w_dequant_buffer", torch.empty((N, K), dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequant_module.dequantize_hip_inplace(self.weight_packed, self.scales, self.w_dequant_buffer, self.group_size)
        batch_size, seq_len, _ = x.shape
        x_2d = x.view(-1, self.K)
        out = torch.matmul(x_2d, self.w_dequant_buffer.T)
        return out.view(batch_size, seq_len, self.N)
