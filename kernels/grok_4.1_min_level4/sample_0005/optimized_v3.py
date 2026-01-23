import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hip_source = r"""
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void int4_dequant_gemm_kernel(
    const __half *x_data, 
    const uint8_t *w_packed_data, 
    const __half *scales_data, 
    __half *out_data, 
    int M, int N, int K, int group_size
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const __half *x_row = x_data + m * K;
    const uint8_t *w_row_packed = w_packed_data + n * (K / 2);
    const __half *scale_row = scales_data + n * (K / group_size);

    float acc = 0.0f;
    int num_groups = K / group_size;
    for (int g = 0; g < num_groups; ++g) {
        float scale_f = __half2float(scale_row[g]);
        int k_start = g * group_size;
        for (int i = 0; i < group_size; i += 2) {
            int k0 = k_start + i;
            int k1 = k0 + 1;
            if (k1 >= K) break;
            uint8_t byte = w_row_packed[k0 / 2];
            float w0_qf = (float)(byte & 0x0F) - 8.0f;
            float w1_qf = (float)((byte >> 4) & 0x0F) - 8.0f;
            acc += __half2float(x_row[k0]) * scale_f * w0_qf;
            acc += __half2float(x_row[k1]) * scale_f * w1_qf;
        }
    }
    out_data[m * N + n] = __float2half(acc);
}

torch::Tensor int4_linear_hip(
    torch::Tensor x, 
    torch::Tensor weight_packed, 
    torch::Tensor scales, 
    int64_t group_size
) {
    int64_t M = x.size(0);
    int64_t K = x.size(1);
    int64_t N = weight_packed.size(0);
    int64_t K_half = weight_packed.size(1);

    torch::Tensor out = torch::empty({M, N}, x.options());

    const int64_t threads = 32;
    dim3 block(static_cast<unsigned int>(threads), static_cast<unsigned int>(threads));
    dim3 grid(
        static_cast<unsigned int>((M + threads - 1) / threads),
        static_cast<unsigned int>((N + threads - 1) / threads)
    );

    int4_dequant_gemm_kernel<<<grid, block>>>(
        x.data_ptr<__half>(), 
        weight_packed.data_ptr<uint8_t>(), 
        scales.data_ptr<__half>(), 
        out.data_ptr<__half>(), 
        static_cast<int>(M), 
        static_cast<int>(N), 
        static_cast<int>(K), 
        static_cast<int>(group_size)
    );

    return out;
}
"""

int4_gemm = load_inline(
    name="int4_gemm",
    cpp_sources=hip_source,
    functions=["int4_linear_hip"],
    verbose=True,
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
        out_2d = self.int4_gemm.int4_linear_hip(x_2d, self.weight_packed, self.scales, self.group_size)
        return out_2d.view(batch_size, seq_len, self.N)


batch_size = 4
seq_len = 2048
K = 4096
N = 11008
group_size = 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]

def get_init_inputs():
    return [K, N, group_size]
