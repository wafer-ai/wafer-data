import os
os.environ["CXX"] = "hipcc"
os.environ["TORCH_ROCM_ARCH"] = "gfx942"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hip_source = r"""
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void int4_dequant_gemm_kernel(
    const float *x_data, 
    const uint8_t *w_packed_data, 
    const float *scales_data, 
    float *out_data, 
    int M, int N, int K, int group_size
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const float *x_row = x_data + m * K;
    const uint8_t *w_row_packed = w_packed_data + n * (K / 2);
    const float *scale_row = scales_data + n * (K / group_size);

    double acc = 0.0;
    int num_groups = K / group_size;
    for (int g = 0; g < num_groups; ++g) {
        double scale_d = (double)scale_row[g];
        int k_start = g * group_size;
        for (int i = 0; i < group_size; i += 2) {
            int k0 = k_start + i;
            int k1 = k0 + 1;
            if (k1 >= K) break;
            uint8_t byte = w_row_packed[k0 / 2];
            double w0_qd = (double)(byte & 0x0F) - 8.0;
            double w1_qd = (double)((byte >> 4) & 0x0F) - 8.0;
            acc += (double)x_row[k0] * scale_d * w0_qd;
            acc += (double)x_row[k1] * scale_d * w1_qd;
        }
    }
    out_data[m * N + n] = (float)acc;
}

torch::Tensor int4_linear_hip(
    torch::Tensor x_fp32, 
    torch::Tensor weight_packed, 
    torch::Tensor scales_fp32, 
    int64_t group_size
) {
    int64_t M = x_fp32.size(0);
    int64_t K = x_fp32.size(1);
    int64_t N = weight_packed.size(0);
    int64_t K_half = weight_packed.size(1);

    torch::Tensor out = torch::empty({M, N}, x_fp32.options());

    const int64_t threads = 32;
    dim3 block(static_cast<unsigned int>(threads), static_cast<unsigned int>(threads));
    dim3 grid(
        static_cast<unsigned int>((M + threads - 1) / threads),
        static_cast<unsigned int>((N + threads - 1) / threads)
    );

    int4_dequant_gemm_kernel<<<grid, block>>>(
        x_fp32.data_ptr<float>(), 
        weight_packed.data_ptr<uint8_t>(), 
        scales_fp32.data_ptr<float>(), 
        out.data_ptr<float>(), 
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

        torch.manual_seed(42)
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
        x_2d_fp32 = x.view(-1, self.K).to(torch.float32)
        scales_fp32 = self.scales.to(torch.float32)
        out_2d_fp32 = self.int4_gemm.int4_linear_hip(x_2d_fp32, self.weight_packed, scales_fp32, self.group_size)
        return out_2d_fp32.to(torch.float16).view(batch_size, seq_len, self.N)


batch_size = 4
seq_len = 2048
K = 4096
N = 11008
group_size = 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]

def get_init_inputs():
    return [K, N, group_size]
