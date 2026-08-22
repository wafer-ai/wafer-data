import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp = """
#include <hip/hip_runtime.h>
#include <cmath>

__device__ float gelu_fp32(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float d = tanhf(0.7978845608f * (x + 0.044715f * x3));
    return 0.5f * x * (1.0f + d);
}

__global__ void fused_matmul_bias_div_gelu_kernel(
    const float* x, const float* w, const float* b, float div, float* out,
    int B, int I, int O
) {
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 64;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b_start = by * TILE_M;
    int o_start = bx * TILE_N;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float acc = 0.0f;

    for (int kt = 0; kt < I; kt += TILE_K) {
        // Load A tile
        if (b_start + ty < B && kt + tx < I) {
            As[ty][tx] = x[(b_start + ty) * I + (kt + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile with multi-load for n
        constexpr int step_n = TILE_N / 16;  // blockDim.y =16
        #pragma unroll
        for (int sn = 0; sn < step_n; ++sn) {
            int ln = ty * step_n + sn;
            if (ln < TILE_N && o_start + ln < O && kt + tx < I) {
                Bs[tx][ln] = w[(o_start + ln) * I + (kt + tx)];
            } else {
                Bs[tx][ln] = 0.0f;
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll 4
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int batch = b_start + ty;
    int row = o_start + tx;
    if (batch < B && row < O) {
        acc += b[row];
        float val = acc / div;
        out[batch * O + row] = gelu_fp32(val);
    }
}

torch::Tensor fused_linear_hip(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor divisor
) {
    int B = x.size(0);
    int I = x.size(1);
    int O = weight.size(0);
    auto out = torch::zeros({B, O}, x.options());
    float divv = divisor.item<float>();

    constexpr int TILE_M = 16;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 64;
    dim3 block(TILE_K, TILE_M);  // 64 x 16
    dim3 grid((O + TILE_N - 1) / TILE_N, (B + TILE_M - 1) / TILE_M);

    fused_matmul_bias_div_gelu_kernel<<<grid, block>>> (
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        divv,
        out.data_ptr<float>(),
        B, I, O
    );

    return out;
}
"""

fused_module = load_inline(
    name="fused",
    cpp_sources=cpp,
    functions=["fused_linear_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused = fused_module

    def forward(self, x):
        weight = self.linear.weight
        bias = self.linear.bias
        divisor_t = torch.tensor(self.divisor, dtype=x.dtype, device=x.device)
        return self.fused.fused_linear_hip(x, weight, bias, divisor_t)

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, output_size, divisor]
