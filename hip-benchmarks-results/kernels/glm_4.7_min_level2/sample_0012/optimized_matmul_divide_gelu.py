import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_divide_gelu_cpp_source = """
#include <hip/hip_runtime.h>

#define TILE 16

__global__ void matmul_divide_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int output_size,
    float divisor
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    for (int k = 0; k < (input_size + TILE - 1) / TILE; k++) {
        int a_col = k * TILE + threadIdx.x;
        int b_row = k * TILE + threadIdx.y;

        if (row < batch_size && a_col < input_size) {
            As[threadIdx.y][threadIdx.x] = input[row * input_size + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < input_size && col < output_size) {
            Bs[threadIdx.y][threadIdx.x] = weight[b_row * output_size + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int n = 0; n < TILE; n++) {
            sum += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < batch_size && col < output_size) {
        sum = sum / divisor;
        float x = sum;
        float x_cube = x * x * x;
        float tanh_arg = 0.7978845608028654f * (x + 0.044715f * x_cube);
        float gelu_val = 0.5f * x * (1.0f + tanhf(tanh_arg));
        output[row * output_size + col] = gelu_val;
    }
}

torch::Tensor matmul_divide_gelu_hip(torch::Tensor input, torch::Tensor weight, float divisor) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(1);

    auto output = torch::zeros({batch_size, output_size}, input.options());

    dim3 block(TILE, TILE);
    dim3 grid((output_size + TILE - 1) / TILE, (batch_size + TILE - 1) / TILE);

    matmul_divide_gelu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size,
        divisor
    );

    return output;
}
"""

matmul_divide_gelu = load_inline(
    name="matmul_divide_gelu",
    cpp_sources=matmul_divide_gelu_cpp_source,
    functions=["matmul_divide_gelu_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor
        self.matmul_divide_gelu = matmul_divide_gelu
        self.register_buffer('weight', torch.randn(output_size, input_size))

    def forward(self, x):
        return self.matmul_divide_gelu.matmul_divide_gelu_hip(x, self.weight.t(), self.divisor)