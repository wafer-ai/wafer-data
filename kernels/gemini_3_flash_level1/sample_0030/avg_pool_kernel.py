
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

avg_pool1d_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 512
#define ELEMENTS_PER_THREAD 4

__global__ void avg_pool1d_kernel_v4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding) {

    int n = blockIdx.z;
    int c = blockIdx.y;
    int out_start_idx = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int tid = threadIdx.x;
    
    extern __shared__ float shared_input[];

    int in_start_idx = out_start_idx * stride - padding;
    int shared_mem_size = BLOCK_SIZE * ELEMENTS_PER_THREAD * stride + kernel_size - 1;

    const float* input_ptr = input + (n * channels + c) * input_length;

    for (int i = tid; i < shared_mem_size; i += BLOCK_SIZE) {
        int input_idx = in_start_idx + i;
        if (input_idx >= 0 && input_idx < input_length) {
            shared_input[i] = input_ptr[input_idx];
        } else {
            shared_input[i] = 0.0f;
        }
    }

    __syncthreads();

    float inv_kernel_size = 1.0f / (float)kernel_size;
    float* output_ptr = output + (n * channels + c) * output_length;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int l_out = out_start_idx + tid + i * BLOCK_SIZE;
        if (l_out < output_length) {
            float sum = 0.0f;
            int shared_base = (tid + i * BLOCK_SIZE) * stride;
            for (int k = 0; k < kernel_size; ++k) {
                sum += shared_input[shared_base + k];
            }
            output_ptr[l_out] = sum * inv_kernel_size;
        }
    }
}

torch::Tensor avg_pool1d_hip(torch::Tensor input, int kernel_size, int stride, int padding) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_length}, input.options());

    dim3 block(BLOCK_SIZE);
    dim3 grid((output_length + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD), channels, batch_size);
    
    int shared_mem_size = (BLOCK_SIZE * ELEMENTS_PER_THREAD * stride + kernel_size - 1) * sizeof(float);

    avg_pool1d_kernel_v4<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding
    );

    return output;
}
"""

avg_pool1d_module = load_inline(
    name="avg_pool1d_v4",
    cpp_sources=avg_pool1d_source,
    functions=["avg_pool1d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool1d_hip = avg_pool1d_module.avg_pool1d_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool1d_hip(x, self.kernel_size, self.stride, self.padding)
