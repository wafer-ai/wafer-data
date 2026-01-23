
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

max_pool1d_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void max_pool1d_kernel(
    const float* input,
    float* output,
    int64_t* indices,
    int batch_size,
    int channels,
    int input_len,
    int output_len,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_len;

    if (idx < total_elements) {
        int n_c = idx / output_len;
        int o = idx % output_len;

        int input_offset = n_c * input_len;
        float max_val = -INFINITY;
        int64_t max_idx = -1;

        for (int k = 0; k < kernel_size; ++k) {
            int input_idx = o * stride - padding + k * dilation;
            float val = -INFINITY;
            if (input_idx >= 0 && input_idx < input_len) {
                val = input[input_offset + input_idx];
            }

            if (val > max_val) {
                max_val = val;
                max_idx = input_idx;
            }
        }
        
        if (max_idx == -1) {
            max_idx = 0; 
        }

        output[idx] = max_val;
        if (return_indices) {
            indices[idx] = max_idx;
        }
    }
}

std::vector<torch::Tensor> max_pool1d_hip(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices) {

    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_len = input.size(2);

    int output_len = (input_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_len}, input.options());
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, channels, output_len}, input.options().dtype(torch::kInt64));
    } else {
        indices = torch::empty({0}, input.options().dtype(torch::kInt64));
    }

    int total_elements = batch_size * channels * output_len;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    max_pool1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        channels,
        input_len,
        output_len,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices
    );

    return {output, indices};
}
"""

max_pool1d_module = load_inline(
    name="max_pool1d_hip",
    cpp_sources=max_pool1d_cpp_source,
    functions=["max_pool1d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, indices = max_pool1d_module.max_pool1d_hip(
            x.contiguous(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices
        )
        if self.return_indices:
            return output, indices
        else:
            return output

def get_inputs():
    batch_size = 64
    features = 192
    sequence_length = 65536
    x = torch.rand(batch_size, features, sequence_length).cuda()
    return [x]

def get_init_inputs():
    kernel_size = 8
    stride      = 1
    padding     = 4
    dilation    = 3
    return_indices = False
    return [kernel_size, stride, padding, dilation, return_indices]
