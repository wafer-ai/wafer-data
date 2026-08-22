import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

sum_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void general_sum_reduce_kernel(const float* input, float* output,
  int size0, int size1, int size2,
  int stride0, int stride1, int stride2,
  int reduce_dim, int reduce_size,
  int total_out) {
    int flat_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_out >= total_out) return;
    float s = 0.0f;
    if (reduce_dim == 0) {
      int coord1 = flat_out / size2;
      int coord2 = flat_out % size2;
      int base = coord1 * stride1 + coord2 * stride2;
      const int unroll = 4;
      for (int k = 0; k < reduce_size; k += unroll) {
        if (k + 0 < reduce_size) s += input[base + (k + 0) * stride0];
        if (k + 1 < reduce_size) s += input[base + (k + 1) * stride0];
        if (k + 2 < reduce_size) s += input[base + (k + 2) * stride0];
        if (k + 3 < reduce_size) s += input[base + (k + 3) * stride0];
      }
    } else if (reduce_dim == 1) {
      int coord0 = flat_out / size2;
      int coord2 = flat_out % size2;
      int base = coord0 * stride0 + coord2 * stride2;
      const int unroll = 4;
      for (int k = 0; k < reduce_size; k += unroll) {
        if (k + 0 < reduce_size) s += input[base + (k + 0) * stride1];
        if (k + 1 < reduce_size) s += input[base + (k + 1) * stride1];
        if (k + 2 < reduce_size) s += input[base + (k + 2) * stride1];
        if (k + 3 < reduce_size) s += input[base + (k + 3) * stride1];
      }
    } else if (reduce_dim == 2) {
      int coord0 = flat_out / size1;
      int coord1 = flat_out % size1;
      int base = coord0 * stride0 + coord1 * stride1;
      const int unroll = 4;
      for (int k = 0; k < reduce_size; k += unroll) {
        if (k + 0 < reduce_size) s += input[base + (k + 0) * stride2];
        if (k + 1 < reduce_size) s += input[base + (k + 1) * stride2];
        if (k + 2 < reduce_size) s += input[base + (k + 2) * stride2];
        if (k + 3 < reduce_size) s += input[base + (k + 3) * stride2];
      }
    }
    output[flat_out] = s;
}

torch::Tensor sum_reduce_hip(torch::Tensor x, int64_t dim_) {
    auto shape = x.sizes();
    if (shape.size() != 3) {
        throw std::runtime_error("Only support 3D tensors");
    }
    int rdim = static_cast<int>(dim_);
    if (rdim < 0 || rdim > 2) {
        throw std::runtime_error("dim out of range");
    }
    int input_sizes[3];
    input_sizes[0] = static_cast<int>(shape[0]);
    input_sizes[1] = static_cast<int>(shape[1]);
    input_sizes[2] = static_cast<int>(shape[2]);
    int reduce_size = input_sizes[rdim];
    int strides[3];
    strides[2] = 1;
    strides[1] = input_sizes[2];
    strides[0] = input_sizes[1] * input_sizes[2];
    int out_sizes[3] = {input_sizes[0], input_sizes[1], input_sizes[2]};
    out_sizes[rdim] = 1;
    std::vector<int64_t> out_shape_vec(3);
    for(int i=0; i<3; i++) out_shape_vec[i] = out_sizes[i];
    auto out_shape = torch::IntArrayRef(out_shape_vec);
    auto options = x.options();
    auto out = torch::zeros(out_shape, options);
    int total_out = out_sizes[0] * out_sizes[1] * out_sizes[2];
    const int threads = 1024;
    int blocks = (total_out + threads - 1) / threads;
    hipLaunchKernelGGL(general_sum_reduce_kernel, dim3(blocks), dim3(threads), 0, 0,
        x.data_ptr<float>(), out.data_ptr<float>(),
        input_sizes[0], input_sizes[1], input_sizes[2],
        strides[0], strides[1], strides[2],
        rdim, reduce_size, total_out);
    return out;
}
"""

sum_reduce = load_inline(
    name="sum_reduce",
    cpp_sources=sum_cpp_source,
    functions=["sum_reduce_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduce = sum_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduce.sum_reduce_hip(x, self.dim)
