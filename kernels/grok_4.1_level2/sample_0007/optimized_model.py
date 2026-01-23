import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_cpp = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void swish_scale_kernel(const float* input, float scale, float* output, int64_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = input[idx];
    float sig = 1.0f / (1.0f + __expf(-x));
    output[idx] = x * sig * scale;
  }
}

torch::Tensor swish_scale_hip(torch::Tensor input, float scale) {
  auto output = torch::empty_like(input);
  int64_t size = input.numel();
  const int block_size = 1024;
  int grid_size = (size + block_size - 1) / block_size;
  swish_scale_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), scale, output.data_ptr<float>(), size);
  return output;
}
"""

swish_scale = load_inline(
    name="swish_scale",
    cpp_sources=swish_cpp,
    functions=["swish_scale_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.swish_scale = swish_scale

    def forward(self, x):
        x = self.linear(x)
        x = self.swish_scale.swish_scale_hip(x, float(self.scaling_factor))
        return x

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
