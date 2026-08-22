import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, dropout_p]

linear_cpp = """
#include <hip/hip_runtime.h>

__global__ void linear_kernel(const float *x, const float *w_t, const float *bias, float *out, int B, int K, int N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * N) return;
    int b = idx / N;
    int n = idx % N;
    float sum = 0.0f;
    constexpr int TILE_K = 32;
    for (int kt = 0; kt < K; kt += TILE_K) {
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            if (kt + kk < K) {
                sum += x[b * K + kt + kk] * w_t[(kt + kk) * N + n];
            }
        }
    }
    out[b * N + n] = sum + bias[n];
}

torch::Tensor linear_hip(torch::Tensor x, torch::Tensor w_t, torch::Tensor bias) {
    auto B = x.size(0);
    auto K = x.size(1);
    auto N = w_t.size(1);
    auto out_options = x.options();
    torch::Tensor out = torch::empty({B, N}, out_options);

    const int block_size = 256;
    const int64_t total_elements = B * N;
    const int64_t num_blocks = (total_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(linear_kernel, dim3((unsigned int)num_blocks), dim3(block_size), 0, 0,
                       x.data_ptr<float>(), w_t.data_ptr<float>(), bias.data_ptr<float>(),
                       out.data_ptr<float>(), (int)B, (int)K, (int)N);

    return out;
}
"""

linear_module = load_inline(
    name="linear_add",
    cpp_sources=linear_cpp,
    functions=["linear_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.reset_parameters()
        self.register_buffer('weight_t', self.weight.t().contiguous())
        self.dropout = nn.Dropout(dropout_p)
        self.linear_hip = linear_module.linear_hip

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        linear_out = self.linear_hip(x, self.weight_t, self.bias)
        x = self.dropout(linear_out)
        x = torch.softmax(x, dim=1)
        return x
