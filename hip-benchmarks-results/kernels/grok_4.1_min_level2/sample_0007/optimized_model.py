import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torch.utils.cpp_extension import load_inline

fused_cpp = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

const int TILE_N = 256;
const int KTILE = 32;

__global__ void fused_tiled_kernel(const float* A, const float* W, const float* b, float scale, float* C, int B, int N, int K) {
    __shared__ float sA[KTILE];
    __shared__ float sWB[TILE_N][KTILE];

    int b_idx = blockIdx.x;
    int nt_idx = blockIdx.y;
    int n_start = nt_idx * TILE_N;
    int tid = threadIdx.x;
    int j = n_start + tid;
    if (b_idx >= B || j >= N) return;

    float acc = b[j];

    for (int k_panel = 0; k_panel < K; k_panel += KTILE) {
        // Load input panel
        if (tid < KTILE) {
            sA[tid] = A[b_idx * K + k_panel + tid];
        }
        __syncthreads();

        // Load weight panel
        for (int lk = 0; lk < KTILE; ++lk) {
            sWB[tid][lk] = W[j * K + k_panel + lk];
        }
        __syncthreads();

        // Compute
#pragma unroll
        for (int lk = 0; lk < KTILE; ++lk) {
            acc += sA[lk] * sWB[tid][lk];
        }
    }

    float val = acc;
    float sig = 1.0f / (1.0f + expf(-val));
    C[b_idx * N + j] = val * sig * scale;
}
torch::Tensor fused_tiled_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale_t) {
    int B = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    auto output = torch::empty({B, N}, input.options());
    float scale = *scale_t.data_ptr<float>();
    dim3 block(TILE_N);
    dim3 grid(B, (N + TILE_N - 1) / TILE_N);
    fused_tiled_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), scale, output.data_ptr<float>(), B, N, K);
    return output;
}
"""

fused_op = load_inline(
    name="fused_tiled",
    cpp_sources=fused_cpp,
    functions=["fused_tiled_hip"],
    verbose=True,
)

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(out_features, dtype=torch.float32)))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        scale_t = torch.tensor([self.scaling_factor], dtype=torch.float32, device=x.device)
        return self.fused_op.fused_tiled_hip(x, self.weight, self.bias, scale_t)
