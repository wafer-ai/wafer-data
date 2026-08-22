import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

# Put HIP code in cpp_sources so the generated pybind stub sees the symbol.
hip_cpp_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float swish_f(float x) {
    return x / (1.0f + __expf(-x));
}

template<int GROUP_SIZE>
__global__ void swish_bias_groupnorm_kernel(
    const float* __restrict__ in,
    const float* __restrict__ add_bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ out,
    int B, int C, int G, float eps)
{
    int bg = (int)blockIdx.x; // 0 .. B*G-1
    int b = bg / G;
    int g = bg - b * G;

    int tid = (int)threadIdx.x; // 0..GROUP_SIZE-1
    constexpr int GS = GROUP_SIZE;

    int group_ch0 = g * GS;
    int c = group_ch0 + tid;

    float v = 0.0f;
    if (c < C) {
        float x = in[b * C + c];
        v = swish_f(x) + add_bias[c];
    }

    __shared__ float sh_sum[GS];
    __shared__ float sh_sumsq[GS];

    sh_sum[tid] = v;
    sh_sumsq[tid] = v * v;
    __syncthreads();

    for (int offset = GS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
            sh_sumsq[tid] += sh_sumsq[tid + offset];
        }
        __syncthreads();
    }

    float mean = sh_sum[0] * (1.0f / (float)GS);
    float var = sh_sumsq[0] * (1.0f / (float)GS) - mean * mean;
    float inv_std = rsqrtf(var + eps);

    if (c < C) {
        float y = (v - mean) * inv_std;
        y = y * gn_weight[c] + gn_bias[c];
        out[b * C + c] = y;
    }
}

torch::Tensor swish_bias_groupnorm_hip(torch::Tensor x, torch::Tensor add_bias, torch::Tensor gn_weight, torch::Tensor gn_bias, int64_t num_groups, double eps_d) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, C]");
    TORCH_CHECK(add_bias.is_cuda() && gn_weight.is_cuda() && gn_bias.is_cuda(), "params must be CUDA/HIP tensors");
    TORCH_CHECK(add_bias.scalar_type() == at::kFloat && gn_weight.scalar_type() == at::kFloat && gn_bias.scalar_type() == at::kFloat, "params must be float32");

    auto x_contig = x.contiguous();
    auto b_contig = add_bias.contiguous();
    auto w_contig = gn_weight.contiguous();
    auto gb_contig = gn_bias.contiguous();

    int B = (int)x_contig.size(0);
    int C = (int)x_contig.size(1);
    int G = (int)num_groups;

    TORCH_CHECK(C % G == 0, "C must be divisible by num_groups");
    int group_size = C / G;
    TORCH_CHECK(group_size == 64, "Optimized kernel assumes group_size=64");

    auto out = torch::empty_like(x_contig);

    dim3 block(64);
    dim3 grid((unsigned int)(B * G));
    float eps = (float)eps_d;

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    hipLaunchKernelGGL((swish_bias_groupnorm_kernel<64>), grid, block, 0, stream,
        (const float*)x_contig.data_ptr<float>(),
        (const float*)b_contig.data_ptr<float>(),
        (const float*)w_contig.data_ptr<float>(),
        (const float*)gb_contig.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, C, G, eps);

    return out;
}
"""

fused_mod = load_inline(
    name="swish_bias_groupnorm_ext",
    cpp_sources=hip_cpp_src,
    functions=["swish_bias_groupnorm_hip"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.fused = fused_mod

    def forward(self, x):
        x = self.matmul(x)
        return self.fused.swish_bias_groupnorm_hip(
            x,
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )


def get_inputs():
    batch_size = 32768
    in_features = 1024
    return [torch.rand(batch_size, in_features, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    in_features = 1024
    out_features = 4096
    num_groups = 64
    bias_shape = (out_features,)
    return [in_features, out_features, num_groups, bias_shape]
