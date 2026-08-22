import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Force HIP compilation via hipcc in this environment
os.environ.setdefault("CXX", "hipcc")

# ------------------------------------------------------------
# HIP extension: fused (residual add + LayerNorm + reshape)
# Input: attn_out (L, B, C) contiguous FP32
#        residual (L, B, C) contiguous FP32
#        gamma/beta (C) FP32
# Output: (B, C, H, W) FP32 contiguous
# ------------------------------------------------------------

cpp_decl = r'''
#include <torch/extension.h>
// Declaration so the auto-generated pybind main.cpp can see the symbol.
torch::Tensor fused_add_layernorm_to_nchw_hip(
    torch::Tensor attn,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t H,
    int64_t W,
    double eps);
'''

hip_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

__device__ __forceinline__ void block_reduce_sum(float &sum, float &sumsq, float* sh_sum, float* sh_sumsq) {
    int tid = (int)threadIdx.x;
    sh_sum[tid] = sum;
    sh_sumsq[tid] = sumsq;
    __syncthreads();

    for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
            sh_sumsq[tid] += sh_sumsq[tid + offset];
        }
        __syncthreads();
    }
    sum = sh_sum[0];
    sumsq = sh_sumsq[0];
}

__global__ void fused_add_layernorm_to_nchw_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int L, int B, int C, int H, int W,
    float eps)
{
    // One block per (l, b)
    int token = (int)blockIdx.x; // [0, L*B)
    int l = token / B;
    int b = token - l * B;

    int h = l / W;
    int w = l - h * W;

    int base = (l * B + b) * C;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int c = (int)threadIdx.x; c < C; c += (int)blockDim.x) {
        float v = attn[base + c] + residual[base + c];
        sum += v;
        sumsq += v * v;
    }

    extern __shared__ float shmem[];
    float* sh_sum = shmem;
    float* sh_sumsq = shmem + blockDim.x;
    block_reduce_sum(sum, sumsq, sh_sum, sh_sumsq);

    float mean = sum / (float)C;
    float var = sumsq / (float)C - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // NCHW output: contiguous
    int hw = H * W;
    int out_spatial = h * W + w;
    int out_base = (b * C) * hw + out_spatial;

    for (int c = (int)threadIdx.x; c < C; c += (int)blockDim.x) {
        float v = attn[base + c] + residual[base + c];
        float y = (v - mean) * inv_std;
        y = y * gamma[c] + beta[c];
        out[out_base + c * hw] = y;
    }
}

torch::Tensor fused_add_layernorm_to_nchw_hip(
    torch::Tensor attn,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t H,
    int64_t W,
    double eps)
{
    CHECK_CUDA(attn);
    CHECK_CUDA(residual);
    CHECK_CUDA(gamma);
    CHECK_CUDA(beta);
    CHECK_FLOAT(attn);
    CHECK_FLOAT(residual);
    CHECK_FLOAT(gamma);
    CHECK_FLOAT(beta);
    CHECK_CONTIGUOUS(attn);
    CHECK_CONTIGUOUS(residual);
    CHECK_CONTIGUOUS(gamma);
    CHECK_CONTIGUOUS(beta);

    TORCH_CHECK(attn.dim() == 3, "attn must be (L,B,C)");
    TORCH_CHECK(residual.sizes() == attn.sizes(), "residual must match attn");

    int64_t L = attn.size(0);
    int64_t B = attn.size(1);
    int64_t C = attn.size(2);

    TORCH_CHECK(gamma.numel() == C && beta.numel() == C, "gamma/beta must have shape (C,)");
    TORCH_CHECK(H * W == L, "H*W must equal sequence length L");

    auto out = torch::empty({B, C, H, W}, attn.options());

    const int threads = 128; // tuned for C=128
    const int blocks = (int)(L * B);
    size_t shmem = (size_t)(2 * threads) * sizeof(float);

    hipLaunchKernelGGL(
        fused_add_layernorm_to_nchw_kernel,
        dim3(blocks), dim3(threads), shmem, 0,
        (const float*)attn.data_ptr<float>(),
        (const float*)residual.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (int)L, (int)B, (int)C, (int)H, (int)W,
        (float)eps);

    return out;
}
'''

_fused_ext = load_inline(
    name="vision_attn_fused_ln",
    cpp_sources=cpp_decl,
    cuda_sources=hip_src,
    functions=["fused_add_layernorm_to_nchw_hip"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # Maintain parameter/key compatibility with the reference
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self._fused = _fused_ext

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        # Make (L,B,C) contiguous (helps attention kernels and our fused LN)
        x_seq = x.view(B, C, L).permute(2, 0, 1).contiguous()

        # Key speed trick: don't materialize/return the gigantic attention weights
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        if not attn_out.is_contiguous():
            attn_out = attn_out.contiguous()

        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        y = self._fused.fused_add_layernorm_to_nchw_hip(attn_out, x_seq, gamma, beta, H, W, self.norm.eps)
        return y


# Benchmark metadata
embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [embed_dim, num_heads]
