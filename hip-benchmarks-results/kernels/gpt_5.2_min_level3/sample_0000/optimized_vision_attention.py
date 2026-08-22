import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force hipcc for ROCm builds
os.environ.setdefault("CXX", "hipcc")

# Fused: residual add + LayerNorm over last dim (C)
# Inputs are expected contiguous with shape [N, C] where N = L*B.

fused_add_layernorm_src = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

__global__ void fused_add_layernorm_kernel(
    const float* __restrict__ a,  // attn_output [N, C]
    const float* __restrict__ b,  // x           [N, C]
    const float* __restrict__ gamma, // [C]
    const float* __restrict__ beta,  // [C]
    float* __restrict__ out,       // [N, C]
    int N, int C, float eps)
{
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (row >= N) return;

    // One block per row, C threads (assume C <= 1024). For this model C=128.
    extern __shared__ float shmem[];
    float* sh_sum = shmem;
    float* sh_sumsq = shmem + blockDim.x;

    float v = 0.0f;
    float x = 0.0f;
    int idx = row * C + tid;
    if (tid < C) {
        v = a[idx] + b[idx];
        x = v;
    }

    float psum = (tid < C) ? x : 0.0f;
    float psumsq = (tid < C) ? x * x : 0.0f;

    sh_sum[tid] = psum;
    sh_sumsq[tid] = psumsq;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
            sh_sumsq[tid] += sh_sumsq[tid + stride];
        }
        __syncthreads();
    }

    float mean = sh_sum[0] / (float)C;
    float var = sh_sumsq[0] / (float)C - mean * mean;
    float inv_std = rsqrtf(var + eps);

    if (tid < C) {
        float y = (v - mean) * inv_std;
        y = y * gamma[tid] + beta[tid];
        out[idx] = y;
    }
}

torch::Tensor fused_add_layernorm_hip(
    torch::Tensor attn_out,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps)
{
    TORCH_CHECK(attn_out.is_cuda(), "attn_out must be CUDA/HIP");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP");
    TORCH_CHECK(attn_out.scalar_type() == torch::kFloat32, "FP32 only");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "FP32 only");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "FP32 only");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "FP32 only");

    attn_out = attn_out.contiguous();
    x = x.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();

    TORCH_CHECK(attn_out.dim() == 2, "attn_out must be [N, C]");
    TORCH_CHECK(x.sizes() == attn_out.sizes(), "x shape mismatch");
    int64_t N = attn_out.size(0);
    int64_t C = attn_out.size(1);
    TORCH_CHECK(gamma.numel() == C, "gamma size mismatch");
    TORCH_CHECK(beta.numel() == C, "beta size mismatch");

    auto out = torch::empty_like(attn_out);

    int threads = 1;
    while (threads < C) threads <<= 1;
    if (threads > 1024) threads = 1024;
    dim3 block(threads);
    dim3 grid((unsigned int)N);
    size_t shmem = sizeof(float) * threads * 2;

    hipStream_t stream = at::hip::getDefaultHIPStream();
    hipLaunchKernelGGL(fused_add_layernorm_kernel, grid, block, shmem, stream,
        attn_out.data_ptr<float>(), x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        out.data_ptr<float>(), (int)N, (int)C, (float)eps);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_layernorm_hip", &fused_add_layernorm_hip, "fused add + layernorm (HIP)");
}
"""

fused_add_layernorm = load_inline(
    name="fused_add_layernorm_ext",
    cpp_sources="",
    cuda_sources=fused_add_layernorm_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self._fused = fused_add_layernorm

    def forward(self, x):
        B, C, H, W = x.shape
        # (L, B, C)
        x_seq = x.view(B, C, H * W).permute(2, 0, 1).contiguous()
        attn_output, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)

        # Fused residual + LN over C
        L, B2, C2 = attn_output.shape
        y2d = self._fused.fused_add_layernorm_hip(
            attn_output.reshape(L * B2, C2),
            x_seq.reshape(L * B2, C2),
            self.norm.weight,
            self.norm.bias,
            float(self.norm.eps),
        )
        y = y2d.view(L, B2, C2)

        y = y.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return y


# Keep same init/input helpers
embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [embed_dim, num_heads]
