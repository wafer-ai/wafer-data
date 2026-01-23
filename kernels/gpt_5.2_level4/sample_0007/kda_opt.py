import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused FP32 HIP kernel for Kimi Delta Attention recurrence.
# Specializes to Dk=Dv=128 (KernelBench config).

os.environ.setdefault("CXX", "hipcc")
os.environ.setdefault("CC", "hipcc")

_kda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float block_reduce_sum_64(float v, float* shared, int tid) {
    shared[tid] = v;
    __syncthreads();
    if (tid < 32) shared[tid] += shared[tid + 32];
    __syncthreads();
    if (tid < 16) shared[tid] += shared[tid + 16];
    __syncthreads();
    if (tid < 8) shared[tid] += shared[tid + 8];
    __syncthreads();
    if (tid < 4) shared[tid] += shared[tid + 4];
    __syncthreads();
    if (tid < 2) shared[tid] += shared[tid + 2];
    __syncthreads();
    if (tid < 1) shared[tid] += shared[tid + 1];
    __syncthreads();
    return shared[0];
}

// 64-thread block, each thread handles two columns j and j+64.
__global__ void kda_recurrence_fwd_f32_128_2col(
    const float* __restrict__ q,     // [B, T, H, 128]
    const float* __restrict__ k,     // [B, T, H, 128]
    const float* __restrict__ v,     // [B, T, H, 128]
    const float* __restrict__ a,     // [B, T, H, 128]
    const float* __restrict__ beta,  // [B, T, H]
    float* __restrict__ out,         // [B, T, H, 128]
    int B, int T, int H
) {
    const int b = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int i = (int)blockIdx.z;  // value channel
    const int tid = (int)threadIdx.x; // 0..63

    const int j0 = tid;
    const int j1 = tid + 64;

    float S0 = 0.0f;
    float S1 = 0.0f;

    __shared__ float red[64];
    __shared__ float sh_a;
    __shared__ float sh_v;
    __shared__ float sh_beta;

    const int stride = 128;
    const long long base_b = ((long long)b) * (long long)T * (long long)H;

    for (int t = 0; t < T; ++t) {
        const long long base_bth = base_b + (long long)t * (long long)H + (long long)h;

        if (tid == 0) {
            sh_a = a[base_bth * stride + i];
            sh_v = v[base_bth * stride + i];
            sh_beta = beta[base_bth];
        }
        __syncthreads();

        const float a_i = sh_a;
        const float v_i = sh_v;
        const float beta_t = sh_beta;

        const float k0 = k[base_bth * stride + j0];
        const float k1 = k[base_bth * stride + j1];
        const float q0 = q[base_bth * stride + j0];
        const float q1 = q[base_bth * stride + j1];

        // dot1 = sum_j S[j] * k[j]
        const float dot1 = block_reduce_sum_64(S0 * k0 + S1 * k1, red, tid);
        const float err = dot1 - v_i;

        // Update both columns
        const float upd = beta_t * err;
        S0 = a_i * S0 - upd * k0;
        S1 = a_i * S1 - upd * k1;

        // dot2 = sum_j S[j] * q[j]
        const float dot2 = block_reduce_sum_64(S0 * q0 + S1 * q1, red, tid);

        if (tid == 0) {
            out[base_bth * stride + i] = dot2;
        }
        __syncthreads();
    }
}

torch::Tensor kda_recurrence_hip(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                torch::Tensor a, torch::Tensor beta) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && a.is_cuda() && beta.is_cuda(), "all inputs must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 && v.dtype() == torch::kFloat32 && a.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32,
                "all inputs must be float32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && a.is_contiguous() && beta.is_contiguous(), "all inputs must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && a.dim() == 4, "q/k/v/a must be 4D");
    TORCH_CHECK(beta.dim() == 3, "beta must be 3D");

    const int B = (int)q.size(0);
    const int T = (int)q.size(1);
    const int H = (int)q.size(2);
    const int Dk = (int)q.size(3);
    const int Dv = (int)v.size(3);

    TORCH_CHECK(Dk == 128 && Dv == 128, "kernel specializes to Dk=Dv=128");

    auto out = torch::empty({B, T, H, Dv}, q.options());

    const dim3 block(64, 1, 1);
    const dim3 grid(B, H, Dv);

    const auto stream = c10::cuda::getCurrentCUDAStream();
    hipStream_t hip_stream = (hipStream_t)stream.stream();

    kda_recurrence_fwd_f32_128_2col<<<grid, block, 0, hip_stream>>>(
        (const float*)q.data_ptr<float>(),
        (const float*)k.data_ptr<float>(),
        (const float*)v.data_ptr<float>(),
        (const float*)a.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, T, H
    );

    return out;
}
"""

_kda_ext = load_inline(
    name="kda_recurrence_ext",
    cpp_sources=_kda_src,
    functions=["kda_recurrence_hip"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        use_dplr: bool = False,
        dplr_rank: int = 4,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_dplr = use_dplr
        self.dplr_rank = dplr_rank
        self.use_short_conv = use_short_conv

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        if use_dplr:
            self.l_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)
            self.r_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)

        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        if use_short_conv:
            self.q_conv = nn.Conv1d(
                num_heads * head_dim_qk,
                num_heads * head_dim_qk,
                kernel_size=conv_kernel_size,
                groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1,
            )
            self.k_conv = nn.Conv1d(
                num_heads * head_dim_qk,
                num_heads * head_dim_qk,
                kernel_size=conv_kernel_size,
                groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1,
            )
            self.v_conv = nn.Conv1d(
                num_heads * head_dim_v,
                num_heads * head_dim_v,
                kernel_size=conv_kernel_size,
                groups=num_heads * head_dim_v,
                padding=conv_kernel_size - 1,
            )

        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :T].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :T].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q = q.view(B, T, self.num_heads, self.head_dim_qk)
        k = k.view(B, T, self.num_heads, self.head_dim_qk)
        v = v.view(B, T, self.num_heads, self.head_dim_v)

        a = torch.sigmoid(self.a_proj(x)).view(B, T, self.num_heads, self.head_dim_v)
        beta = torch.sigmoid(self.b_proj(x)).view(B, T, self.num_heads)

        if self.use_dplr:
            _ = self.l_proj(x).view(B, T, self.num_heads, self.dplr_rank)
            _ = self.r_proj(x).view(B, T, self.num_heads, self.dplr_rank)

        k = k * self.scale

        o = _kda_ext.kda_recurrence_hip(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            a.contiguous(),
            beta.contiguous(),
        )

        o = self.o_norm(o)
        g = torch.sigmoid(self.g_proj(x)).view(B, T, self.num_heads, self.head_dim_v)
        o = o * g
        o = o.reshape(B, T, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)
        return o


batch_size = 4
seq_len = 2048
hidden_size = 2048
num_heads = 16
head_dim_qk = 128
head_dim_v = 128


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
