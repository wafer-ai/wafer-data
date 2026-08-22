import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

kda_cpp_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

static inline __device__ int64_t idx4(int64_t a, int64_t b, int64_t c, int64_t d,
                                     int64_t B, int64_t C, int64_t D) {
    return ((a * B + b) * C + c) * D + d;
}

__global__ void kda_recurrence_fwd_kernel(
    const float* __restrict__ q,    // [B,T,H,Dk]
    const float* __restrict__ k,    // [B,T,H,Dk]
    const float* __restrict__ v,    // [B,T,H,Dv]
    const float* __restrict__ a,    // [B,T,H,Dv]
    const float* __restrict__ beta, // [B,T,H]
    float* __restrict__ S,          // [B,H,Dv,Dk]
    float* __restrict__ scratch,    // [B,H,Dv]
    float* __restrict__ out,        // [B,T,H,Dv]
    int B, int T, int H, int Dk, int Dv)
{
    int b = (int)blockIdx.x;
    int h = (int)blockIdx.y;

    extern __shared__ float smem[];
    float* qv = smem;              // Dk
    float* kv = qv + Dk;           // Dk
    float* betp = kv + Dk;         // 1 float

    float* Sbh = S + (((b * H + h) * Dv) * Dk);
    float* scr = scratch + ((b * H + h) * Dv);

    for (int t = 0; t < T; ++t) {
        if (threadIdx.x < Dk) {
            int j = threadIdx.x;
            qv[j] = q[idx4(b, t, h, j, T, H, Dk)];
            kv[j] = k[idx4(b, t, h, j, T, H, Dk)];
        }
        if (threadIdx.x == 0) {
            *betp = beta[((b * T + t) * H + h)];
        }
        __syncthreads();

        float bet = *betp;

        // Pass 1: err[i] = (S @ k)[i] - v[i]
        if (threadIdx.x < Dv) {
            int i = threadIdx.x;
            const float* Si = Sbh + i * Dk;
            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 128; ++j) {
                if (j < Dk) acc = fmaf(Si[j], kv[j], acc);
            }
            float vv = v[idx4(b, t, h, i, T, H, Dv)];
            scr[i] = acc - vv;
        }
        __syncthreads();

        // Pass 2: update S
        for (int idx = threadIdx.x; idx < Dv * Dk; idx += blockDim.x) {
            int i = idx / Dk;
            int j = idx - i * Dk;
            float ai = a[idx4(b, t, h, i, T, H, Dv)];
            float e = scr[i];
            float kj = kv[j];
            float s = Sbh[idx];
            Sbh[idx] = fmaf(ai, s, -bet * e * kj);
        }
        __syncthreads();

        // Pass 3: output
        if (threadIdx.x < Dv) {
            int i = threadIdx.x;
            const float* Si = Sbh + i * Dk;
            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 128; ++j) {
                if (j < Dk) acc = fmaf(Si[j], qv[j], acc);
            }
            out[idx4(b, t, h, i, T, H, Dv)] = acc;
        }
        __syncthreads();
    }
}

torch::Tensor kda_recurrence_fwd(torch::Tensor q,
                                torch::Tensor k,
                                torch::Tensor v,
                                torch::Tensor a,
                                torch::Tensor beta) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA/HIP");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && a.is_contiguous() && beta.is_contiguous(), "contiguous inputs");

    int B = (int)q.size(0);
    int T = (int)q.size(1);
    int H = (int)q.size(2);
    int Dk = (int)q.size(3);
    int Dv = (int)v.size(3);

    auto out = torch::empty({B, T, H, Dv}, q.options());
    auto S = torch::zeros({B, H, Dv, Dk}, q.options());
    auto scratch = torch::empty({B, H, Dv}, q.options());

    dim3 grid(B, H, 1);
    int threads = 256;
    size_t shmem = sizeof(float) * (Dk + Dk + 1);

    hipLaunchKernelGGL(kda_recurrence_fwd_kernel,
                      grid, dim3(threads), shmem, 0,
                      (const float*)q.data_ptr<float>(),
                      (const float*)k.data_ptr<float>(),
                      (const float*)v.data_ptr<float>(),
                      (const float*)a.data_ptr<float>(),
                      (const float*)beta.data_ptr<float>(),
                      (float*)S.data_ptr<float>(),
                      (float*)scratch.data_ptr<float>(),
                      (float*)out.data_ptr<float>(),
                      B, T, H, Dk, Dv);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kda_recurrence_fwd", &kda_recurrence_fwd, "KDA recurrence forward (HIP)");
}
"""

kda_ext = load_inline(
    name="kda_recurrence_ext",
    cpp_sources=kda_cpp_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
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
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.k_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.v_conv = nn.Conv1d(
                num_heads * head_dim_v, num_heads * head_dim_v,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_v,
                padding=conv_kernel_size - 1
            )

        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).contiguous()
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()

        a = torch.sigmoid(self.a_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()
        beta = torch.sigmoid(self.b_proj(x)).view(batch_size, seq_len, self.num_heads).contiguous()

        k = (k * self.scale).contiguous()

        o = kda_ext.kda_recurrence_fwd(q, k, v, a, beta)

        o = self.o_norm(o)
        g = torch.sigmoid(self.g_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)
        return o


def get_inputs():
    batch_size = 4
    seq_len = 2048
    hidden_size = 2048
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)]


def get_init_inputs():
    hidden_size = 2048
    num_heads = 16
    head_dim_qk = 128
    head_dim_v = 128
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
