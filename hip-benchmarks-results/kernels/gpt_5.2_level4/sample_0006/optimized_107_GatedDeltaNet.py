import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ROCm/HIP build
os.environ.setdefault("CXX", "hipcc")

# Kernel specializes for Dk=128, Dv=128 (the benchmark configuration).
# Parallelization: each block handles one (batch, head, row_tile). Each warp (64 threads)
# handles one row of S (length Dk=128 split as 2 columns per lane). State lives in registers.
# k and q are cached in shared per timestep to amortize global loads across warps.

hip_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    // AMD wavefront is 64 lanes.
    for (int offset = 32; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 64);
    }
    return v;
}

// ROWS_PER_BLOCK warps per block; each warp computes one row of S.
// Dk=128 -> each lane owns 2 columns: lane and lane+64.

template<int ROWS_PER_BLOCK>
__global__ void gated_delta_forward_kernel128(
    const float* __restrict__ q,     // [B,T,H,128]
    const float* __restrict__ k,     // [B,T,H,128]
    const float* __restrict__ v,     // [B,T,H,128]
    const float* __restrict__ alpha, // [B,T,H]
    const float* __restrict__ beta,  // [B,T,H]
    float* __restrict__ out,         // [B,T,H,128]
    int B, int T, int H, float scale
) {
    constexpr int Dk = 128;
    constexpr int Dv = 128;

    int tiles_per_bh = (Dv + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    int idx = (int)blockIdx.x; // [0, B*H*tiles_per_bh)
    int tile = idx % tiles_per_bh;
    int bh = idx / tiles_per_bh;
    int b = bh / H;
    int h = bh - b * H;

    int tid = (int)threadIdx.x;
    int warp = tid >> 6;     // /64
    int lane = tid & 63;     // %64

    int row = tile * ROWS_PER_BLOCK + warp;
    if (warp >= ROWS_PER_BLOCK || row >= Dv) return;

    // Per-thread state: two columns for this row.
    float s0 = 0.0f;
    float s1 = 0.0f;

    // Shared cache for k and q for the current timestep.
    __shared__ float k_sh[Dk];
    __shared__ float q_sh[Dk];
    __shared__ float a_sh;
    __shared__ float b_sh;

    for (int t = 0; t < T; ++t) {
        int base = (b * T + t) * H + h; // [B,T,H]
        const float* k_ptr = k + ((size_t)base * (size_t)Dk);
        const float* q_ptr = q + ((size_t)base * (size_t)Dk);
        const float* v_ptr = v + ((size_t)base * (size_t)Dv);
        float* out_ptr = out + ((size_t)base * (size_t)Dv);

        // Cooperative load of k and q into shared (first 128 threads)
        if (tid < Dk) {
            k_sh[tid] = k_ptr[tid] * scale;
            q_sh[tid] = q_ptr[tid];
        }
        if (tid == 0) {
            a_sh = alpha[base];
            b_sh = beta[base];
        }
        __syncthreads();

        float a = a_sh;
        float bt = b_sh;

        // S @ k for this row
        float k0 = k_sh[lane];
        float k1 = k_sh[lane + 64];
        float partial = fmaf(s0, k0, s1 * k1);
        float sum = warp_reduce_sum(partial);

        // error = sum - v[row]
        float err;
        if (lane == 0) {
            err = sum - v_ptr[row];
        }
        err = __shfl(err, 0, 64);

        // Update state and compute output dot with q in the same pass
        float new_s0 = fmaf(a, s0, -bt * err * k0);
        float new_s1 = fmaf(a, s1, -bt * err * k1);
        s0 = new_s0;
        s1 = new_s1;

        float q0 = q_sh[lane];
        float q1 = q_sh[lane + 64];
        float opart = fmaf(new_s0, q0, new_s1 * q1);
        float osum = warp_reduce_sum(opart);
        if (lane == 0) {
            out_ptr[row] = osum;
        }

        __syncthreads();
    }
}

torch::Tensor gated_delta_forward_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta,
    double scale
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA/HIP tensor");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.dtype() == torch::kFloat32 && v.dtype() == torch::kFloat32, "k,v must be float32");
    TORCH_CHECK(alpha.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32, "alpha,beta must be float32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && alpha.is_contiguous() && beta.is_contiguous(), "all inputs must be contiguous");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must be 4D");
    TORCH_CHECK(alpha.dim() == 3 && beta.dim() == 3, "alpha,beta must be 3D");

    int B = (int)q.size(0);
    int T = (int)q.size(1);
    int H = (int)q.size(2);
    int Dk = (int)q.size(3);
    int Dv = (int)v.size(3);

    TORCH_CHECK(Dk == 128, "This optimized kernel requires head_dim_qk=128");
    TORCH_CHECK(Dv == 128, "This optimized kernel requires head_dim_v=128");
    TORCH_CHECK(k.size(0) == B && k.size(1) == T && k.size(2) == H && k.size(3) == Dk, "k shape mismatch");
    TORCH_CHECK(v.size(0) == B && v.size(1) == T && v.size(2) == H && v.size(3) == Dv, "v shape mismatch");
    TORCH_CHECK(alpha.size(0) == B && alpha.size(1) == T && alpha.size(2) == H, "alpha shape mismatch");
    TORCH_CHECK(beta.size(0) == B && beta.size(1) == T && beta.size(2) == H, "beta shape mismatch");

    auto out = torch::empty({B, T, H, Dv}, q.options());

    constexpr int ROWS_PER_BLOCK = 8; // 8 warps -> 512 threads
    int tiles_per_bh = (Dv + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    dim3 grid((unsigned int)(B * H * tiles_per_bh));
    dim3 block(ROWS_PER_BLOCK * 64);

    hipStream_t stream = at::hip::getDefaultHIPStream();
    hipLaunchKernelGGL(
        (gated_delta_forward_kernel128<ROWS_PER_BLOCK>),
        grid,
        block,
        0,
        stream,
        (const float*)q.data_ptr<float>(),
        (const float*)k.data_ptr<float>(),
        (const float*)v.data_ptr<float>(),
        (const float*)alpha.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, T, H, (float)scale
    );

    return out;
}
"""

_gated_delta_ext = load_inline(
    name="gated_delta_ext_v2",
    cpp_sources=hip_src,
    functions=["gated_delta_forward_hip"],
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
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_short_conv = use_short_conv

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

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
        self._ext = _gated_delta_ext

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

        q = q.view(B, T, self.num_heads, self.head_dim_qk).contiguous()
        k = k.view(B, T, self.num_heads, self.head_dim_qk).contiguous()
        v = v.view(B, T, self.num_heads, self.head_dim_v).contiguous()

        alpha = torch.sigmoid(self.a_proj(x)).contiguous()
        beta = torch.sigmoid(self.b_proj(x)).contiguous()

        o = self._ext.gated_delta_forward_hip(q, k, v, alpha, beta, float(self.scale))

        o = self.o_norm(o)

        g = torch.sigmoid(self.g_proj(x))
        g = g.view(B, T, self.num_heads, self.head_dim_v)
        o = o * g

        o = o.reshape(B, T, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)
        return o


# Configuration matching typical LLM settings
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
