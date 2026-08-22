import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

kda_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>

__global__ void kda_kernel(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    const float* a_ptr,
    const float* beta_ptr,
    float* o_ptr,
    int B, int T, int H
) {
    constexpr int D = 128;
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_states = B * H * D;
    if (state_idx >= total_states) return;
    int b = state_idx / (H * D);
    int temp = state_idx % (H * D);
    int h = temp / D;
    int chan = temp % D;

    int stride_seq_head = H * D;
    int stride_beta = H;

    int qk_base = b * T * stride_seq_head + h * D;
    int dv_base = b * T * stride_seq_head + h * D + chan;
    int beta_base = b * T * stride_beta + h;

    float local_s[D] = {};
    float kt[D];

    for (int t = 0; t < T; t++) {
#pragma unroll
        for (int j = 0; j < D; j++) {
            int kidx = qk_base + t * stride_seq_head + j;
            kt[j] = k_ptr[kidx];
        }

        float delta = 0.0f;
#pragma unroll
        for (int j = 0; j < D; j++) {
            delta += local_s[j] * kt[j];
        }

        int vidx = dv_base + t * stride_seq_head;
        float vt = v_ptr[vidx];
        float at = a_ptr[vidx];
        float betat = beta_ptr[beta_base + t * stride_beta];
        float error = delta - vt;

#pragma unroll
        for (int j = 0; j < D; j++) {
            local_s[j] = at * local_s[j] - betat * error * kt[j];
        }

        float ot = 0.0f;
#pragma unroll
        for (int j = 0; j < D; j++) {
            int qidx = qk_base + t * stride_seq_head + j;
            float qt = q_ptr[qidx];
            ot += local_s[j] * qt;
        }

        int oidx = dv_base + t * stride_seq_head;
        o_ptr[oidx] = ot;
    }
}

torch::Tensor compute_kda_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
) {
    auto B = q.size(0);
    auto T = q.size(1);
    auto H = q.size(2);
    auto Dq = q.size(3);
    auto Dv = v.size(3);

    TORCH_CHECK(Dq == Dv && Dq == 128, "KDA kernel only supports head_dim_qk = head_dim_v = 128");

    auto o = torch::empty({B, T, H, Dv}, q.options());

    const int block_size = 256;
    dim3 threads(block_size);
    dim3 blocks((B * H * Dv + block_size - 1) / block_size);

    kda_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        a.data_ptr<float>(),
        beta.data_ptr<float>(),
        o.data_ptr<float>(),
        (int)B, (int)T, (int)H
    );

    TORCH_CHECK(hipGetLastError() == hipSuccess, "kda_kernel launch failed: ", hipGetErrorString(hipGetLastError()));
    hipDeviceSynchronize();

    return o;
}
"""

kda = load_inline(
    name="kda",
    cpp_sources=kda_cpp,
    functions=["compute_kda_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized Kimi Delta Attention with custom HIP kernel for the recurrent state update.
    """

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

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        # Channel-wise gating
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

        self.custom_kda = kda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

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

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        a = torch.sigmoid(self.a_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        beta = torch.sigmoid(self.b_proj(x))

        if self.use_dplr:
            l = self.l_proj(x).view(batch_size, seq_len, self.num_heads, self.dplr_rank)
            r = self.r_proj(x).view(batch_size, seq_len, self.num_heads, self.dplr_rank)

        k = k * self.scale

        # Make contiguous for kernel
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        beta = beta.contiguous()

        # Custom HIP kernel
        o = self.custom_kda.compute_kda_hip(q, k, v, a, beta)

        o = self.o_norm(o)
        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
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
