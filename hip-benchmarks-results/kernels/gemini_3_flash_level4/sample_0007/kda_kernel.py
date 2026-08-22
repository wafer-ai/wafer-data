
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

kda_recurrence_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val, float* shared_mem, int dk, int head_dim_qk) {
    int lane = dk % 32;
    int wid = dk / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();
    
    // Assume dk < 1024, so wid < 32
    val = (dk < (head_dim_qk + 31) / 32) ? shared_mem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// Higher precision version using double for the state and intermediate sums
__global__ void kda_recurrence_kernel(
    const float* __restrict__ q,       // (B, T, H, DK)
    const float* __restrict__ k,       // (B, T, H, DK)
    const float* __restrict__ v,       // (B, T, H, DV)
    const float* __restrict__ a,       // (B, T, H, DV)
    const float* __restrict__ beta,    // (B, T, H)
    float* __restrict__ out,           // (B, T, H, DV)
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim_qk,
    int head_dim_v) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int dv = blockIdx.z;
    int dk = threadIdx.x;

    if (dk >= head_dim_qk) return;

    // Use double for the state to reduce accumulation error
    double s = 0.0;

    extern __shared__ float shared_mem[];

    for (int t = 0; t < seq_len; ++t) {
        int base_qk = ((b * seq_len + t) * num_heads + h) * head_dim_qk;
        int base_v = ((b * seq_len + t) * num_heads + h) * head_dim_v;
        int base_beta = (b * seq_len + t) * num_heads + h;

        float kt_dk = k[base_qk + dk];
        float dot_s_prev_k_val = (float)(s * (double)kt_dk);
        float dot_s_prev_k = blockReduceSum(dot_s_prev_k_val, shared_mem, dk, head_dim_qk);
        // blockReduceSum broadcasts the result to all threads only if you handle it
        // Wait, blockReduceSum only returns the correct sum to thread 0
        // We need to broadcast it.
        if (dk == 0) shared_mem[0] = dot_s_prev_k;
        __syncthreads();
        dot_s_prev_k = shared_mem[0];

        float vt = v[base_v + dv];
        float at = a[base_v + dv];
        float bt = beta[base_beta];

        double error = (double)dot_s_prev_k - (double)vt;
        s = (double)at * s - (double)bt * error * (double)kt_dk;

        float qt_dk = q[base_qk + dk];
        float ot_dv_val = (float)(s * (double)qt_dk);
        float ot_dv = blockReduceSum(ot_dv_val, shared_mem, dk, head_dim_qk);
        
        if (dk == 0) {
            out[base_v + dv] = ot_dv;
        }
        __syncthreads();
    }
}

torch::Tensor kda_recurrence_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta) {
    
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int head_dim_qk = q.size(3);
    int head_dim_v = v.size(3);

    auto out = torch::empty_like(v);

    dim3 grid(batch_size, num_heads, head_dim_v);
    dim3 block(head_dim_qk);
    // Shared memory for block reduction (max 32 warps)
    int shared_mem_size = 32 * sizeof(float);

    kda_recurrence_kernel<<<grid, block, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        a.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim_qk,
        head_dim_v
    );

    return out;
}
"""

kda_cuda = load_inline(
    name="kda_recurrence",
    cpp_sources=kda_recurrence_cpp_source,
    functions=["kda_recurrence_hip"],
    verbose=True,
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
            q = F.silu(self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2))
            k = F.silu(self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2))
            v = F.silu(self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2))

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).contiguous()
        k = (k * self.scale).view(batch_size, seq_len, self.num_heads, self.head_dim_qk).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()

        a = torch.sigmoid(self.a_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()
        beta = torch.sigmoid(self.b_proj(x)).view(batch_size, seq_len, self.num_heads).contiguous()

        o = kda_cuda.kda_recurrence_hip(q, k, v, a, beta)

        o = self.o_norm(o)
        g = torch.sigmoid(self.g_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o

