
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gated_delta_recurrence_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <algorithm>

__global__ void gated_delta_recurrence_kernel(
    const float* __restrict__ q,      // (B, L, H, Dk)
    const float* __restrict__ k,      // (B, L, H, Dk)
    const float* __restrict__ v,      // (B, L, H, Dv)
    const float* __restrict__ alpha,  // (B, L, H)
    const float* __restrict__ beta,   // (B, L, H)
    float* __restrict__ o,            // (B, L, H, Dv)
    int B, int L, int H, int Dk, int Dv
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    
    // Use a fixed size for s_row in registers.
    // Dk and Dv are 128.
    float s_row[128];
    for (int j = 0; j < 128; ++j) {
        s_row[j] = 0.0f;
    }
    
    extern __shared__ float shared_mem[];
    float* k_t_shared = shared_mem;            // Size Dk
    float* q_t_shared = shared_mem + Dk;       // Size Dk

    for (int t = 0; t < L; ++t) {
        // Load k_t, q_t into shared memory using float4 if possible
        if (Dk % 4 == 0) {
            for (int j = tid; j < Dk / 4; j += blockDim.x) {
                float4 k4 = reinterpret_cast<const float4*>(k + ((b * L + t) * H + h) * Dk)[j];
                float4 q4 = reinterpret_cast<const float4*>(q + ((b * L + t) * H + h) * Dk)[j];
                reinterpret_cast<float4*>(k_t_shared)[j] = k4;
                reinterpret_cast<float4*>(q_t_shared)[j] = q4;
            }
        } else {
            for (int j = tid; j < Dk; j += blockDim.x) {
                k_t_shared[j] = k[((b * L + t) * H + h) * Dk + j];
                q_t_shared[j] = q[((b * L + t) * H + h) * Dk + j];
            }
        }
        __syncthreads();
        
        float alpha_t = alpha[(b * L + t) * H + h];
        float beta_t = beta[(b * L + t) * H + h];

        if (tid < Dv) {
            // 1. Compute sk_i = S_i @ k_t
            float sk_i = 0.0f;
            for (int j = 0; j < Dk; ++j) {
                sk_i += s_row[j] * k_t_shared[j];
            }
            
            // 2. Compute error_i = sk_i - v_ti
            float v_ti = v[((b * L + t) * H + h) * Dv + tid];
            float error_i = sk_i - v_ti;
            
            // 3. Update S_i: S_i = alpha * S_i - beta * error_i * k_t
            for (int j = 0; j < Dk; ++j) {
                s_row[j] = alpha_t * s_row[j] - beta_t * error_i * k_t_shared[j];
            }
            
            // 4. Compute ot_i = S_i @ q_t
            float ot_i = 0.0f;
            for (int j = 0; j < Dk; ++j) {
                ot_i += s_row[j] * q_t_shared[j];
            }
            
            // Write ot_i
            o[((b * L + t) * H + h) * Dv + tid] = ot_i;
        }
        __syncthreads();
    }
}

torch::Tensor gated_delta_recurrence_hip(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, 
    torch::Tensor alpha, torch::Tensor beta
) {
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    alpha = alpha.contiguous();
    beta = beta.contiguous();

    int B = q.size(0);
    int L = q.size(1);
    int H = q.size(2);
    int Dk = q.size(3);
    int Dv = v.size(3);

    auto o = torch::empty_like(v);

    dim3 blocks(H, B);
    dim3 threads(std::max(Dk, Dv));

    int shared_mem_size = 2 * Dk * sizeof(float);

    gated_delta_recurrence_kernel<<<blocks, threads, shared_mem_size>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        alpha.data_ptr<float>(), beta.data_ptr<float>(),
        o.data_ptr<float>(), B, L, H, Dk, Dv
    );

    return o;
}
"""

gated_delta_recurrence_lib = load_inline(
    name="gated_delta_recurrence",
    cpp_sources=gated_delta_recurrence_source,
    functions=["gated_delta_recurrence_hip"],
    verbose=True,
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

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk) * self.scale
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        alpha = torch.sigmoid(self.a_proj(x))
        beta = torch.sigmoid(self.b_proj(x))

        o = gated_delta_recurrence_lib.gated_delta_recurrence_hip(q, k, v, alpha, beta)

        o = self.o_norm(o)
        g = torch.sigmoid(self.g_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = (o * g).reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o

