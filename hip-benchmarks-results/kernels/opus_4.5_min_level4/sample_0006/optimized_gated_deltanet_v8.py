import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# v8: Tiled approach with registers for state elements
gated_deltanet_cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Each thread handles multiple elements of the state matrix (TILE_V rows)
#define TILE_V 4  // Each thread handles 4 rows of V

__global__ void gated_deltanet_recurrence_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int d_qk,
    int d_v
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    extern __shared__ float shared_mem[];
    float* k_shared = shared_mem;
    float* v_shared = k_shared + d_qk;
    float* S_k = v_shared + d_v;
    float* q_shared = S_k + d_v;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int state_size = d_v * d_qk;
    
    float* S = state + (batch_idx * num_heads + head_idx) * state_size;
    
    // Initialize state
    for (int i = tid; i < state_size; i += block_size) {
        S[i] = 0.0f;
    }
    __syncthreads();
    
    for (int t = 0; t < seq_len; t++) {
        const int qkv_base = ((batch_idx * seq_len + t) * num_heads + head_idx);
        const int ab_idx = (batch_idx * seq_len + t) * num_heads + head_idx;
        
        // Load k and v
        for (int i = tid; i < d_qk; i += block_size) {
            k_shared[i] = k[qkv_base * d_qk + i];
        }
        for (int i = tid; i < d_v; i += block_size) {
            v_shared[i] = v[qkv_base * d_v + i];
        }
        __syncthreads();
        
        const float alpha_t = alpha[ab_idx];
        const float beta_t = beta[ab_idx];
        
        // Compute S_k = S @ k - each thread handles d_v / block_size rows
        for (int i = tid; i < d_v; i += block_size) {
            float sum = 0.0f;
            const int base = i * d_qk;
            #pragma unroll 8
            for (int j = 0; j < d_qk; j++) {
                sum += S[base + j] * k_shared[j];
            }
            S_k[i] = sum;
        }
        __syncthreads();
        
        // Update state - each thread handles d_v*d_qk / block_size elements
        for (int idx = tid; idx < state_size; idx += block_size) {
            const int i = idx / d_qk;
            const int j = idx % d_qk;
            const float error_i = S_k[i] - v_shared[i];
            S[idx] = alpha_t * S[idx] - beta_t * error_i * k_shared[j];
        }
        __syncthreads();
        
        // Load q
        for (int i = tid; i < d_qk; i += block_size) {
            q_shared[i] = q[qkv_base * d_qk + i];
        }
        __syncthreads();
        
        // Output = S @ q
        const int out_base = qkv_base * d_v;
        for (int i = tid; i < d_v; i += block_size) {
            float sum = 0.0f;
            const int base = i * d_qk;
            #pragma unroll 8
            for (int j = 0; j < d_qk; j++) {
                sum += S[base + j] * q_shared[j];
            }
            output[out_base + i] = sum;
        }
        __syncthreads();
    }
}

torch::Tensor gated_deltanet_recurrence(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta
) {
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int d_qk = q.size(3);
    int d_v = v.size(3);
    
    auto output = torch::zeros({batch_size, seq_len, num_heads, d_v}, q.options());
    auto state = torch::zeros({batch_size, num_heads, d_v, d_qk}, q.options());
    
    dim3 grid(batch_size, num_heads);
    int block_size = 256;  // Back to 256
    
    size_t shared_mem_size = (2 * d_qk + 2 * d_v) * sizeof(float);
    
    gated_deltanet_recurrence_kernel<<<grid, block_size, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        alpha.data_ptr<float>(),
        beta.data_ptr<float>(),
        state.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        d_qk,
        d_v
    );
    
    return output;
}
"""

gated_deltanet_cpp_header = """
torch::Tensor gated_deltanet_recurrence(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta
);
"""

gated_deltanet_module = load_inline(
    name="gated_deltanet_v8",
    cpp_sources=gated_deltanet_cpp_header,
    cuda_sources=gated_deltanet_cpp_source,
    functions=["gated_deltanet_recurrence"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "-mcumode", "-mwavefrontsize64"],
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
        self.gated_deltanet = gated_deltanet_module

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

        alpha = torch.sigmoid(self.a_proj(x)).contiguous()
        beta = torch.sigmoid(self.b_proj(x)).contiguous()

        k = k * self.scale

        o = self.gated_deltanet.gated_deltanet_recurrence(q, k, v, alpha, beta)

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
