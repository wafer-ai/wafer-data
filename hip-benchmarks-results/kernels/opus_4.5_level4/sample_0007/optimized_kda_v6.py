import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized KDA kernel with reduced synchronizations
kda_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define MAX_DIM 128

// Highly optimized kernel with minimal synchronizations
__global__ __launch_bounds__(128, 8)
void kda_recurrence_kernel_v6(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ a,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim_qk,
    int head_dim_v
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    int tid = threadIdx.x;
    
    // Shared memory - double buffered for overlap
    extern __shared__ float shared_mem[];
    float* k_shared = shared_mem;                 // 128
    float* q_shared = shared_mem + 128;           // 128
    float* v_shared = shared_mem + 256;           // 128
    float* a_shared = shared_mem + 384;           // 128
    float* Sk_shared = shared_mem + 512;          // 128
    
    // Each thread handles one row
    float S_row[MAX_DIM];
    
    #pragma unroll
    for (int c = 0; c < 128; c++) {
        S_row[c] = 0.0f;
    }
    
    // Compute base offsets
    const int stride_qk = num_heads * 128;
    const int stride_v = num_heads * 128;
    const int stride_beta = num_heads;
    
    const int base_qk = batch_idx * seq_len * stride_qk + head_idx * 128;
    const int base_v = batch_idx * seq_len * stride_v + head_idx * 128;
    const int base_beta = batch_idx * seq_len * stride_beta + head_idx;
    
    // Process sequence
    for (int t = 0; t < seq_len; t++) {
        const int off_qk = base_qk + t * stride_qk;
        const int off_v = base_v + t * stride_v;
        const int off_beta = base_beta + t * stride_beta;
        
        // Load all data - coalesced access
        k_shared[tid] = k[off_qk + tid];
        q_shared[tid] = q[off_qk + tid];
        v_shared[tid] = v[off_v + tid];
        a_shared[tid] = a[off_v + tid];
        
        __syncthreads();
        
        const float beta_t = beta[off_beta];
        
        // Compute S @ k for row tid
        float dot = 0.0f;
        #pragma unroll 16
        for (int c = 0; c < 128; c++) {
            dot += S_row[c] * k_shared[c];
        }
        Sk_shared[tid] = dot;
        
        __syncthreads();
        
        // Update state
        const float a_r = a_shared[tid];
        const float error = Sk_shared[tid] - v_shared[tid];
        const float coeff = beta_t * error;
        
        // Fused state update and output computation
        float out_dot = 0.0f;
        #pragma unroll 16
        for (int c = 0; c < 128; c++) {
            S_row[c] = a_r * S_row[c] - coeff * k_shared[c];
            out_dot += S_row[c] * q_shared[c];
        }
        
        // Write output directly
        output[off_v + tid] = out_dot;
        
        __syncthreads();
    }
}

torch::Tensor kda_recurrence_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
) {
    auto batch_size = q.size(0);
    auto seq_len = q.size(1);
    auto num_heads = q.size(2);
    auto head_dim_qk = q.size(3);
    auto head_dim_v = v.size(3);
    
    auto output = torch::zeros_like(v);
    
    dim3 grid(batch_size, num_heads);
    int block_size = 128;
    
    // 5 arrays of 128 floats each
    int shared_mem_size = 128 * 5 * sizeof(float);
    
    kda_recurrence_kernel_v6<<<grid, block_size, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        a.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim_qk,
        head_dim_v
    );
    
    return output;
}
"""

kda_cpp_source = """
torch::Tensor kda_recurrence_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
);
"""

kda_module = load_inline(
    name="kda_module_v6",
    cpp_sources=kda_cpp_source,
    cuda_sources=kda_kernel_source,
    functions=["kda_recurrence_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized Kimi Delta Attention with fused HIP kernel for recurrence.
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

        # Delta learning rate
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # DPLR low-rank factors (optional)
        if use_dplr:
            self.l_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)
            self.r_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        # Optional short convolution
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

        # Output gate with normalization
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        # Scaling factor
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Optional short convolution
        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).contiguous()
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()

        # Compute channel-wise gating
        a = torch.sigmoid(self.a_proj(x))
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()

        # Delta learning rate
        beta = torch.sigmoid(self.b_proj(x)).contiguous()

        # Scale keys
        k = k * self.scale

        # Use fused HIP kernel for recurrence
        o = kda_module.kda_recurrence_hip(q, k, v, a, beta)

        # Apply output normalization per head
        o = self.o_norm(o)

        # Apply output gate
        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        # Reshape and project output
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o


# Configuration
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
