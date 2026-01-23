import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused KDA recurrence kernel
kda_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused KDA state update kernel
// Each thread block handles one (batch, head) pair
// Processes the entire sequence, keeping state in registers/shared memory
__global__ void kda_recurrence_kernel(
    const float* __restrict__ q,      // (batch, seq, heads, head_dim_qk)
    const float* __restrict__ k,      // (batch, seq, heads, head_dim_qk)
    const float* __restrict__ v,      // (batch, seq, heads, head_dim_v)
    const float* __restrict__ a,      // (batch, seq, heads, head_dim_v) - channel-wise gates
    const float* __restrict__ beta,   // (batch, seq, heads)
    float* __restrict__ output,       // (batch, seq, heads, head_dim_v)
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim_qk,
    int head_dim_v
) {
    // Each block handles one (batch, head) pair
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    // Thread handles one or more rows of the state matrix S[head_dim_v, head_dim_qk]
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Shared memory for k_t and intermediate results
    extern __shared__ float shared_mem[];
    float* k_shared = shared_mem;                          // head_dim_qk
    float* q_shared = shared_mem + head_dim_qk;            // head_dim_qk
    float* v_shared = shared_mem + 2 * head_dim_qk;        // head_dim_v
    float* a_shared = shared_mem + 2 * head_dim_qk + head_dim_v;  // head_dim_v
    float* Sk_shared = shared_mem + 2 * head_dim_qk + 2 * head_dim_v;  // head_dim_v
    
    // Each thread maintains a portion of the state matrix
    // State S is (head_dim_v, head_dim_qk)
    // Thread tid handles rows [tid, tid + num_threads, tid + 2*num_threads, ...]
    
    // Local state for this thread's rows
    const int max_rows_per_thread = 16;  // Assume head_dim_v <= num_threads * max_rows_per_thread
    const int max_cols = 128;
    float S_local[max_rows_per_thread][max_cols];
    
    // Initialize state to zero
    for (int r = 0; r < max_rows_per_thread; r++) {
        for (int c = 0; c < head_dim_qk; c++) {
            S_local[r][c] = 0.0f;
        }
    }
    
    // Base offsets
    int qk_base = batch_idx * seq_len * num_heads * head_dim_qk + head_idx * head_dim_qk;
    int v_base = batch_idx * seq_len * num_heads * head_dim_v + head_idx * head_dim_v;
    int beta_base = batch_idx * seq_len * num_heads + head_idx;
    
    // Process each timestep
    for (int t = 0; t < seq_len; t++) {
        // Load k_t, q_t, v_t, a_t into shared memory
        int qk_offset = qk_base + t * num_heads * head_dim_qk;
        int v_offset = v_base + t * num_heads * head_dim_v;
        
        // Collaborative load
        for (int i = tid; i < head_dim_qk; i += num_threads) {
            k_shared[i] = k[qk_offset + i];
            q_shared[i] = q[qk_offset + i];
        }
        for (int i = tid; i < head_dim_v; i += num_threads) {
            v_shared[i] = v[v_offset + i];
            a_shared[i] = a[v_offset + i];
        }
        
        __syncthreads();
        
        float beta_t = beta[beta_base + t * num_heads];
        
        // Compute S @ k for each row this thread handles
        // And store in Sk_shared
        for (int r = tid; r < head_dim_v; r += num_threads) {
            int local_r = (r - tid) / num_threads;
            if (local_r < max_rows_per_thread) {
                float dot = 0.0f;
                for (int c = 0; c < head_dim_qk; c++) {
                    dot += S_local[local_r][c] * k_shared[c];
                }
                Sk_shared[r] = dot;
            }
        }
        
        __syncthreads();
        
        // Update state: S = a_t * S - beta_t * (S@k - v) @ k^T
        // For each row r: S[r,:] = a[r] * S[r,:] - beta_t * (Sk[r] - v[r]) * k[:]
        for (int r = tid; r < head_dim_v; r += num_threads) {
            int local_r = (r - tid) / num_threads;
            if (local_r < max_rows_per_thread) {
                float a_r = a_shared[r];
                float error = Sk_shared[r] - v_shared[r];
                float coeff = beta_t * error;
                
                for (int c = 0; c < head_dim_qk; c++) {
                    S_local[local_r][c] = a_r * S_local[local_r][c] - coeff * k_shared[c];
                }
            }
        }
        
        __syncthreads();
        
        // Compute output: o = S @ q for each row
        for (int r = tid; r < head_dim_v; r += num_threads) {
            int local_r = (r - tid) / num_threads;
            if (local_r < max_rows_per_thread) {
                float dot = 0.0f;
                for (int c = 0; c < head_dim_qk; c++) {
                    dot += S_local[local_r][c] * q_shared[c];
                }
                output[v_offset + r] = dot;
            }
        }
        
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
    int block_size = 128;  // Threads per block
    
    // Shared memory: k(head_dim_qk) + q(head_dim_qk) + v(head_dim_v) + a(head_dim_v) + Sk(head_dim_v)
    int shared_mem_size = (2 * head_dim_qk + 3 * head_dim_v) * sizeof(float);
    
    kda_recurrence_kernel<<<grid, block_size, shared_mem_size>>>(
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
    name="kda_module",
    cpp_sources=kda_cpp_source,
    cuda_sources=kda_kernel_source,
    functions=["kda_recurrence_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
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
