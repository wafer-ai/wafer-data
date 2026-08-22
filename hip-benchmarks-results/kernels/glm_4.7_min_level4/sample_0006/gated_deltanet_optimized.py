import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gated_delta_cpp_source = """
#include <hip/hip_runtime.h>

// Optimized kernel: use shared memory to reduce global memory access
__global__ void gated_delta_kernel(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    const float* alpha_ptr,
    const float* beta_ptr,
    float* o_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim_qk, int head_dim_v) {

    __shared__ float shared_k[128];  // k vector for this timestep
    __shared__ float shared_q[128];  // q vector for this timestep
    __shared__ float alpha_t_shared;
    __shared__ float beta_t_shared;

    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    int row_idx = threadIdx.x;  // which row of state/output this thread handles
    if (row_idx >= head_dim_v) return;

    int q_stride = num_heads * head_dim_qk;
    int k_stride = num_heads * head_dim_qk;
    int v_stride = num_heads * head_dim_v;
    int alpha_stride = num_heads;
    int beta_stride = num_heads;
    int o_stride = num_heads * head_dim_v;

    // State row S[row_idx, :] initialized to zero
    float S_row[128];
    for (int j = 0; j < 128; j++) {
        S_row[j] = 0.0f;
    }

    // Loop over timesteps
    for (int t = 0; t < seq_len; t++) {
        // Load alpha_t, beta_t via thread 0
        if (row_idx == 0) {
            alpha_t_shared = alpha_ptr[batch_idx * seq_len * alpha_stride + t * alpha_stride + head_idx];
            beta_t_shared = beta_ptr[batch_idx * seq_len * beta_stride + t * beta_stride + head_idx];
        }
        __syncthreads();

        // Load k_t vector into shared memory (collaboratively)
        int k_offset = batch_idx * seq_len * k_stride + t * k_stride + head_idx * head_dim_qk;
        for (int j = row_idx; j < 128; j += head_dim_v) {
            shared_k[j] = k_ptr[k_offset + j];
        }
        __syncthreads();

        // Get v_t for this row
        float v_t = v_ptr[batch_idx * seq_len * v_stride + t * v_stride + head_idx * head_dim_v + row_idx];

        // Compute S_k for this row using shared_k
        float S_k = 0.0f;
        for (int j = 0; j < 128; j++) {
            S_k += S_row[j] * shared_k[j];
        }

        float error = S_k - v_t;

        // Update S_row using shared_k
        float alpha_t = alpha_t_shared;
        float beta_t = beta_t_shared;
        for (int j = 0; j < 128; j++) {
            S_row[j] = alpha_t * S_row[j] - beta_t * error * shared_k[j];
        }

        // Load q_t vector into shared memory (collaboratively)
        int q_offset = batch_idx * seq_len * q_stride + t * q_stride + head_idx * head_dim_qk;
        for (int j = row_idx; j < 128; j += head_dim_v) {
            shared_q[j] = q_ptr[q_offset + j];
        }
        __syncthreads();

        // Compute output using shared_q
        float o_t = 0.0f;
        for (int j = 0; j < 128; j++) {
            o_t += S_row[j] * shared_q[j];
        }

        // Store output
        int o_offset = batch_idx * seq_len * o_stride + t * o_stride + head_idx * head_dim_v + row_idx;
        o_ptr[o_offset] = o_t;
    }
}

torch::Tensor gated_delta_hip(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor alpha, torch::Tensor beta,
    int num_heads, int head_dim_qk, int head_dim_v) {
    
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    auto o = torch::zeros_like(v);
    
    // Use 128 threads per block (one per head_dim_v)
    dim3 grid(batch_size, num_heads);
    dim3 block(128);

    gated_delta_kernel<<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        alpha.data_ptr<float>(),
        beta.data_ptr<float>(),
        o.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim_qk, head_dim_v
    );

    return o;
}
"""

gated_delta_lib = load_inline(
    name="gated_delta",
    cpp_sources=gated_delta_cpp_source,
    functions=["gated_delta_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """Optimized Gated DeltaNet with fused HIP kernel for state recurrence."""

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

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        # Gating projections
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        # Optional short convolution for local context
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

        # Output gate with RMSNorm + SiLU
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        # Scaling factor for keys
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused HIP kernel for state recurrence.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        k = self.k_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        v = self.v_proj(x)  # (batch, seq, num_heads * head_dim_v)

        # Optional short convolution
        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # Reshape for multi-head attention (keep contiguous for kernel)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Compute gating values
        alpha = torch.sigmoid(self.a_proj(x))  # (batch, seq, num_heads)
        beta = torch.sigmoid(self.b_proj(x))   # (batch, seq, num_heads)

        # Scale keys
        k = k * self.scale

        # Fused HIP kernel for state recurrence and output computation
        o = gated_delta_lib.gated_delta_hip(
            q.contiguous(), k.contiguous(), v.contiguous(),
            alpha.contiguous(), beta.contiguous(),
            self.num_heads, self.head_dim_qk, self.head_dim_v
        )

        # Apply output normalization per head
        o = self.o_norm(o)

        # Apply output gate
        g = torch.sigmoid(self.g_proj(x))  # (batch, seq, num_heads * head_dim_v)
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        # Reshape and project output
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o