import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for Kimi Delta Attention timestep computation
# This kernel fuses multiple operations from the recurrence loop:
# 1. S @ k (matrix-vector multiplication)
# 2. error = S_k - v
# 3. outer product error @ k^T
# 4. diagonal gating diag(a) @ S
# 5. state update S = S_gated - beta * error_outer_k
# 6. output computation o = S_new @ q

kimi_delta_hip_source = """
#include <hip/hip_runtime.h>

// Fused kernel for one timestep of KDA recurrence
// Input shapes:
//   S:    [batch, num_heads, head_dim_v, head_dim_qk] - current state (in-place updated)
//   k_t:  [batch, num_heads, head_dim_qk]
//   v_t:  [batch, num_heads, head_dim_v]
//   a_t:  [batch, num_heads, head_dim_v] - channel-wise gates
//   q_t:  [batch, num_heads, head_dim_qk]
//   beta: [batch, num_heads]
// Output:
//   S is updated in-place to new state
//   o_t:  [batch, num_heads, head_dim_v] - output using NEW state

__global__ void kimi_delta_fused_kernel(
    float* __restrict__ S,
    const float* __restrict__ k_t,
    const float* __restrict__ v_t,
    const float* __restrict__ a_t,
    const float* __restrict__ q_t,
    const float* __restrict__ beta,
    float* __restrict__ o_t,
    int batch_size, int num_heads,
    int head_dim_v, int head_dim_qk
) {
    // Use 3D grid: batch, head, rows of head_dim_v
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int row = blockIdx.z * blockDim.z + threadIdx.z; 
    
    if (batch >= batch_size || head >= num_heads || row >= head_dim_v) {
        return;
    }
    
    // Base indices for this batch, head, and row
    int S_base = (batch * num_heads + head) * head_dim_v * head_dim_qk;
    int k_base = (batch * num_heads + head) * head_dim_qk;
    int v_base = (batch * num_heads + head) * head_dim_v;
    int a_base = (batch * num_heads + head) * head_dim_v;
    int q_base = (batch * num_heads + head) * head_dim_qk;
    int beta_idx = batch * num_heads + head;
    int o_t_base = (batch * num_heads + head) * head_dim_v;
    
    // Get gating parameters for this row
    float a_val = a_t[a_base + row];
    float v_val = v_t[v_base + row];
    float beta_val = beta[beta_idx];
    
    // Phase 1: Compute S_k = S @ k for this row (dot product along k dimension)
    // Use OLD state values before update
    float S_k_val = 0.0f;
    for (int k = 0; k < head_dim_qk; k++) {
        S_k_val += S[S_base + row * head_dim_qk + k] * k_t[k_base + k];
    }
    
    // Error: S @ k - v
    float error = S_k_val - v_val;
    
    // Phase 2: Update each column of S for this row to get S_new
    // And accumulate output: o = S_new @ q
    float o_val = 0.0f;
    
    for (int col = 0; col < head_dim_qk; col++) {
        // Get old S value
        float S_val = S[S_base + row * head_dim_qk + col];
        
        // Apply diagonal gating: diag(a) @ S
        float S_gated = a_val * S_val;
        
        // Outer product contribution: beta * error * k^T
        float outer_prod = beta_val * error * k_t[k_base + col];
        
        // State update: S_new (update in place)
        float S_new = S_gated - outer_prod;
        S[S_base + row * head_dim_qk + col] = S_new;
        
        // Accumulate output: S_new @ q
        o_val += S_new * q_t[q_base + col];
    }
    
    // Write output
    o_t[o_t_base + row] = o_val;
}

torch::Tensor kimi_delta_fused_hip(
    torch::Tensor S, torch::Tensor k_t, torch::Tensor v_t,
    torch::Tensor a_t, torch::Tensor q_t, torch::Tensor beta
) {
    int batch_size = S.size(0);
    int num_heads = S.size(1);
    int head_dim_v = S.size(2);
    int head_dim_qk = S.size(3);
    
    auto o_t = torch::zeros({batch_size, num_heads, head_dim_v}, S.options());
    
    // Use appropriate block size for rows
    int thread_per_block = 256;
    int rows_per_block = thread_per_block;
    dim3 blockDim(thread_per_block);
    dim3 gridDim(batch_size, num_heads, (head_dim_v + rows_per_block - 1) / rows_per_block);
    
    // Use default stream (0)
    kimi_delta_fused_kernel<<<gridDim, blockDim, 0, 0>>>(
        S.data_ptr<float>(), k_t.data_ptr<float>(), v_t.data_ptr<float>(),
        a_t.data_ptr<float>(), q_t.data_ptr<float>(), beta.data_ptr<float>(),
        o_t.data_ptr<float>(),
        batch_size, num_heads, head_dim_v, head_dim_qk
    );
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        AT_ERROR("HIP kernel launch failed: ", hipGetErrorString(err));
    }
    
    hipDeviceSynchronize();
    
    return o_t;
}
"""

kimi_delta_module = load_inline(
    name="kimi_delta",
    cpp_sources=kimi_delta_hip_source,
    functions=["kimi_delta_fused_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized Kimi Delta Attention with custom HIP kernels
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
        self.scale = head_dim_qk ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        # Channel-wise gating
        self.a_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=True)
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
        
        # Custom HIP kernel module
        self.kimi_delta_hip = kimi_delta_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized HIP kernel
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Optional short convolution (keep as PyTorch ops - these are small)
        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Channel-wise gating
        a = torch.sigmoid(self.a_proj(x))
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Delta learning rate
        beta = torch.sigmoid(self.b_proj(x))

        # DPLR low-rank factors (optional)
        if self.use_dplr:
            l = self.l_proj(x).view(batch_size, seq_len, self.num_heads, self.dplr_rank)
            r = self.r_proj(x).view(batch_size, seq_len, self.num_heads, self.dplr_rank)

        # Scale keys
        k = k * self.scale

        # Initialize state
        S = torch.zeros(
            batch_size, self.num_heads, self.head_dim_v, self.head_dim_qk,
            device=device, dtype=dtype
        )

        outputs = []

        # Process each timestep with fused kernel
        for t in range(seq_len):
            q_t = q[:, t, :, :]   # (batch, num_heads, head_dim_qk)
            k_t = k[:, t, :, :]   # (batch, num_heads, head_dim_qk)
            v_t = v[:, t, :, :]   # (batch, num_heads, head_dim_v)
            a_t = a[:, t, :, :]   # (batch, num_heads, head_dim_v)
            beta_t = beta[:, t, :]  # (batch, num_heads)
            
            # Call fused HIP kernel - this replaces multiple PyTorch ops per timestep
            # The kernel updates S in-place and returns o_t
            o_t = self.kimi_delta_hip.kimi_delta_fused_hip(
                S, k_t, v_t, a_t, q_t, beta_t
            )
            
            outputs.append(o_t)

        # Stack outputs
        o = torch.stack(outputs, dim=1)

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