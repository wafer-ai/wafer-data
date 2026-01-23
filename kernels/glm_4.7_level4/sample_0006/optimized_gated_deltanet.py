import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized kernel for outer product computation needed in Gated DeltaNet
# Computes error_outer_k = error * k_t^T efficiently
outer_product_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void outer_product_kernel(
    const float* error,       // (batch, num_heads, head_dim_v)
    const float* k,           // (batch, num_heads, head_dim_qk)
    float* output,            // (batch, num_heads, head_dim_v, head_dim_qk)
    int batch_size,
    int num_heads,
    int head_dim_v,
    int head_dim_qk)
{
    int batch = blockIdx.x;
    int head = blockIdx.y;
    
    if (batch >= batch_size || head >= num_heads) return;
    
    // Linearized thread index
    int idx = threadIdx.x;
    int total_v_threads = blockDim.x;
    
    // Loop over head_dim_v
    for (int v_idx = idx; v_idx < head_dim_v; v_idx += total_v_threads) {
        // Get error value for this head and v_idx
        float e_val = error[batch * num_heads * head_dim_v + head * head_dim_v + v_idx];
        
        // Compute outer product row
        float* out_row = output + batch * num_heads * head_dim_v * head_dim_qk + 
                         head * head_dim_v * head_dim_qk + 
                         v_idx * head_dim_qk;
        
        const float* k_ptr = k + batch * num_heads * head_dim_qk + head * head_dim_qk;
        
        // Compute this row of outer product
        for (int j = 0; j < head_dim_qk; j++) {
            out_row[j] = e_val * k_ptr[j];
        }
    }
}

torch::Tensor outer_product_hip(torch::Tensor error, torch::Tensor k) {
    auto batch_size = error.size(0);
    auto num_heads = error.size(1);
    auto head_dim_v = error.size(2);
    auto head_dim_qk = k.size(2);
    
    auto output = torch::empty({batch_size, num_heads, head_dim_v, head_dim_qk}, error.options());
    
    dim3 grid(batch_size, num_heads);
    int threads = 256;
    
    hipLaunchKernelGGL(
        outer_product_kernel,
        grid, threads, 0, 0,
        error.data_ptr<float>(),
        k.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        head_dim_v,
        head_dim_qk
    );
    
    return output;
}
"""

outer_product_module = load_inline(
    name="outer_product_module",
    cpp_sources=outer_product_cpp_source,
    functions=["outer_product_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized Gated DeltaNet with HIP kernels for expensive outer product operations.
    """
    def __init__(self, hidden_size, num_heads, head_dim_qk, head_dim_v, use_short_conv=True, conv_kernel_size=4):
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

        # Output gate
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)
        self.scale = head_dim_qk ** -0.5
        
        # Load custom kernel for outer product
        self.outer_product_kernel = outer_product_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        k = self.k_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        v = self.v_proj(x)  # (batch, seq, num_heads * head_dim_v)

        # Optional short convolution
        if self.use_short_conv:
            # (batch, seq, dim) -> (batch, dim, seq) for conv1d
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

        # Compute gating values
        alpha = torch.sigmoid(self.a_proj(x))  # (batch, seq, num_heads)
        beta = torch.sigmoid(self.b_proj(x))   # (batch, seq, num_heads)

        # Scale keys to prevent state explosion
        k = k * self.scale

        # Initialize state matrix: (batch, num_heads, head_dim_v, head_dim_qk)
        S = torch.zeros(
            batch_size, self.num_heads, self.head_dim_v, self.head_dim_qk,
            device=device, dtype=dtype
        )

        outputs = []

        # Recurrence loop - only replacing einsum with custom kernel
        for t in range(seq_len):
            # Get current timestep values
            q_t = q[:, t, :, :]   # (batch, num_heads, head_dim_qk)
            k_t = k[:, t, :, :]   # (batch, num_heads, head_dim_qk)
            v_t = v[:, t, :, :]   # (batch, num_heads, head_dim_v)
            alpha_t = alpha[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (batch, num_heads, 1, 1)
            beta_t = beta[:, t, :].unsqueeze(-1).unsqueeze(-1)    # (batch, num_heads, 1, 1)

            # Compute S @ k: (batch, num_heads, head_dim_v, head_dim_qk) @ (batch, num_heads, head_dim_qk, 1)
            #             -> (batch, num_heads, head_dim_v, 1)
            k_t_col = k_t.unsqueeze(-1)  # (batch, num_heads, head_dim_qk, 1)
            S_k = torch.matmul(S, k_t_col).squeeze(-1)  # (batch, num_heads, head_dim_v)

            # Compute error: S @ k - v
            error = S_k - v_t  # (batch, num_heads, head_dim_v)

            # Outer product: error @ k^T -> (batch, num_heads, head_dim_v, head_dim_qk)
            # USE CUSTOM KERNEL HERE
            error_outer_k = self.outer_product_kernel.outer_product_hip(error, k_t)

            # State update: S = alpha * S - beta * error @ k^T
            S = alpha_t * S - beta_t * error_outer_k

            # Output: o = S @ q
            q_t_col = q_t.unsqueeze(-1)  # (batch, num_heads, head_dim_qk, 1)
            o_t = torch.matmul(S, q_t_col).squeeze(-1)  # (batch, num_heads, head_dim_v)

            outputs.append(o_t)

        # Stack outputs: (seq, batch, num_heads, head_dim_v) -> (batch, seq, num_heads, head_dim_v)
        o = torch.stack(outputs, dim=1)  # (batch, seq, num_heads, head_dim_v)

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