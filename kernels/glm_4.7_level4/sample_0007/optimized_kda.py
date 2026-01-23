import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

dapl_kiminet_hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <hip/hip_runtime_api.h>

// Optimized kernel using warp-level primitives for synchronization
__global__ void kda_state_update_kernel_device(
    float* S,
    const float* k,
    const float* q,
    const float* v,
    const float* a,
    const float* beta,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim_v,
    int head_dim_qk
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * num_heads * head_dim_v * head_dim_qk;
    if (tid >= total_elements) return;
    
    // Decompose tid
    int temp = tid;
    int j = temp % head_dim_qk;
    temp /= head_dim_qk;
    int i = temp % head_dim_v;
    temp /= head_dim_v;
    int h = temp % num_heads;
    int b = temp / num_heads;
    
    int S_base = b * num_heads * head_dim_v * head_dim_qk +
                 h * head_dim_v * head_dim_qk +
                 i * head_dim_qk + j;
    
    float S_ij = S[S_base];
    
    for (int t = 0; t < seq_len; t++) {
        // Read row snapshot of S before update using shared memory
        __shared__ float row_snapshot[256 + 32];  // +32 for padding
        
        int S_row_base = b * num_heads * head_dim_v * head_dim_qk +
                        h * head_dim_v * head_dim_qk +
                        i * head_dim_qk;
        
        // Each thread loads its S[i,j] value to shared memory
        if (j < head_dim_qk && j < 256) {
            row_snapshot[j] = S[S_row_base + j];
        }
        __syncthreads();
        
        // Compute (S @ k)[i] using shared memory (consistent snapshot)
        float S_k = 0.0f;
        int k_idx_base = b * seq_len * num_heads * head_dim_qk +
                        t * num_heads * head_dim_qk +
                        h * head_dim_qk;
        
        for (int kd = 0; kd < head_dim_qk; kd++) {
            S_k += row_snapshot[kd] * k[k_idx_base + kd];
        }
        
        // Read v_t[i]
        int v_idx = b * seq_len * num_heads * head_dim_v +
                   t * num_heads * head_dim_v +
                   h * head_dim_v +
                   i;
        float v_ti = v[v_idx];
        
        // Error
        float error = S_k - v_ti;
        
        // Read beta_t
        int beta_idx = b * seq_len * num_heads + t * num_heads + h;
        float beta_t = beta[beta_idx];
        float k_tj = k[k_idx_base + j];
        
        // Delta
        float delta = -beta_t * error * k_tj;
        
        // Read a_t[i]
        int a_idx = b * seq_len * num_heads * head_dim_v +
                    t * num_heads * head_dim_v +
                    h * head_dim_v +
                    i;
        float a_ti = a[a_idx];
        
        // Update S (use OLD S_ij from register, which is pre-update value)
        S_ij = a_ti * S_ij + delta;
        S[S_base] = S_ij;
        
        // Output accumulation
        int q_idx_base = b * seq_len * num_heads * head_dim_qk +
                        t * num_heads * head_dim_qk +
                        h * head_dim_qk;
        float q_tj = q[q_idx_base + j];
        
        int output_idx = b * seq_len * num_heads * head_dim_v +
                        t * num_heads * head_dim_v +
                        h * head_dim_v +
                        i;
        atomicAdd(&output[output_idx], S_ij * q_tj);
    }
}

torch::Tensor kda_state_update_kernel(
    torch::Tensor S,
    torch::Tensor k,
    torch::Tensor q,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta,
    torch::Tensor output,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len,
    int64_t head_dim_v,
    int64_t head_dim_qk
) {
    const int block_size = 256;
    int64_t total_elements = batch_size * num_heads * head_dim_v * head_dim_qk;
    int64_t num_blocks = (total_elements + block_size - 1) / block_size;
    int shared_mem_size = (256 + 32) * sizeof(float);
    
    kda_state_update_kernel_device<<<num_blocks, block_size, shared_mem_size>>>(
        S.data_ptr<float>(),
        k.data_ptr<float>(),
        q.data_ptr<float>(),
        v.data_ptr<float>(),
        a.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim_v,
        head_dim_qk
    );
    
    return output;
}
"""

dapl_kiminet = load_inline(
    name="dapl_kiminet",
    cpp_sources=dapl_kiminet_hip_source,
    functions=["kda_state_update_kernel"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized Kimi Delta Attention with fused HIP kernel
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

        # Scaling factor for keys
        self.scale = head_dim_qk ** -0.5

        # Custom HIP kernel module
        self.kda_kernel = dapl_kiminet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized HIP kernel.
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

        # Reshape for multi-head attention and get tensors
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Channel-wise gating
        a = torch.sigmoid(self.a_proj(x))
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Delta learning rate
        beta = torch.sigmoid(self.b_proj(x))

        # Scale keys
        k = k * self.scale

        # Initialize state matrix
        S = torch.zeros(
            batch_size, self.num_heads, self.head_dim_v, self.head_dim_qk,
            device=device, dtype=dtype
        )

        # Initialize output tensor (will accumulate results)
        output = torch.zeros(
            batch_size, seq_len, self.num_heads, self.head_dim_v,
            device=device, dtype=dtype
        )

        # Get contiguous versions for kernel
        S = S.contiguous()
        k = k.contiguous()
        q = q.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        beta = beta.contiguous()
        output = output.contiguous()

        # Launch fused kernel
        self.kda_kernel.kda_state_update_kernel(
            S, k, q, v, a, beta, output,
            batch_size, self.num_heads, seq_len,
            self.head_dim_v, self.head_dim_qk
        )

        # Apply output normalization per head
        o = self.o_norm(output)

        # Apply output gate
        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        # Reshape and project output
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o