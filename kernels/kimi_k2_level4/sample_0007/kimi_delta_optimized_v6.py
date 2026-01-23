import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fixed version with 64-bit masks for ROCm compatibility
kimi_delta_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <ATen/ATen.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128

__global__ void kimi_delta_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ a,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim_qk,
    int head_dim_v
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = blockIdx.z;
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || row >= head_dim_v) {
        return;
    }
    
    extern __shared__ float shared_mem[];
    float* S_row = shared_mem;
    
    // Initialize S_row
    for (int i = tid; i < head_dim_qk; i += BLOCK_SIZE) {
        S_row[i] = 0.0f;
    }
    __syncthreads();
    
    // Use 64-bit mask for ROCm compatibility
    const unsigned long long FULL_MASK = 0xffffffffULL;
    
    for (int t = 0; t < seq_len; ++t) {
        int base_idx = ((batch_idx * seq_len + t) * num_heads + head_idx);
        int qk_base = base_idx * head_dim_qk;
        int v_base = base_idx * head_dim_v;
        int beta_idx = batch_idx * seq_len * num_heads + t * num_heads + head_idx;
        
        // Load k_t and q_t
        float k_val = (lane_id < head_dim_qk) ? k[qk_base + lane_id] : 0.0f;
        float q_val = (lane_id < head_dim_qk) ? q[qk_base + lane_id] : 0.0f;
        
        // Load broadcast values
        float v_val = 0.0f, a_val = 0.0f, beta_val = 0.0f;
        if (tid == 0) {
            v_val = v[v_base + row];
            a_val = a[v_base + row];
            beta_val = beta[beta_idx];
        }
        
        // Broadcast using 64-bit mask
        v_val = __shfl_sync(FULL_MASK, v_val, 0);
        a_val = __shfl_sync(FULL_MASK, a_val, 0);
        beta_val = __shfl_sync(FULL_MASK, beta_val, 0);
        
        // Compute S_row @ k_t
        float sum = (lane_id < head_dim_qk) ? S_row[lane_id] * k_val : 0.0f;
        
        // Warp reduction with 64-bit mask
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
        float S_k = __shfl_sync(FULL_MASK, sum, 0);
        
        // Error
        float error = S_k - v_val;
        
        // Update S
        if (lane_id < head_dim_qk) {
            S_row[lane_id] = S_row[lane_id] * a_val - beta_val * error * k_val;
        }
        
        // Compute output: o[row] = (S @ q_t)[row]
        float out_val = (lane_id < head_dim_qk) ? S_row[lane_id] * q_val : 0.0f;
        
        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            out_val += __shfl_down_sync(FULL_MASK, out_val, offset);
        }
        
        if (lane_id == 0) {
            out[v_base + row] = out_val;
        }
        
        __syncthreads();
    }
}

torch::Tensor kimi_delta_attention_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && a.dim() == 4 && beta.dim() == 3, 
                "Dimension mismatch");
    
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int head_dim_qk = q.size(3);
    int head_dim_v = v.size(3);
    
    auto out = torch::zeros_like(v);
    
    dim3 grid(batch_size, num_heads, head_dim_v);
    dim3 block(BLOCK_SIZE);
    size_t shared_mem_size = head_dim_qk * sizeof(float);
    
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    a = a.contiguous();
    beta = beta.contiguous();
    
    hipLaunchKernelGGL(
        kimi_delta_attention_kernel,
        grid,
        block,
        shared_mem_size,
        0,
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

# Load the custom HIP kernel
kimi_delta_attention = load_inline(
    name="kimi_delta_attention",
    cpp_sources=kimi_delta_attention_cpp_source,
    functions=["kimi_delta_attention_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized Kimi Delta Attention with custom HIP kernel.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        head_dim_qk: int = 128,
        head_dim_v: int = 128,
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

        # Channel-wise gating (per-channel decay gates)
        self.a_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=True)

        # Delta learning rate (scalar per head)
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

        # Output gate with normalization
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        # Scaling factor for keys
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of optimized Kimi Delta Attention."""
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
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Compute channel-wise gating
        a = torch.sigmoid(self.a_proj(x))
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Compute delta learning rate
        beta = torch.sigmoid(self.b_proj(x))

        # Scale keys
        k = k * self.scale

        # Use custom HIP kernel for the core recurrence
        o = kimi_delta_attention.kimi_delta_attention_hip(q, k, v, a, beta)

        # Apply output normalization
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
    """Generate random input tensors for benchmarking."""
    return [torch.randn(batch_size, seq_len, hidden_size).cuda()]


def get_init_inputs():
    """Return initialization parameters for the model."""
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
