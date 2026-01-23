import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# HIP kernel code for optimized Kimi Delta Attention recurrence
kimi_delta_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <ATen/ATen.h>

#define WARP_SIZE 32

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
    // Grid: (batch_size, num_heads, head_dim_v)
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = blockIdx.z;  // Row index in S matrix
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    
    // Offset calculations for accessing tensors
    int qk_offset = ((batch_idx * seq_len * num_heads + head_idx) * head_dim_qk);
    int v_offset = ((batch_idx * seq_len * num_heads + head_idx) * head_dim_v);
    int out_offset = ((batch_idx * seq_len * num_heads + head_idx) * head_dim_v);
    
    // Shared memory for one row of S matrix (size: head_dim_qk)
    extern __shared__ float shared_mem[];
    float* S_row = shared_mem;
    
    // Initialize S_row to zero
    for (int i = tid; i < head_dim_qk; i += blockDim.x) {
        S_row[i] = 0.0f;
    }
    __syncthreads();
    
    // Process all timesteps in sequence within this persistent kernel
    for (int t = 0; t < seq_len; ++t) {
        // Load k_t for this timestep (coalesced memory access)
        float k_t_val = 0.0f;
        if (lane_id < head_dim_qk) {
            k_t_val = k[qk_offset + t * num_heads * head_dim_qk + lane_id];
        }
        
        // Load v_t[row], a_t[row], beta_t (broadcast values from thread 0)
        float v_t_row = 0.0f, a_t_row = 0.0f, beta_t_value = 0.0f;
        if (tid == 0) {
            v_t_row = v[v_offset + t * num_heads * head_dim_v + row];
            a_t_row = a[v_offset + t * num_heads * head_dim_v + row];
            beta_t_value = beta[batch_idx * seq_len * num_heads + t * num_heads + head_idx];
        }
        // Broadcast to all threads in the block
        v_t_row = __shfl_sync(0xffffffff, v_t_row, 0);
        a_t_row = __shfl_sync(0xffffffff, a_t_row, 0);
        beta_t_value = __shfl_sync(0xffffffff, beta_t_value, 0);
        
        // Compute S_row @ k_t: sum_j S_row[j] * k_t[j]
        float sum = 0.0f;
        if (lane_id < head_dim_qk) {
            sum = S_row[lane_id] * k_t_val;
        }
        
        // Warp reduction using the new HIP shuffle API
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        float S_k_row = sum;  // Only valid in lane 0
        
        // Broadcast S_k_row to all threads in the warp
        S_k_row = __shfl_sync(0xffffffff, S_k_row, 0);
        
        // Error = S_k_row - v_t_row
        float error_row = S_k_row - v_t_row;
        
        // Apply diagonal gating: S_row[j] *= a_t_row
        if (lane_id < head_dim_qk) {
            S_row[lane_id] *= a_t_row;
        }
        
        // Update S: S_row -= beta_t * error_row * k_t^T
        if (lane_id < head_dim_qk) {
            S_row[lane_id] -= beta_t_value * error_row * k_t_val;
        }
        
        // Compute output: o_t[row] = (S @ q_t)[row]
        float q_t_val = 0.0f;
        if (lane_id < head_dim_qk) {
            q_t_val = q[qk_offset + t * num_heads * head_dim_qk + lane_id];
        }
        
        float o_sum = 0.0f;
        if (lane_id < head_dim_qk) {
            o_sum = S_row[lane_id] * q_t_val;
        }
        
        // Warp reduction for output
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            o_sum += __shfl_down_sync(0xffffffff, o_sum, offset);
        }
        
        // Store output (only lane 0 writes)
        if (lane_id == 0) {
            out[out_offset + t * num_heads * head_dim_v + row] = o_sum;
        }
        
        __syncthreads();  // Ensure S_row is updated before next timestep
    }
}

torch::Tensor kimi_delta_attention_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
) {
    // Input shape checks
    TORCH_CHECK(q.dim() == 4, "q must be 4D tensor (batch, seq, heads, dim_qk)");
    TORCH_CHECK(k.dim() == 4, "k must be 4D tensor (batch, seq, heads, dim_qk)");
    TORCH_CHECK(v.dim() == 4, "v must be 4D tensor (batch, seq, heads, dim_v)");
    TORCH_CHECK(a.dim() == 4, "a must be 4D tensor (batch, seq, heads, dim_v)");
    TORCH_CHECK(beta.dim() == 3, "beta must be 3D tensor (batch, seq, heads)");
    
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int head_dim_qk = q.size(3);
    int head_dim_v = v.size(3);
    
    // Output shape: (batch, seq, num_heads, head_dim_v)
    auto out = torch::zeros_like(v);
    
    // Ensure contiguous memory layout
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    a = a.contiguous();
    beta = beta.contiguous();
    
    // Grid: (batch_size, num_heads, head_dim_v)
    dim3 grid(batch_size, num_heads, head_dim_v);
    
    // Block: 128 threads (multiple of warp size for efficiency)
    dim3 block(128);
    
    // Shared memory: head_dim_qk floats per block
    size_t shared_mem_size = head_dim_qk * sizeof(float);
    
    // Get HIP stream from input tensor
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    
    // Launch kernel
    hipLaunchKernelGGL(
        kimi_delta_attention_kernel,
        grid,
        block,
        shared_mem_size,
        stream,
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
    Optimized Kimi Delta Attention with custom HIP kernel for the core recurrence.
    The kernel parallelizes the computation across batch, heads, and rows of the state matrix,
    while using persistent threads to process the sequence dimension efficiently.
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

        # Channel-wise gating (per-channel decay gates)
        self.a_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=True)

        # Delta learning rate (scalar per head)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # DPLR low-rank factors (optional, currently disabled for benchmarking)
        if use_dplr:
            self.l_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)
            self.r_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)

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

        # Output gate with normalization
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        # Scaling factor for keys
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of optimized Kimi Delta Attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        k = self.k_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        v = self.v_proj(x)  # (batch, seq, num_heads * head_dim_v)

        # Optional short convolution with SiLU activation
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

        # Compute channel-wise gating (per-channel decay)
        a = torch.sigmoid(self.a_proj(x))  # (batch, seq, num_heads * head_dim_v)
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Compute delta learning rate (scalar per head)
        beta = torch.sigmoid(self.b_proj(x))  # (batch, seq, num_heads)

        # Scale keys
        k = k * self.scale

        # OPTIMIZED: Use custom HIP kernel for the core recurrence
        # This replaces the inefficient sequential loop with parallel persistent threads
        o = kimi_delta_attention.kimi_delta_attention_hip(q, k, v, a, beta)

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

# Configuration matching Kimi Linear paper settings
batch_size = 4
seq_len = 2048
hidden_size = 2048
num_heads = 16
head_dim_qk = 128  # Key/query dimension per head
head_dim_v = 128   # Value dimension per head


def get_inputs():
    """Generate random input tensors for benchmarking."""
    return [torch.randn(batch_size, seq_len, hidden_size).cuda()]


def get_init_inputs():
    """Return initialization parameters for the model."""
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
