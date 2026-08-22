import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# C++ header for function declarations
deltanet_cpp_source = """
#include <torch/extension.h>

torch::Tensor gated_deltanet_recurrence(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta
);
"""

# Optimized HIP kernel with separated steps and less synchronization
deltanet_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define HEAD_DIM 128

// Optimized kernel with fixed head dimensions for better register usage
__global__ void gated_deltanet_recurrence_kernel(
    const float* __restrict__ q,      // [batch, seq_len, num_heads, head_dim_qk]
    const float* __restrict__ k,      // [batch, seq_len, num_heads, head_dim_qk]
    const float* __restrict__ v,      // [batch, seq_len, num_heads, head_dim_v]
    const float* __restrict__ alpha,  // [batch, seq_len, num_heads]
    const float* __restrict__ beta,   // [batch, seq_len, num_heads]
    float* __restrict__ output,       // [batch, seq_len, num_heads, head_dim_v]
    float* __restrict__ state,        // [batch, num_heads, head_dim_v, head_dim_qk]
    int batch_size,
    int seq_len,
    int num_heads
) {
    // Each block handles one (batch, head) pair
    int batch_idx = blockIdx.x / num_heads;
    int head_idx = blockIdx.x % num_heads;
    
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // State base pointer for this (batch, head)
    int state_base = (batch_idx * num_heads + head_idx) * HEAD_DIM * HEAD_DIM;
    float* S = state + state_base;
    
    const int state_size = HEAD_DIM * HEAD_DIM;
    
    // Shared memory for k, v, q vectors and S_k intermediate
    __shared__ float k_shared[HEAD_DIM];
    __shared__ float v_shared[HEAD_DIM];
    __shared__ float q_shared[HEAD_DIM];
    __shared__ float S_k[HEAD_DIM];
    
    // Initialize state to zero
    for (int i = tid; i < state_size; i += num_threads) {
        S[i] = 0.0f;
    }
    __syncthreads();
    
    // Loop over sequence
    for (int t = 0; t < seq_len; t++) {
        // Load k, v, q for this timestep
        int base_qk = ((batch_idx * seq_len + t) * num_heads + head_idx) * HEAD_DIM;
        int base_v = ((batch_idx * seq_len + t) * num_heads + head_idx) * HEAD_DIM;
        int base_gate = (batch_idx * seq_len + t) * num_heads + head_idx;
        
        // Coalesced loads into shared memory
        if (tid < HEAD_DIM) {
            k_shared[tid] = k[base_qk + tid];
            q_shared[tid] = q[base_qk + tid];
            v_shared[tid] = v[base_v + tid];
        }
        __syncthreads();
        
        float alpha_t = alpha[base_gate];
        float beta_t = beta[base_gate];
        
        // Step 1: Compute S @ k -> S_k [head_dim_v]
        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            float sum = 0.0f;
            int row_base = i * HEAD_DIM;
            #pragma unroll 8
            for (int j = 0; j < HEAD_DIM; j++) {
                sum += S[row_base + j] * k_shared[j];
            }
            S_k[i] = sum;
        }
        __syncthreads();
        
        // Step 2: Compute error = S_k - v and update S
        for (int idx = tid; idx < state_size; idx += num_threads) {
            int i = idx / HEAD_DIM;  // v dimension
            int j = idx % HEAD_DIM;  // k dimension
            
            float error_i = S_k[i] - v_shared[i];
            float s_val = S[idx];
            S[idx] = alpha_t * s_val - beta_t * error_i * k_shared[j];
        }
        __syncthreads();
        
        // Step 3: Compute output: o = S @ q -> [head_dim_v]
        int out_base = ((batch_idx * seq_len + t) * num_heads + head_idx) * HEAD_DIM;
        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            float sum = 0.0f;
            int row_base = i * HEAD_DIM;
            #pragma unroll 8
            for (int j = 0; j < HEAD_DIM; j++) {
                sum += S[row_base + j] * q_shared[j];
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
    int head_dim_qk = q.size(3);
    int head_dim_v = v.size(3);
    
    auto output = torch::zeros({batch_size, seq_len, num_heads, head_dim_v}, 
                               q.options());
    
    // Allocate state matrix in global memory
    auto state = torch::zeros({batch_size, num_heads, head_dim_v, head_dim_qk},
                              q.options());
    
    // Ensure contiguous
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    alpha = alpha.contiguous();
    beta = beta.contiguous();
    
    int num_blocks = batch_size * num_heads;
    int threads_per_block = 1024;
    
    hipLaunchKernelGGL(gated_deltanet_recurrence_kernel, 
                       dim3(num_blocks), dim3(threads_per_block), 0, 0,
                       q.data_ptr<float>(),
                       k.data_ptr<float>(),
                       v.data_ptr<float>(),
                       alpha.data_ptr<float>(),
                       beta.data_ptr<float>(),
                       output.data_ptr<float>(),
                       state.data_ptr<float>(),
                       batch_size, seq_len, num_heads);
    
    return output;
}
"""

deltanet_module = load_inline(
    name="gated_deltanet_hip_v6",
    cpp_sources=deltanet_cpp_source,
    cuda_sources=deltanet_kernel_source,
    functions=["gated_deltanet_recurrence"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized Gated DeltaNet with fused HIP kernel for recurrence.
    """

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

        # Compute gating values
        alpha = torch.sigmoid(self.a_proj(x))  # (batch, seq, num_heads)
        beta = torch.sigmoid(self.b_proj(x))   # (batch, seq, num_heads)

        # Scale keys
        k = k * self.scale

        # Use fused HIP kernel for recurrence
        o = deltanet_module.gated_deltanet_recurrence(q, k, v, alpha, beta)

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


def custom_kernel(inputs):
    x = inputs[0]
    batch_size = x.size(0)
    hidden_size = x.size(2)
    num_heads = 16
    head_dim_qk = 128
    head_dim_v = 128
    
    model = ModelNew(hidden_size, num_heads, head_dim_qk, head_dim_v).cuda()
    model.eval()
    
    with torch.no_grad():
        return model(x)
