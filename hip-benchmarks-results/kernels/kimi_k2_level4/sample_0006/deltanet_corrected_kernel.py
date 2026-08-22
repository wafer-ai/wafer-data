import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Corrected HIP kernel for Gated DeltaNet - fixed state update and output timing
gated_delta_kernel_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void gated_delta_timestep_kernel(
    const float* __restrict__ S_old,  // Old state matrix
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ S_new,  // New state matrix
    float* __restrict__ o,
    int batch_size,
    int num_heads,
    int d_v,
    int d_k
) {
    // Get batch and head indices
    int b = blockIdx.x / num_heads;
    int h = blockIdx.x % num_heads;
    int tid_s = threadIdx.x;  // Thread along d_v dimension
    int tid_k = threadIdx.y;  // Thread along d_k dimension
    
    // Bounds check
    if (b >= batch_size || h >= num_heads || tid_k >= d_k) return;
    
    // Calculate base pointers for this batch and head
    const float* k_ptr = k + (b * num_heads + h) * d_k;
    const float* q_ptr = q + (b * num_heads + h) * d_k;
    const float* v_ptr = v + (b * num_heads + h) * d_v;
    const float alpha_val = alpha[b * num_heads + h];
    const float beta_val = beta[b * num_heads + h];
    
    // Shared memory to store S @ k for all d_v elements
    extern __shared__ float shared_mem[];
    float* S_k_sum = shared_mem;  // length = blockDim.x (min(d_v, 128))
    
    // Initialize shared memory
    if (tid_s < blockDim.x) {
        S_k_sum[tid_s] = 0.0f;
    }
    __syncthreads();
    
    // Step 1: Compute partial S @ k across threads in block
    float partial_sum = 0.0f;
    for (int dv = tid_s; dv < d_v; dv += blockDim.x) {
        const float* S_row_ptr = S_old + ((b * num_heads + h) * d_v + dv) * d_k;
        float sum = 0.0f;
        
        // Each thread processes multiple d_k elements
        int dk_per_thread = (d_k + blockDim.y - 1) / blockDim.y;
        int dk_start = tid_k * dk_per_thread;
        int dk_end = min(dk_start + dk_per_thread, d_k);
        
        for (int dk = dk_start; dk < dk_end; ++dk) {
            sum += S_row_ptr[dk] * k_ptr[dk];
        }
        
        // Atomic add to shared memory
        atomicAdd(&S_k_sum[dv], sum);
    }
    __syncthreads();
    
    // Step 2: Compute error and update state
    if (tid_k == 0 && tid_s < d_v) {
        float total_S_k = S_k_sum[tid_s];
        float v_val = v_ptr[tid_s];
        float error_val = total_S_k - v_val;
        
        // Update state: S_new = alpha * S_old - beta * (error @ k^T)
        for (int dk = 0; dk < d_k; ++dk) {
            const float* S_row_ptr = S_old + ((b * num_heads + h) * d_v + tid_s) * d_k;
            float* S_new_row_ptr = S_new + ((b * num_heads + h) * d_v + tid_s) * d_k;
            
            float S_val = S_row_ptr[dk];
            float k_val = k_ptr[dk];
            float outer = error_val * k_val;
            S_new_row_ptr[dk] = alpha_val * S_val - beta_val * outer;
        }
    }
    
    // Step 3: Compute output with NEW state (not old state)
    __syncthreads();
    
    if (tid_k == 0 && tid_s < d_v) {
        float o_sum = 0.0f;
        const float* S_new_row_ptr = S_new + ((b * num_heads + h) * d_v + tid_s) * d_k;
        
        for (int dk = 0; dk < d_k; ++dk) {
            o_sum += S_new_row_ptr[dk] * q_ptr[dk];
        }
        
        // Store output
        o[(b * num_heads + h) * d_v + tid_s] = o_sum;
    }
}

torch::Tensor gated_delta_timestep_hip(
    torch::Tensor S,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta
) {
    // Get dimensions
    int batch_size = S.size(0);
    int num_heads = S.size(1);
    int d_v = S.size(2);
    int d_k = S.size(3);
    
    // Ensure tensors are contiguous
    S = S.contiguous();
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    alpha = alpha.contiguous();
    beta = beta.contiguous();
    
    // Allocate outputs
    auto S_new = torch::zeros_like(S);
    auto o = torch::zeros({batch_size, num_heads, d_v}, 
                         torch::TensorOptions().dtype(S.dtype()).device(S.device()));
    
    // Configure kernel launch - 2D threads: d_v along x, some threads along y for parallel d_k
    dim3 blocks(batch_size * num_heads);
    int threads_x = std::min(d_v, 128);  // Along d_v dimension
    int threads_y = 4;  // Parallelize across d_k dimension
    dim3 threads(threads_x, threads_y);
    
    // Shared memory for S @ k results
    size_t shared_mem_size = d_v * sizeof(float);
    
    // Launch kernel
    hipLaunchKernelGGL(
        gated_delta_timestep_kernel,
        blocks,
        threads,
        shared_mem_size,
        0,
        S.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        alpha.data_ptr<float>(),
        beta.data_ptr<float>(),
        S_new.data_ptr<float>(),
        o.data_ptr<float>(),
        batch_size,
        num_heads,
        d_v,
        d_k
    );
    
    hipDeviceSynchronize();
    
    // Copy new state back to S for next timestep
    S.copy_(S_new);
    
    return o;
}
"""

# Compile the kernel
gated_delta_kernel = load_inline(
    name="gated_delta_kernel",
    cpp_sources=gated_delta_kernel_cpp,
    functions=["gated_delta_timestep_hip"],
    verbose=True,
    extra_cflags=["-O3"],
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
        self.gated_delta_kernel = gated_delta_kernel

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

        # Output gate with RMSNorm + SiLU
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        # Scaling factor
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

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

        # Initialize state
        S = torch.zeros(
            batch_size, self.num_heads, self.head_dim_v, self.head_dim_qk,
            device=device, dtype=dtype
        )

        outputs = []

        # Process each timestep with fused kernel
        for t in range(seq_len):
            # Get current timestep values
            q_t = q[:, t, :, :]   # (batch, num_heads, d_k)
            k_t = k[:, t, :, :]   # (batch, num_heads, d_k)
            v_t = v[:, t, :, :]   # (batch, num_heads, d_v)
            alpha_t = alpha[:, t, :]   # (batch, num_heads)
            beta_t = beta[:, t, :]     # (batch, num_heads)

            # Call fused HIP kernel
            o_t = self.gated_delta_kernel.gated_delta_timestep_hip(
                S, q_t, k_t, v_t, alpha_t, beta_t
            )
            
            # o_t is (batch_size, num_heads, d_v)
            outputs.append(o_t)

        # Stack outputs
        o = torch.stack(outputs, dim=1)  # (batch, seq, num_heads, d_v)

        # Apply output normalization per head
        o = self.o_norm(o)

        # Apply output gate
        g = torch.sigmoid(self.g_proj(x))  # (batch, seq, num_heads * d_v)
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        # Reshape and project output
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o


# Configuration (kept the same for fair comparison)
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