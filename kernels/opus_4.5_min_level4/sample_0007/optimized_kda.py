import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused HIP kernel for Kimi Delta Attention recurrence
kda_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused kernel for the KDA recurrence
// Processes multiple batch*head combinations in parallel
// Each block handles one (batch, head) pair, processing all timesteps sequentially
// but with fused operations for each timestep

__global__ void kda_recurrence_kernel(
    const float* __restrict__ q,      // (batch, seq, heads, dk)
    const float* __restrict__ k,      // (batch, seq, heads, dk)
    const float* __restrict__ v,      // (batch, seq, heads, dv)
    const float* __restrict__ a,      // (batch, seq, heads, dv) - channel gates
    const float* __restrict__ beta,   // (batch, seq, heads)
    float* __restrict__ output,       // (batch, seq, heads, dv)
    float* __restrict__ S,            // (batch, heads, dv, dk) - state buffer
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim_qk,
    int head_dim_v
) {
    // Each block handles one (batch, head) pair
    int bh_idx = blockIdx.x;
    int b = bh_idx / num_heads;
    int h = bh_idx % num_heads;
    
    if (b >= batch_size) return;
    
    int tid = threadIdx.x;
    int dv = head_dim_v;
    int dk = head_dim_qk;
    
    // State offset for this (batch, head)
    int s_offset = (b * num_heads + h) * dv * dk;
    
    // Shared memory for current k, v, q, a, and intermediate results
    extern __shared__ float shared_mem[];
    float* s_k = shared_mem;                    // dk
    float* s_v = s_k + dk;                      // dv
    float* s_q = s_v + dv;                      // dk
    float* s_a = s_q + dk;                      // dv
    float* s_Sk = s_a + dv;                     // dv (S @ k result)
    float* s_error = s_Sk + dv;                 // dv
    float* s_out = s_error + dv;                // dv
    
    // Process each timestep
    for (int t = 0; t < seq_len; t++) {
        // Load k, v, q, a for this timestep
        int qk_offset = ((b * seq_len + t) * num_heads + h) * dk;
        int v_offset = ((b * seq_len + t) * num_heads + h) * dv;
        int beta_offset = (b * seq_len + t) * num_heads + h;
        
        // Cooperative load of k, v, q, a
        for (int i = tid; i < dk; i += blockDim.x) {
            s_k[i] = k[qk_offset + i];
            s_q[i] = q[qk_offset + i];
        }
        for (int i = tid; i < dv; i += blockDim.x) {
            s_v[i] = v[v_offset + i];
            s_a[i] = a[v_offset + i];
        }
        __syncthreads();
        
        float beta_t = beta[beta_offset];
        
        // Compute S @ k -> s_Sk (each thread handles some rows of S)
        for (int i = tid; i < dv; i += blockDim.x) {
            float sum = 0.0f;
            int row_offset = s_offset + i * dk;
            for (int j = 0; j < dk; j++) {
                sum += S[row_offset + j] * s_k[j];
            }
            s_Sk[i] = sum;
        }
        __syncthreads();
        
        // Compute error = S @ k - v
        for (int i = tid; i < dv; i += blockDim.x) {
            s_error[i] = s_Sk[i] - s_v[i];
        }
        __syncthreads();
        
        // Update S: S = diag(a) @ S - beta * error @ k^T
        // Each thread handles multiple elements of S
        for (int i = tid; i < dv * dk; i += blockDim.x) {
            int row = i / dk;
            int col = i % dk;
            float s_old = S[s_offset + i];
            // Apply channel-wise gate and delta update
            S[s_offset + i] = s_a[row] * s_old - beta_t * s_error[row] * s_k[col];
        }
        __syncthreads();
        
        // Compute output: o = S @ q
        for (int i = tid; i < dv; i += blockDim.x) {
            float sum = 0.0f;
            int row_offset = s_offset + i * dk;
            for (int j = 0; j < dk; j++) {
                sum += S[row_offset + j] * s_q[j];
            }
            output[v_offset + i] = sum;
        }
        __syncthreads();
    }
}

torch::Tensor kda_forward_hip(
    torch::Tensor q,      // (batch, seq, heads, dk)
    torch::Tensor k,      // (batch, seq, heads, dk)
    torch::Tensor v,      // (batch, seq, heads, dv)
    torch::Tensor a,      // (batch, seq, heads, dv)
    torch::Tensor beta    // (batch, seq, heads)
) {
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int head_dim_qk = q.size(3);
    int head_dim_v = v.size(3);
    
    auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    
    // Output tensor
    auto output = torch::zeros({batch_size, seq_len, num_heads, head_dim_v}, options);
    
    // State tensor (initialized to zeros)
    auto S = torch::zeros({batch_size, num_heads, head_dim_v, head_dim_qk}, options);
    
    // Launch configuration
    int num_blocks = batch_size * num_heads;
    int threads_per_block = 256;
    
    // Shared memory: k(dk) + v(dv) + q(dk) + a(dv) + Sk(dv) + error(dv) + out(dv)
    size_t shared_mem_size = (head_dim_qk * 2 + head_dim_v * 5) * sizeof(float);
    
    hipLaunchKernelGGL(kda_recurrence_kernel, 
        dim3(num_blocks), dim3(threads_per_block), shared_mem_size, 0,
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        a.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        S.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim_qk, head_dim_v
    );
    
    return output;
}
"""

kda_cpp_source = """
torch::Tensor kda_forward_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
);
"""

kda_module = load_inline(
    name="kda_hip",
    cpp_sources=kda_cpp_source,
    cuda_sources=kda_hip_source,
    functions=["kda_forward_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
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
        
        # Store HIP module
        self.kda_hip = kda_module

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

        # Channel-wise gating
        a = torch.sigmoid(self.a_proj(x))
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()

        # Delta learning rate
        beta = torch.sigmoid(self.b_proj(x)).contiguous()

        # Scale keys
        k = k * self.scale

        # Use fused HIP kernel for recurrence
        o = self.kda_hip.kda_forward_hip(q, k, v, a, beta)

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


# Keep same configuration
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


def custom_kernel(inputs):
    x = inputs[0].cuda()
    model = ModelNew(hidden_size, num_heads, head_dim_qk, head_dim_v).cuda()
    model.eval()
    with torch.no_grad():
        return model(x)
