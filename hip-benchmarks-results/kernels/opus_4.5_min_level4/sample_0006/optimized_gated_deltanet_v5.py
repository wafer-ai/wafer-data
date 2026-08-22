import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# Optimized delta rule kernel with warp-level reductions
gated_deltanet_cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define WARP_SIZE 64
#define D_QK 128
#define D_V 128

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Kernel with explicit d_qk=128, d_v=128 for better optimization
__global__ void gated_deltanet_recurrence_kernel(
    const float* __restrict__ q,      // (batch, seq, heads, 128)
    const float* __restrict__ k,      // (batch, seq, heads, 128)
    const float* __restrict__ v,      // (batch, seq, heads, 128)
    const float* __restrict__ alpha,  // (batch, seq, heads)
    const float* __restrict__ beta,   // (batch, seq, heads)
    float* __restrict__ state,        // (batch, heads, 128, 128)
    float* __restrict__ output,       // (batch, seq, heads, 128)
    int batch_size,
    int seq_len,
    int num_heads
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    extern __shared__ float shared_mem[];
    float* k_shared = shared_mem;                // 128
    float* v_shared = k_shared + D_QK;           // 128
    float* S_k = v_shared + D_V;                 // 128
    float* q_shared = S_k + D_V;                 // 128
    float* partial_sums = q_shared + D_QK;       // For warp reduction
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    constexpr int state_size = D_V * D_QK;
    
    float* S = state + (batch_idx * num_heads + head_idx) * state_size;
    
    // Initialize state
    for (int i = tid; i < state_size; i += blockDim.x) {
        S[i] = 0.0f;
    }
    __syncthreads();
    
    for (int t = 0; t < seq_len; t++) {
        int qkv_base = ((batch_idx * seq_len + t) * num_heads + head_idx);
        int ab_idx = (batch_idx * seq_len + t) * num_heads + head_idx;
        
        // Load k and v into shared memory  
        for (int i = tid; i < D_QK; i += blockDim.x) {
            k_shared[i] = k[qkv_base * D_QK + i];
        }
        for (int i = tid; i < D_V; i += blockDim.x) {
            v_shared[i] = v[qkv_base * D_V + i];
        }
        __syncthreads();
        
        float alpha_t = alpha[ab_idx];
        float beta_t = beta[ab_idx];
        
        // Compute S_k = S @ k using all threads
        // Each thread handles one row (or multiple rows if needed)
        for (int i = tid; i < D_V; i += blockDim.x) {
            float sum = 0.0f;
            int base = i * D_QK;
            
            // Fully unroll for D_QK=128
            #pragma unroll 16
            for (int j = 0; j < D_QK; j += 8) {
                sum += S[base + j + 0] * k_shared[j + 0];
                sum += S[base + j + 1] * k_shared[j + 1];
                sum += S[base + j + 2] * k_shared[j + 2];
                sum += S[base + j + 3] * k_shared[j + 3];
                sum += S[base + j + 4] * k_shared[j + 4];
                sum += S[base + j + 5] * k_shared[j + 5];
                sum += S[base + j + 6] * k_shared[j + 6];
                sum += S[base + j + 7] * k_shared[j + 7];
            }
            S_k[i] = sum;
        }
        __syncthreads();
        
        // Update state: S = alpha * S - beta * (S_k - v) @ k^T
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / D_QK;
            int j = idx % D_QK;
            float error_i = S_k[i] - v_shared[i];
            float update = alpha_t * S[idx] - beta_t * error_i * k_shared[j];
            S[idx] = update;
        }
        __syncthreads();
        
        // Load q
        for (int i = tid; i < D_QK; i += blockDim.x) {
            q_shared[i] = q[qkv_base * D_QK + i];
        }
        __syncthreads();
        
        // Output = S @ q
        int out_base = qkv_base * D_V;
        for (int i = tid; i < D_V; i += blockDim.x) {
            float sum = 0.0f;
            int base = i * D_QK;
            
            #pragma unroll 16
            for (int j = 0; j < D_QK; j += 8) {
                sum += S[base + j + 0] * q_shared[j + 0];
                sum += S[base + j + 1] * q_shared[j + 1];
                sum += S[base + j + 2] * q_shared[j + 2];
                sum += S[base + j + 3] * q_shared[j + 3];
                sum += S[base + j + 4] * q_shared[j + 4];
                sum += S[base + j + 5] * q_shared[j + 5];
                sum += S[base + j + 6] * q_shared[j + 6];
                sum += S[base + j + 7] * q_shared[j + 7];
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
    int d_qk = q.size(3);
    int d_v = v.size(3);
    
    auto output = torch::zeros({batch_size, seq_len, num_heads, d_v}, q.options());
    auto state = torch::zeros({batch_size, num_heads, d_v, d_qk}, q.options());
    
    dim3 grid(batch_size, num_heads);
    int block_size = 512;
    
    // Shared memory: k[128] + v[128] + S_k[128] + q[128] + partial_sums[8]
    size_t shared_mem_size = (2 * d_qk + 2 * d_v + 8) * sizeof(float);
    
    gated_deltanet_recurrence_kernel<<<grid, block_size, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        alpha.data_ptr<float>(),
        beta.data_ptr<float>(),
        state.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads
    );
    
    return output;
}
"""

gated_deltanet_cpp_header = """
torch::Tensor gated_deltanet_recurrence(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta
);
"""

gated_deltanet_module = load_inline(
    name="gated_deltanet_v5",
    cpp_sources=gated_deltanet_cpp_header,
    cuda_sources=gated_deltanet_cpp_source,
    functions=["gated_deltanet_recurrence"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
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

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

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

        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        self.scale = head_dim_qk ** -0.5
        self.gated_deltanet = gated_deltanet_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).contiguous()
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()

        alpha = torch.sigmoid(self.a_proj(x)).contiguous()
        beta = torch.sigmoid(self.b_proj(x)).contiguous()

        k = k * self.scale

        o = self.gated_deltanet.gated_deltanet_recurrence(q, k, v, alpha, beta)

        o = self.o_norm(o)

        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o


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
