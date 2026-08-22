import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Best kernel based on v5 approach with 256 threads
kda_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64

// 256 threads: threads 0-127 handle first 64 cols, threads 128-255 handle last 64 cols
// No atomics version - use warp shuffle for reduction
__global__ __launch_bounds__(256, 4)
void kda_recurrence_kernel_v8(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ a,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim_qk,
    int head_dim_v
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    const int tid = threadIdx.x;
    
    // Shared memory
    extern __shared__ float shared_mem[];
    float* k_shared = shared_mem;                          // 128
    float* q_shared = shared_mem + 128;                    // 128
    float* v_shared = shared_mem + 256;                    // 128
    float* a_shared = shared_mem + 384;                    // 128
    float* reduce_shared = shared_mem + 512;               // 256 for reduction
    
    // Thread layout: tid 0-127 handle cols 0-63, tid 128-255 handle cols 64-127
    const int row = tid & 127;         // Which row this thread contributes to (0-127)
    const int col_half = tid >> 7;     // 0 for first half, 1 for second half
    const int col_offset = col_half * 64;
    
    // Each thread handles half of a row (64 elements)
    float S_half[64];
    
    #pragma unroll
    for (int c = 0; c < 64; c++) {
        S_half[c] = 0.0f;
    }
    
    // Strides
    const int stride_qk = num_heads * 128;
    const int stride_v = num_heads * 128;
    const int stride_beta = num_heads;
    
    const int base_qk = batch_idx * seq_len * stride_qk + head_idx * 128;
    const int base_v = batch_idx * seq_len * stride_v + head_idx * 128;
    const int base_beta = batch_idx * seq_len * stride_beta + head_idx;
    
    // Main loop
    for (int t = 0; t < seq_len; t++) {
        const int off_qk = base_qk + t * stride_qk;
        const int off_v = base_v + t * stride_v;
        const int off_beta = base_beta + t * stride_beta;
        
        // Load data - coalesced
        if (tid < 128) {
            k_shared[tid] = k[off_qk + tid];
            q_shared[tid] = q[off_qk + tid];
            v_shared[tid] = v[off_v + tid];
            a_shared[tid] = a[off_v + tid];
        }
        __syncthreads();
        
        const float beta_t = beta[off_beta];
        
        // Compute partial dot product S[row, col_offset:col_offset+64] @ k[col_offset:col_offset+64]
        float partial = 0.0f;
        #pragma unroll 8
        for (int c = 0; c < 64; c++) {
            partial += S_half[c] * k_shared[col_offset + c];
        }
        
        // Store in reduce buffer for this thread
        reduce_shared[tid] = partial;
        __syncthreads();
        
        // Reduce: thread tid and tid^128 hold two halves of the same row
        // Thread in first half does the reduction
        float Sk;
        if (col_half == 0) {
            Sk = reduce_shared[row] + reduce_shared[row + 128];
            reduce_shared[row] = Sk;  // Store final Sk for row
        }
        __syncthreads();
        
        // Read Sk, a, v for this row
        const float Sk_row = reduce_shared[row];
        const float a_r = a_shared[row];
        const float error = Sk_row - v_shared[row];
        const float coeff = beta_t * error;
        
        // Update state
        #pragma unroll 8
        for (int c = 0; c < 64; c++) {
            S_half[c] = a_r * S_half[c] - coeff * k_shared[col_offset + c];
        }
        __syncthreads();
        
        // Compute partial output S[row, :] @ q
        float partial_out = 0.0f;
        #pragma unroll 8
        for (int c = 0; c < 64; c++) {
            partial_out += S_half[c] * q_shared[col_offset + c];
        }
        
        reduce_shared[tid] = partial_out;
        __syncthreads();
        
        // Reduce for output
        if (col_half == 0) {
            float out = reduce_shared[row] + reduce_shared[row + 128];
            output[off_v + row] = out;
        }
        __syncthreads();
    }
}

torch::Tensor kda_recurrence_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
) {
    auto batch_size = q.size(0);
    auto seq_len = q.size(1);
    auto num_heads = q.size(2);
    auto head_dim_qk = q.size(3);
    auto head_dim_v = v.size(3);
    
    auto output = torch::zeros_like(v);
    
    dim3 grid(batch_size, num_heads);
    int block_size = 256;
    
    // k(128) + q(128) + v(128) + a(128) + reduce(256)
    int shared_mem_size = (128 * 4 + 256) * sizeof(float);
    
    kda_recurrence_kernel_v8<<<grid, block_size, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        a.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim_qk,
        head_dim_v
    );
    
    return output;
}
"""

kda_cpp_source = """
torch::Tensor kda_recurrence_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
);
"""

kda_module = load_inline(
    name="kda_module_v8",
    cpp_sources=kda_cpp_source,
    cuda_sources=kda_kernel_source,
    functions=["kda_recurrence_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
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

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        if use_dplr:
            self.l_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)
            self.r_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)

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

        a = torch.sigmoid(self.a_proj(x))
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v).contiguous()
        beta = torch.sigmoid(self.b_proj(x)).contiguous()

        k = k * self.scale

        o = kda_module.kda_recurrence_hip(q, k, v, a, beta)

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
