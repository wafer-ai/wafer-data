import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define HEAD_DIM 128

__global__ void kimi_delta_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ a,
    const float* __restrict__ beta,
    float* __restrict__ o,
    int seq_len,
    int num_heads,
    int stride_t,
    int stride_beta_t,
    float scale
) {
    // Grid: (batch_size, num_heads)
    // Block: (HEAD_DIM, 1, 1) -> 128 threads
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    // Offsets
    // q, k, v, a, o are (B, T, H, D)
    // stride_t = H * D
    // stride for H is D (128)
    
    // Base pointers for this batch and head
    long long offset_base = (long long)b * (seq_len * stride_t) + h * HEAD_DIM;
    
    const float* q_ptr = q + offset_base;
    const float* k_ptr = k + offset_base;
    const float* v_ptr = v + offset_base;
    const float* a_ptr = a + offset_base;
    float* o_ptr = o + offset_base;
    
    // Beta is (B, T, H)
    // stride_beta_t = H
    // stride for H is 1
    long long offset_beta = (long long)b * (seq_len * stride_beta_t) + h;
    const float* beta_ptr = beta + offset_beta;

    // State S[row] kept in registers
    // Thread i handles row i of S.
    float s_row[HEAD_DIM];
    
    // Initialize state to 0
    #pragma unroll
    for (int j = 0; j < HEAD_DIM; ++j) {
        s_row[j] = 0.0f;
    }
    
    __shared__ float k_shared[HEAD_DIM];
    __shared__ float q_shared[HEAD_DIM];

    for (int t = 0; t < seq_len; ++t) {
        int offset_step = t * stride_t;
        
        // Load k and q to shared
        // Apply scale to k immediately
        k_shared[tid] = k_ptr[offset_step + tid] * scale;
        q_shared[tid] = q_ptr[offset_step + tid];
        
        float v_val = v_ptr[offset_step + tid];
        float a_val = a_ptr[offset_step + tid];
        
        float beta_val = beta_ptr[t * stride_beta_t]; 
        
        __syncthreads();
        
        // 1. Compute y = S * k
        float y_val = 0.0f;
        #pragma unroll
        for (int j = 0; j < HEAD_DIM; ++j) {
            y_val += s_row[j] * k_shared[j];
        }
        
        // 2. Error
        float error = y_val - v_val;
        
        // 3. Update S
        #pragma unroll
        for (int j = 0; j < HEAD_DIM; ++j) {
            s_row[j] = a_val * s_row[j] - beta_val * error * k_shared[j];
        }
        
        // 4. Output
        float out_val = 0.0f;
        #pragma unroll
        for (int j = 0; j < HEAD_DIM; ++j) {
            out_val += s_row[j] * q_shared[j];
        }
        
        o_ptr[offset_step + tid] = out_val;
        
        __syncthreads();
    }
}

void kimi_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta,
    torch::Tensor o,
    float scale
) {
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    // head_dim is 128
    
    int stride_t = num_heads * HEAD_DIM;
    int stride_beta_t = num_heads;
    
    dim3 grid(batch_size, num_heads);
    dim3 block(HEAD_DIM);
    
    kimi_delta_kernel<<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        a.data_ptr<float>(),
        beta.data_ptr<float>(),
        o.data_ptr<float>(),
        seq_len, num_heads,
        stride_t, stride_beta_t,
        scale
    );
}
"""

kimi_ops = load_inline(
    name="kimi_ops",
    cpp_sources=cpp_source,
    functions=["kimi_forward_cuda"],
    extra_cflags=["-O3"],
    verbose=True
)

class ModelNew(nn.Module):
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

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        a = torch.sigmoid(self.a_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        beta = torch.sigmoid(self.b_proj(x))

        # Ensure contiguous memory for kernel
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        beta = beta.contiguous()
        
        o = torch.empty_like(v)
        
        # Check dimensions
        assert self.head_dim_qk == 128 and self.head_dim_v == 128, "Kernel optimized for 128 dim"
        
        kimi_ops.kimi_forward_cuda(
            q, k, v, a, beta, o, self.scale
        )
        
        o = self.o_norm(o)
        g = torch.sigmoid(self.g_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim_v)
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
    return [torch.randn(batch_size, seq_len, hidden_size).cuda()]

def get_init_inputs():
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
