import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math
import os

# Set compiler to hipcc for ROCm
os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

#define WARP_SIZE 64

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Optimized kernel for Gated DeltaNet recurrence
// Assumptions: 
// - HEAD_DIM is 128 (can be templated, but hardcoded/checked for this optimization)
// - BlockDim.x = 256 (4 warps)
// - Processes 4 rows of Dv per block
template <int HEAD_DIM>
__global__ void gated_deltanet_fwd_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ o,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim_v
) {
    // Shared memory for k and q (HEAD_DIM each)
    extern __shared__ float shared_mem[];
    float* k_shared = shared_mem;              // [HEAD_DIM]
    float* q_shared = shared_mem + HEAD_DIM;   // [HEAD_DIM]

    // Determine workload
    // Grid: (head_dim_v / 4, num_heads, batch_size)
    const int rows_per_block = 4;
    int row_start = blockIdx.x * rows_per_block;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Each warp handles one row of the output/value dimension
    int my_row = row_start + warp_id;
    
    // Bounds check
    if (my_row >= head_dim_v) return;

    // Strides
    // q, k, v: [B, T, H, D]
    // Flat index: b * (T*H*D) + t * (H*D) + h * D + d
    long stride_seq_q = (long)num_heads * HEAD_DIM;
    long stride_seq_v = (long)num_heads * head_dim_v;
    long stride_head_q = HEAD_DIM;
    long stride_head_v = head_dim_v;
    
    // alpha, beta: [B, T, H]
    long stride_seq_s = num_heads;
    
    // Base pointers for this sequence and head
    long batch_offset_q = (long)batch_idx * seq_len * stride_seq_q + (long)head_idx * stride_head_q;
    long batch_offset_v = (long)batch_idx * seq_len * stride_seq_v + (long)head_idx * stride_head_v;
    long batch_offset_s = (long)batch_idx * seq_len * stride_seq_s + (long)head_idx;
    
    const float* q_ptr = q + batch_offset_q;
    const float* k_ptr = k + batch_offset_q; // k has same shape as q
    const float* v_ptr = v + batch_offset_v;
    const float* a_ptr = alpha + batch_offset_s;
    const float* b_ptr = beta + batch_offset_s;
    float* o_ptr = o + batch_offset_v;

    // Initialize state
    // Distributed state: each thread holds (HEAD_DIM / WARP_SIZE) elements
    // For 128 dim, 64 threads -> 2 elements per thread.
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / WARP_SIZE;
    float s[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        s[i] = 0.0f;
    }

    for (int t = 0; t < seq_len; ++t) {
        // 1. Cooperative load k_t and q_t into shared memory
        // We need to load 2 * HEAD_DIM floats.
        // Block size is 256. 2 * 128 = 256.
        // Perfect mapping: each thread loads 1 float.
        // Or if block size != 256 or dims different, use loops.
        // Here assuming block_size=256 and HEAD_DIM=128 for max optimization.
        
        if (tid < HEAD_DIM) {
            k_shared[tid] = k_ptr[t * stride_seq_q + tid];
        } else if (tid < 2 * HEAD_DIM) {
            q_shared[tid - HEAD_DIM] = q_ptr[t * stride_seq_q + (tid - HEAD_DIM)];
        }
        __syncthreads();

        // 2. Load scalars
        float a_val = a_ptr[t * stride_seq_s];
        float b_val = b_ptr[t * stride_seq_s];
        // Load v value for this row (same for all threads in warp)
        float v_val = v_ptr[t * stride_seq_v + my_row];

        // 3. Compute dot(s, k)
        float dot_sk = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            // s[i] corresponds to index (i * WARP_SIZE + lane_id)
            dot_sk += s[i] * k_shared[i * WARP_SIZE + lane_id];
        }
        dot_sk = warp_reduce_sum(dot_sk);
        // Broadcast result to all lanes in warp
        dot_sk = __shfl(dot_sk, 0);

        // 4. Compute error
        float error = dot_sk - v_val;
        
        // 5. Update S and compute dot(s_new, q)
        float dot_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int k_idx = i * WARP_SIZE + lane_id;
            float k_val = k_shared[k_idx];
            
            // s = alpha * s - beta * error * k
            s[i] = a_val * s[i] - b_val * error * k_val;
            
            // accumulate dot(s, q)
            dot_sq += s[i] * q_shared[k_idx];
        }

        // 6. Output
        dot_sq = warp_reduce_sum(dot_sq);
        
        if (lane_id == 0) {
            o_ptr[t * stride_seq_v + my_row] = dot_sq;
        }
        
        // Barrier before next iteration overwrites shared memory
        __syncthreads();
    }
}

torch::Tensor gated_deltanet_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta
) {
    auto B = q.size(0);
    auto T = q.size(1);
    auto H = q.size(2);
    auto D_k = q.size(3);
    auto D_v = v.size(3);
    
    auto o = torch::empty_like(v);
    
    // Grid configuration
    // Block size 256 (4 warps)
    int block_size = 256;
    int rows_per_block = 4;
    
    // X dimension: blocks needed for D_v rows
    int grid_x = (D_v + rows_per_block - 1) / rows_per_block;
    int grid_y = H;
    int grid_z = B;
    
    dim3 grid(grid_x, grid_y, grid_z);
    
    // Shared mem: 2 * D_k * 4 bytes
    int shared_mem_size = 2 * D_k * 4;
    
    // Dispatch
    if (D_k == 128) {
        gated_deltanet_fwd_kernel<128><<<grid, block_size, shared_mem_size>>>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            alpha.data_ptr<float>(),
            beta.data_ptr<float>(),
            o.data_ptr<float>(),
            B, T, H, D_v
        );
    } else {
        // Fallback for non-optimized dimensions if needed, 
        // but for this problem we expect D_k=128.
        // Trigger error or just fail? 
        // We will assume 128 as per problem statement optimization target.
        TORCH_CHECK(false, "Unsupported head_dim_qk for optimized kernel (expected 128)");
    }
    
    return o;
}
"""

gated_deltanet_module = load_inline(
    name="gated_deltanet_kernels",
    cpp_sources=cpp_source,
    functions=["gated_deltanet_fwd"],
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

        # Scaling factor
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
        alpha = torch.sigmoid(self.a_proj(x))
        beta = torch.sigmoid(self.b_proj(x))

        # Scale keys
        k = k * self.scale

        # Run optimized kernel
        # Ensure memory is contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        alpha = alpha.contiguous()
        beta = beta.contiguous()
        
        o = gated_deltanet_module.gated_deltanet_fwd(q, k, v, alpha, beta)
        
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
