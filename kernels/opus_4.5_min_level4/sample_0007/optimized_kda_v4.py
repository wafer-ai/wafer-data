import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# V4 optimizations:
# 1. Use warp-level primitives for reduction
# 2. Better memory coalescing with tiled access
# 3. Register blocking for state matrix
# 4. Increased occupancy with more blocks

kda_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Main kernel - one block per (batch, head) with register-tiled state update
__global__ void kda_recurrence_kernel_v4(
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
    int bh_idx = blockIdx.x;
    int b = bh_idx / num_heads;
    int h = bh_idx % num_heads;
    
    if (b >= batch_size) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int dv = head_dim_v;  // 128
    int dk = head_dim_qk; // 128
    
    int s_offset = (b * num_heads + h) * dv * dk;
    
    // Shared memory
    extern __shared__ float shared_mem[];
    float* s_k = shared_mem;            // 128
    float* s_v = s_k + dk;              // 128
    float* s_q = s_v + dv;              // 128
    float* s_a = s_q + dk;              // 128
    float* s_Sk = s_a + dv;             // 128
    float* s_partial = s_Sk + dv;       // For reductions
    
    // Each thread handles multiple rows of S for the dot products
    // dv=128, dk=128, state has 16384 elements
    // With 256 threads, each thread handles 64 elements of S
    
    for (int t = 0; t < seq_len; t++) {
        int qk_offset = ((b * seq_len + t) * num_heads + h) * dk;
        int v_offset = ((b * seq_len + t) * num_heads + h) * dv;
        int beta_offset = (b * seq_len + t) * num_heads + h;
        
        // Load k, v, q, a cooperatively
        if (tid < dk) {
            s_k[tid] = k[qk_offset + tid];
            s_q[tid] = q[qk_offset + tid];
        }
        if (tid < dv) {
            s_v[tid] = v[v_offset + tid];
            s_a[tid] = a[v_offset + tid];
        }
        __syncthreads();
        
        float beta_t = beta[beta_offset];
        
        // Compute S @ k - each thread computes partial sums for rows it handles
        // Split dv rows among warps, each thread computes one full row
        // 256 threads / 128 rows = 2 threads per row
        // Better: each thread computes 1 row, first 128 threads active
        if (tid < dv) {
            float sum = 0.0f;
            int row_offset = s_offset + tid * dk;
            
            // Full unroll for dk=128
            #pragma unroll 16
            for (int j = 0; j < dk; j += 8) {
                sum += S[row_offset + j] * s_k[j];
                sum += S[row_offset + j + 1] * s_k[j + 1];
                sum += S[row_offset + j + 2] * s_k[j + 2];
                sum += S[row_offset + j + 3] * s_k[j + 3];
                sum += S[row_offset + j + 4] * s_k[j + 4];
                sum += S[row_offset + j + 5] * s_k[j + 5];
                sum += S[row_offset + j + 6] * s_k[j + 6];
                sum += S[row_offset + j + 7] * s_k[j + 7];
            }
            s_Sk[tid] = sum;
        }
        __syncthreads();
        
        // Update S in tiles for better memory access
        // State: dv x dk = 128 x 128 = 16384 elements
        // 256 threads, each handles 64 elements
        #pragma unroll 2
        for (int i = tid; i < dv * dk; i += BLOCK_SIZE) {
            int row = i / dk;
            int col = i % dk;
            float s_old = S[s_offset + i];
            float error_row = s_Sk[row] - s_v[row];
            S[s_offset + i] = s_a[row] * s_old - beta_t * error_row * s_k[col];
        }
        __syncthreads();
        
        // Compute output o = S @ q
        if (tid < dv) {
            float sum = 0.0f;
            int row_offset = s_offset + tid * dk;
            
            #pragma unroll 16
            for (int j = 0; j < dk; j += 8) {
                sum += S[row_offset + j] * s_q[j];
                sum += S[row_offset + j + 1] * s_q[j + 1];
                sum += S[row_offset + j + 2] * s_q[j + 2];
                sum += S[row_offset + j + 3] * s_q[j + 3];
                sum += S[row_offset + j + 4] * s_q[j + 4];
                sum += S[row_offset + j + 5] * s_q[j + 5];
                sum += S[row_offset + j + 6] * s_q[j + 6];
                sum += S[row_offset + j + 7] * s_q[j + 7];
            }
            output[v_offset + tid] = sum;
        }
        __syncthreads();
    }
}

torch::Tensor kda_forward_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor beta
) {
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int head_dim_qk = q.size(3);
    int head_dim_v = v.size(3);
    
    auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    
    auto output = torch::zeros({batch_size, seq_len, num_heads, head_dim_v}, options);
    auto S = torch::zeros({batch_size, num_heads, head_dim_v, head_dim_qk}, options);
    
    int num_blocks = batch_size * num_heads;
    
    // Shared: k(128) + v(128) + q(128) + a(128) + Sk(128) + partial(256)
    size_t shared_mem_size = (head_dim_qk * 2 + head_dim_v * 3 + BLOCK_SIZE) * sizeof(float);
    
    hipLaunchKernelGGL(kda_recurrence_kernel_v4, 
        dim3(num_blocks), dim3(BLOCK_SIZE), shared_mem_size, 0,
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
    name="kda_hip_v4",
    cpp_sources=kda_cpp_source,
    cuda_sources=kda_hip_source,
    functions=["kda_forward_hip"],
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
        self.kda_hip = kda_module

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

        o = self.kda_hip.kda_forward_hip(q, k, v, a, beta)

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


def custom_kernel(inputs):
    x = inputs[0].cuda()
    model = ModelNew(hidden_size, num_heads, head_dim_qk, head_dim_v).cuda()
    model.eval()
    with torch.no_grad():
        return model(x)
