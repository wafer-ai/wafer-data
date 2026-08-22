import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# V5: Based on v3's successful approach with additional optimizations
# 1. Use float4 vectorized loads/stores for state matrix
# 2. Aggressive register usage
# 3. Better instruction scheduling

kda_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 512

__global__ void kda_recurrence_kernel_v5(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ a,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ S,
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
    int dv = head_dim_v;  // 128
    int dk = head_dim_qk; // 128
    
    int s_offset = (b * num_heads + h) * dv * dk;
    
    extern __shared__ float shared_mem[];
    float* s_k = shared_mem;            // 128
    float* s_v = s_k + dk;              // 128
    float* s_q = s_v + dv;              // 128
    float* s_a = s_q + dk;              // 128
    float* s_Sk = s_a + dv;             // 128
    
    // Pre-compute strides for better memory access
    const int qk_stride = num_heads * dk;
    const int v_stride = num_heads * dv;
    const int beta_stride = num_heads;
    
    int base_qk_offset = (b * seq_len * num_heads + h) * dk;
    int base_v_offset = (b * seq_len * num_heads + h) * dv;
    int base_beta_offset = b * seq_len * num_heads + h;
    
    for (int t = 0; t < seq_len; t++) {
        int qk_offset = base_qk_offset + t * qk_stride;
        int v_offset = base_v_offset + t * v_stride;
        int beta_offset = base_beta_offset + t * beta_stride;
        
        // Load vectors - use float4 where possible
        if (tid < 32) {
            // Load k: 128 floats = 32 float4s
            float4* k_vec = (float4*)s_k;
            const float4* k_src = (const float4*)(k + qk_offset);
            k_vec[tid] = k_src[tid];
            
            // Load q
            float4* q_vec = (float4*)s_q;
            const float4* q_src = (const float4*)(q + qk_offset);
            q_vec[tid] = q_src[tid];
            
            // Load v
            float4* v_vec = (float4*)s_v;
            const float4* v_src = (const float4*)(v + v_offset);
            v_vec[tid] = v_src[tid];
            
            // Load a
            float4* a_vec = (float4*)s_a;
            const float4* a_src = (const float4*)(a + v_offset);
            a_vec[tid] = a_src[tid];
        }
        __syncthreads();
        
        float beta_t = beta[beta_offset];
        
        // Compute S @ k - only first 128 threads needed
        if (tid < dv) {
            float sum = 0.0f;
            int row_offset = s_offset + tid * dk;
            
            // Use float4 for loading state
            const float4* S_row = (const float4*)(S + row_offset);
            const float4* k_vec = (const float4*)s_k;
            
            #pragma unroll 32
            for (int j = 0; j < 32; j++) {
                float4 s_val = S_row[j];
                float4 k_val = k_vec[j];
                sum += s_val.x * k_val.x + s_val.y * k_val.y + 
                       s_val.z * k_val.z + s_val.w * k_val.w;
            }
            s_Sk[tid] = sum;
        }
        __syncthreads();
        
        // Update S using float4 operations
        // 16384 elements / 4 = 4096 float4s
        // 512 threads -> 8 float4s per thread
        int num_vec4 = (dv * dk) / 4;
        for (int i = tid; i < num_vec4; i += BLOCK_SIZE) {
            int elem_idx = i * 4;
            int row = elem_idx / dk;
            int col = elem_idx % dk;
            
            // Read state as float4
            float4* S_ptr = (float4*)(S + s_offset);
            float4 s_old = S_ptr[i];
            
            float gate = s_a[row];
            float error = s_Sk[row] - s_v[row];
            float be = beta_t * error;
            
            // Read k values
            float4 k_val = ((float4*)s_k)[col / 4];
            
            // Update each component
            float4 s_new;
            s_new.x = gate * s_old.x - be * k_val.x;
            s_new.y = gate * s_old.y - be * k_val.y;
            s_new.z = gate * s_old.z - be * k_val.z;
            s_new.w = gate * s_old.w - be * k_val.w;
            
            S_ptr[i] = s_new;
        }
        __syncthreads();
        
        // Compute output o = S @ q
        if (tid < dv) {
            float sum = 0.0f;
            int row_offset = s_offset + tid * dk;
            
            const float4* S_row = (const float4*)(S + row_offset);
            const float4* q_vec = (const float4*)s_q;
            
            #pragma unroll 32
            for (int j = 0; j < 32; j++) {
                float4 s_val = S_row[j];
                float4 q_val = q_vec[j];
                sum += s_val.x * q_val.x + s_val.y * q_val.y + 
                       s_val.z * q_val.z + s_val.w * q_val.w;
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
    
    size_t shared_mem_size = (head_dim_qk * 2 + head_dim_v * 3) * sizeof(float);
    
    hipLaunchKernelGGL(kda_recurrence_kernel_v5, 
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
    name="kda_hip_v5",
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
