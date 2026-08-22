
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_Q 96
#define TILE_K 8
#define HEAD_DIM 128
#define HEAD_DIM_PAD 132
#define THREADS_PER_Q 4

__global__ void gqa_kernel_opt(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    const int stride_q_b, const int stride_q_h, const int stride_q_l, const int stride_q_d,
    const int stride_k_b, const int stride_k_h, const int stride_k_l, const int stride_k_d,
    const int stride_v_b, const int stride_v_h, const int stride_v_l, const int stride_v_d,
    const int stride_o_b, const int stride_o_h, const int stride_o_l, const int stride_o_d,
    const int num_kv_groups,
    const int seq_len,
    const float scale)
{
    int batch_idx = blockIdx.z;
    int head_q_idx = blockIdx.y;
    int head_kv_idx = head_q_idx / num_kv_groups;
    int q_chunk_idx = blockIdx.x;
    
    int tid = threadIdx.x; 
    
    __shared__ float s_Q[TILE_Q][HEAD_DIM_PAD];
    __shared__ float s_K[TILE_K][HEAD_DIM_PAD];
    __shared__ float s_V[TILE_K][HEAD_DIM_PAD];
    
    int q_start = q_chunk_idx * TILE_Q;
    long long q_base_offset = (long long)batch_idx * stride_q_b + (long long)head_q_idx * stride_q_h;
    
    // Vectorized Load Q
    // TILE_Q * HEAD_DIM = 96 * 128 = 12288 floats = 3072 float4s.
    // 384 threads. 3072 / 384 = 8 iterations.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx_linear = tid + i * 384;
        int r = (idx_linear * 4) / HEAD_DIM;
        int c = (idx_linear * 4) % HEAD_DIM;
        
        int q_idx_global = q_start + r;
        
        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if (q_idx_global < seq_len) {
            long long offset = q_base_offset + (long long)q_idx_global * stride_q_l + c;
            val = *reinterpret_cast<const float4*>(&Q[offset]);
        }
        *reinterpret_cast<float4*>(&s_Q[r][c]) = val;
    }
    
    __syncthreads();
    
    int q_local = tid / THREADS_PER_Q; 
    int lane = tid % THREADS_PER_Q; 
    
    float4 my_out[8];
    #pragma unroll
    for(int i=0; i<8; ++i) my_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float max_score = -1e30f;
    float sum_exp = 0.0f;
    
    int num_k_tiles = (seq_len + TILE_K - 1) / TILE_K;
    long long k_base_offset = (long long)batch_idx * stride_k_b + (long long)head_kv_idx * stride_k_h;
    long long v_base_offset = (long long)batch_idx * stride_v_b + (long long)head_kv_idx * stride_v_h;

    for (int t = 0; t < num_k_tiles; ++t) {
        int k_start = t * TILE_K;
        
        __syncthreads(); 
        
        // Load K, V Vectorized
        // TILE_K * HEAD_DIM = 8 * 128 = 1024 floats = 256 float4s.
        // 384 threads. First 256 threads load 1.
        if (tid < 256) {
            int idx_linear = tid;
            int r = (idx_linear * 4) / HEAD_DIM;
            int c = (idx_linear * 4) % HEAD_DIM;
            int k_idx_global = k_start + r;
            
            float4 k_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 v_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            
            if (k_idx_global < seq_len) {
                long long k_off = k_base_offset + (long long)k_idx_global * stride_k_l + c;
                long long v_off = v_base_offset + (long long)k_idx_global * stride_v_l + c;
                k_val = *reinterpret_cast<const float4*>(&K[k_off]);
                v_val = *reinterpret_cast<const float4*>(&V[v_off]);
            }
            
            *reinterpret_cast<float4*>(&s_K[r][c]) = k_val;
            *reinterpret_cast<float4*>(&s_V[r][c]) = v_val;
        }
        __syncthreads();
        
        if (q_start + q_local < seq_len) {
            int q_idx_g = q_start + q_local;
            
            for (int k = 0; k < TILE_K; ++k) {
                int k_idx_g = k_start + k;
                if (k_idx_g >= seq_len) break;
                
                if (k_idx_g > q_idx_g) continue;
                
                float dot = 0.0f;
                int c_start = lane * 32;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int d = c_start + i * 4;
                    float4 q_v = *reinterpret_cast<float4*>(&s_Q[q_local][d]);
                    float4 k_v = *reinterpret_cast<float4*>(&s_K[k][d]);
                    dot += q_v.x * k_v.x + q_v.y * k_v.y + q_v.z * k_v.z + q_v.w * k_v.w;
                }
                
                dot += __shfl_xor(dot, 1);
                dot += __shfl_xor(dot, 2);
                
                float score = dot * scale;
                float new_max = max(max_score, score);
                float exp_s = expf(score - new_max);
                float correction = expf(max_score - new_max);
                
                max_score = new_max;
                sum_exp = sum_exp * correction + exp_s;
                
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int d = c_start + i * 4;
                    float4 v_v = *reinterpret_cast<float4*>(&s_V[k][d]);
                    
                    my_out[i].x = my_out[i].x * correction + exp_s * v_v.x;
                    my_out[i].y = my_out[i].y * correction + exp_s * v_v.y;
                    my_out[i].z = my_out[i].z * correction + exp_s * v_v.z;
                    my_out[i].w = my_out[i].w * correction + exp_s * v_v.w;
                }
            }
        }
    }
    
    if (q_start + q_local < seq_len) {
        float inv_sum = 1.0f / (sum_exp + 1e-6f);
        long long out_base = (long long)batch_idx * stride_o_b + 
                             (long long)head_q_idx * stride_o_h + 
                             (long long)(q_start + q_local) * stride_o_l;
                             
        int c_start = lane * 32;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
             int d = c_start + i * 4;
             float4 out_val;
             out_val.x = my_out[i].x * inv_sum;
             out_val.y = my_out[i].y * inv_sum;
             out_val.z = my_out[i].z * inv_sum;
             out_val.w = my_out[i].w * inv_sum;
             
             *reinterpret_cast<float4*>(&Out[out_base + d]) = out_val;
        }
    }
}

torch::Tensor gqa_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double scale) 
{
    auto B = Q.size(0);
    auto H_Q = Q.size(1);
    auto L = Q.size(2);
    auto H_KV = K.size(1);
    
    auto Out = torch::empty_like(Q);
    int group_size = H_Q / H_KV;
    
    dim3 grid((L + 95) / 96, H_Q, B);
    dim3 block(384); 
    
    gqa_kernel_opt<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        Out.data_ptr<float>(),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        group_size,
        L,
        (float)scale
    );
    
    return Out;
}
"""

gqa_cpp = load_inline(
    name="gqa_cpp_opt_v3",
    cpp_sources=cpp_source,
    functions=["gqa_forward"],
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout
        self.softmax_scale = head_dim ** (-0.5)

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = gqa_cpp.gqa_forward(
            query_states, 
            key_states, 
            value_states, 
            self.softmax_scale
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output
