
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

#define HS 96
#define TR 64
#define TC 32

inline __device__ float4 load_float4(const float* ptr, int idx) {
    return reinterpret_cast<const float4*>(ptr)[idx];
}

__global__ void fused_attention_kernel_v4(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Y,
    int B, int nh, int T, float scale) 
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int t_tile = blockIdx.z;
    
    int t_start = t_tile * TR;
    int tid_x = threadIdx.x; // 0..31
    int tid_y = threadIdx.y; // 0..31
    int tid = tid_y * 32 + tid_x;

    long long head_offset = (long long)(b * nh + h) * T * HS;
    
    const float* Q_ptr = Q + head_offset;
    const float* K_ptr = K + head_offset;
    const float* V_ptr = V + head_offset;
    float* Y_ptr = Y + head_offset;

    __shared__ float Q_sh[TR][HS];
    __shared__ float K_sh[TC][HS];
    __shared__ float V_sh[TC][HS];

    // Registers for 2 rows
    float acc0_0 = 0.0f, acc0_1 = 0.0f, acc0_2 = 0.0f;
    float acc1_0 = 0.0f, acc1_1 = 0.0f, acc1_2 = 0.0f;
    float l0 = 0.0f, l1 = 0.0f;
    float m0 = -1e30f, m1 = -1e30f;

    // Load Q
    // TR*HS/4 = 64*24 = 1536 float4s.
    for(int i = tid; i < 1536; i += 1024) {
        int r = i / 24;
        int c4 = i % 24;
        if (t_start + r < T) {
            float4 val = load_float4(Q_ptr, ((t_start + r) * 24 + c4));
            reinterpret_cast<float4*>(&Q_sh[r][0])[c4] = val;
        } else {
            reinterpret_cast<float4*>(&Q_sh[r][0])[c4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    
    __syncthreads();

    int num_k_tiles = (T + TC - 1) / TC;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int k_start = k_tile * TC;
        
        if (k_start > t_start + TR - 1) break;

        // Load K, V
        // TC*HS/4 = 32*24 = 768 float4s.
        if (tid < 768) {
            int r = tid / 24;
            int c4 = tid % 24;
            if (k_start + r < T) {
                float4 valK = load_float4(K_ptr, ((k_start + r) * 24 + c4));
                reinterpret_cast<float4*>(&K_sh[r][0])[c4] = valK;
                float4 valV = load_float4(V_ptr, ((k_start + r) * 24 + c4));
                reinterpret_cast<float4*>(&V_sh[r][0])[c4] = valV;
            } else {
                reinterpret_cast<float4*>(&K_sh[r][0])[c4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                reinterpret_cast<float4*>(&V_sh[r][0])[c4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        // Compute Row 0
        {
            int r_local = tid_y;
            int q_row_global = t_start + r_local;
            
            if (q_row_global < T) {
                float q0 = Q_sh[r_local][tid_x];
                float q1 = Q_sh[r_local][tid_x + 32];
                float q2 = Q_sh[r_local][tid_x + 64];

                for (int j = 0; j < TC; ++j) {
                    int k_idx_global = k_start + j;
                    if (k_idx_global > q_row_global || k_idx_global >= T) continue;

                    float dot = 0.0f;
                    dot += q0 * K_sh[j][tid_x];
                    dot += q1 * K_sh[j][tid_x + 32];
                    dot += q2 * K_sh[j][tid_x + 64];

                    for (int offset = 16; offset > 0; offset /= 2) {
                        dot += __shfl_xor(dot, offset);
                    }
                    
                    dot *= scale;
                    
                    float m_prev = m0;
                    m0 = fmaxf(m0, dot);
                    
                    float exp_score = expf(dot - m0);
                    float correction = expf(m_prev - m0);
                    
                    l0 = l0 * correction + exp_score;
                    
                    acc0_0 = acc0_0 * correction + exp_score * V_sh[j][tid_x];
                    acc0_1 = acc0_1 * correction + exp_score * V_sh[j][tid_x + 32];
                    acc0_2 = acc0_2 * correction + exp_score * V_sh[j][tid_x + 64];
                }
            }
        }

        // Compute Row 1
        {
            int r_local = tid_y + 32;
            int q_row_global = t_start + r_local;
            
            if (q_row_global < T) {
                float q0 = Q_sh[r_local][tid_x];
                float q1 = Q_sh[r_local][tid_x + 32];
                float q2 = Q_sh[r_local][tid_x + 64];

                for (int j = 0; j < TC; ++j) {
                    int k_idx_global = k_start + j;
                    if (k_idx_global > q_row_global || k_idx_global >= T) continue;

                    float dot = 0.0f;
                    dot += q0 * K_sh[j][tid_x];
                    dot += q1 * K_sh[j][tid_x + 32];
                    dot += q2 * K_sh[j][tid_x + 64];

                    for (int offset = 16; offset > 0; offset /= 2) {
                        dot += __shfl_xor(dot, offset);
                    }
                    
                    dot *= scale;
                    
                    float m_prev = m1;
                    m1 = fmaxf(m1, dot);
                    
                    float exp_score = expf(dot - m1);
                    float correction = expf(m_prev - m1);
                    
                    l1 = l1 * correction + exp_score;
                    
                    acc1_0 = acc1_0 * correction + exp_score * V_sh[j][tid_x];
                    acc1_1 = acc1_1 * correction + exp_score * V_sh[j][tid_x + 32];
                    acc1_2 = acc1_2 * correction + exp_score * V_sh[j][tid_x + 64];
                }
            }
        }
        __syncthreads();
    }

    // Write output Row 0
    {
        int r_local = tid_y;
        int q_row_global = t_start + r_local;
        if (q_row_global < T) {
            float inv_l = 1.0f / l0;
            Y_ptr[q_row_global * HS + tid_x] = acc0_0 * inv_l;
            Y_ptr[q_row_global * HS + tid_x + 32] = acc0_1 * inv_l;
            Y_ptr[q_row_global * HS + tid_x + 64] = acc0_2 * inv_l;
        }
    }
    // Write output Row 1
    {
        int r_local = tid_y + 32;
        int q_row_global = t_start + r_local;
        if (q_row_global < T) {
            float inv_l = 1.0f / l1;
            Y_ptr[q_row_global * HS + tid_x] = acc1_0 * inv_l;
            Y_ptr[q_row_global * HS + tid_x + 32] = acc1_1 * inv_l;
            Y_ptr[q_row_global * HS + tid_x + 64] = acc1_2 * inv_l;
        }
    }
}

torch::Tensor fused_attention_hip(torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale) {
    auto B = q.size(0);
    auto nh = q.size(1);
    auto T = q.size(2);
    
    auto y = torch::empty_like(q);
    
    dim3 grid(B, nh, (T + TR - 1) / TR);
    dim3 block(32, 32); // 1024 threads

    fused_attention_kernel_v4<<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        y.data_ptr<float>(),
        B, nh, T, scale
    );
    
    return y;
}
"""

fused_attn_module = load_inline(
    name="fused_attention_v4",
    cpp_sources=cpp_source,
    functions=["fused_attention_hip"],
    verbose=True,
    extra_cflags=['-O3', '--gpu-max-threads-per-block=1024', '-ffast-math']
)

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()

        scale = 1.0 / math.sqrt(k.size(-1))
        y = fused_attn_module.fused_attention_hip(q, k, v, scale)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

def get_inputs():
    batch_size = 128
    seq_len = 512
    n_embd = 768
    return [torch.rand(batch_size, seq_len, n_embd).cuda()]

def get_init_inputs():
    n_embd = 768
    n_head = 8
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    max_seqlen = 1024
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
