import os
os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

cpp_source = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cfloat>
#include <cmath>

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int MAX_HS = 96;
constexpr float NEG_INF = -1e38f;

__global__ void flash_fwd_kernel(const float *Q, const float *K, const float *V, float *O,
                                 int64_t B, int64_t nh, int64_t T, int64_t hs, float scale,
                                 int64_t stride_b, int64_t stride_h, int64_t stride_t) {
  const int64_t num_q_tiles = (T + BLOCK_M - 1) / BLOCK_M;
  int64_t bid = blockIdx.x;
  int64_t bh_idx = bid / num_q_tiles;
  int64_t q_tile_idx = bid % num_q_tiles;
  int64_t b = bh_idx / nh;
  int64_t h = bh_idx % nh;
  int64_t q_start = q_tile_idx * BLOCK_M;
  if (q_start >= T) return;

  extern __shared__ float smem[];
  const int off_q = 0;
  const int off_k = off_q + BLOCK_M * MAX_HS;
  const int off_v = off_k + BLOCK_N * MAX_HS;
  const int off_P = off_v + BLOCK_N * MAX_HS;
  const int off_s = off_P + BLOCK_M * MAX_HS;
  const int off_rm = off_s + BLOCK_M * BLOCK_N;
  const int off_rl = off_rm + BLOCK_M;
  const int off_tm = off_rl + BLOCK_M;
  const int off_tl = off_tm + BLOCK_M;

  float *q_sh = smem + off_q;
  float *k_sh = smem + off_k;
  float *v_sh = smem + off_v;
  float *P = smem + off_P;
  float *s_sh = smem + off_s;
  float *running_m = smem + off_rm;
  float *running_l = smem + off_rl;
  float *tile_m = smem + off_tm;
  float *tile_l = smem + off_tl;

  // load q tile
  for (int i = threadIdx.x; i < BLOCK_M * hs; i += blockDim.x) {
    int row = i / hs;
    int col = i % hs;
    int ti = q_start + row;
    q_sh[row * MAX_HS + col] = (ti < T) ? Q[b * stride_b + h * stride_h + ti * stride_t + col] : 0.0f;
  }
  __syncthreads();

  // init running stats
  if (threadIdx.x < BLOCK_M) {
    int sr = threadIdx.x;
    running_m[sr] = NEG_INF;
    running_l[sr] = 0.0f;
#pragma unroll
    for (int d = 0; d < hs; d++) {
      P[sr * MAX_HS + d] = 0.0f;
    }
  }
  __syncthreads();

  // kv tiles loop
  for (int kv_tile = 0; kv_tile <= (int)q_tile_idx; ++kv_tile) {
    int kv_start = kv_tile * BLOCK_N;
    // load k
    for (int i = threadIdx.x; i < BLOCK_N * hs; i += blockDim.x) {
      int row = i / hs;
      int col = i % hs;
      int ti = kv_start + row;
      k_sh[row * MAX_HS + col] = (ti < T) ? K[b * stride_b + h * stride_h + ti * stride_t + col] : 0.0f;
    }
    __syncthreads();
    // load v
    for (int i = threadIdx.x; i < BLOCK_N * hs; i += blockDim.x) {
      int row = i / hs;
      int col = i % hs;
      int ti = kv_start + row;
      v_sh[row * MAX_HS + col] = (ti < T) ? V[b * stride_b + h * stride_h + ti * stride_t + col] : 0.0f;
    }
    __syncthreads();
    // compute s_sh M x N
    for (int i_s = threadIdx.x; i_s < BLOCK_M * BLOCK_N; i_s += blockDim.x) {
      int sr = i_s / BLOCK_N;
      int sc = i_s % BLOCK_N;
      if (sr < BLOCK_M) {
        int qi = q_start + sr;
        int kj = kv_start + sc;
        float dot = 0.0f;
        const float4 *q4 = reinterpret_cast<const float4 *>(q_sh + sr * MAX_HS);
        const float4 *k4 = reinterpret_cast<const float4 *>(k_sh + sc * MAX_HS);
#pragma unroll
        for (int vec = 0; vec < hs / 4; vec++) {
          float4 prod = q4[vec] * k4[vec];
          dot += prod.x + prod.y + prod.z + prod.w;
        }
        float s_val = dot * scale;
        if (qi >= T || kj >= T || kj > qi) {
          s_val = NEG_INF;
        }
        s_sh[sr * BLOCK_N + sc] = s_val;
      }
    }
    __syncthreads();
    // compute tile_m and tile_l
    if (threadIdx.x < BLOCK_M) {
      int sr = threadIdx.x;
      float lmax = s_sh[sr * BLOCK_N + 0];
#pragma unroll
      for (int sc = 1; sc < BLOCK_N; sc++) {
        lmax = fmaxf(lmax, s_sh[sr * BLOCK_N + sc]);
      }
      tile_m[sr] = lmax;
      float lsum = 0.0f;
#pragma unroll
      for (int sc = 0; sc < BLOCK_N; sc++) {
        lsum += expf(s_sh[sr * BLOCK_N + sc] - lmax);
      }
      tile_l[sr] = lsum;
    }
    __syncthreads();
    // update
    if (threadIdx.x < BLOCK_M) {
      int sr = threadIdx.x;
      float tm = tile_m[sr];
      float tl = tile_l[sr];
      float m_new = fmaxf(running_m[sr], tm);
      float alpha = expf(running_m[sr] - m_new);
      float beta = expf(tm - m_new);
      running_l[sr] = running_l[sr] * alpha + tl * beta;
#pragma unroll 2
      for (int d = 0; d < hs; d++) {
        float tile_pd = 0.0f;
#pragma unroll
        for (int sc = 0; sc < BLOCK_N; sc++) {
          float e = expf(s_sh[sr * BLOCK_N + sc] - tm);
          tile_pd += e * v_sh[sc * MAX_HS + d];
        }
        P[sr * MAX_HS + d] = P[sr * MAX_HS + d] * alpha + tile_pd * beta;
      }
      running_m[sr] = m_new;
    }
    __syncthreads();
  }

  // write out
  if (threadIdx.x < BLOCK_M) {
    int sr = threadIdx.x;
    int ti = q_start + sr;
    if (ti < T) {
      float il = 1.0f / running_l[sr];
#pragma unroll 2
      for (int d = 0; d < hs; d++) {
        O[b * stride_b + h * stride_h + ti * stride_t + d] = P[sr * MAX_HS + d] * il;
      }
    }
  }
}

torch::Tensor causal_flash_attn_hip(torch::Tensor q_, torch::Tensor k_, torch::Tensor v_, float scale) {
  torch::Tensor q = q_.contiguous();
  torch::Tensor k = k_.contiguous();
  torch::Tensor v = v_.contiguous();
  torch::Tensor out = torch::empty_like(q);
  int64_t B = q.size(0);
  int64_t nh = q.size(1);
  int64_t T = q.size(2);
  int64_t hs = q.size(3);
  TORCH_CHECK(q.scalar_type() == torch::kFloat, "Must be FP32");
  int64_t stride_b = nh * T * hs;
  int64_t stride_h = T * hs;
  int64_t stride_t = hs;
  int64_t num_q_tiles = (T + BLOCK_M - 1) / BLOCK_M;
  dim3 grid(B * nh * num_q_tiles);
  dim3 block(64);
  size_t shmem_bytes = (BLOCK_M * MAX_HS * 4LL + BLOCK_M * BLOCK_N + BLOCK_M * 4LL * 4) * sizeof(float);
  flash_fwd_kernel<<<grid, block, shmem_bytes>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
                                                 B, nh, T, hs, scale, stride_b, stride_h, stride_t);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err));
  }
  hipDeviceSynchronize();
  return out;
}
'''

flash_module = load_inline(
    name="flash_attn",
    cpp_sources=cpp_source,
    functions=["causal_flash_attn_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.flash = flash_module

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()
        scale = 1.0 / math.sqrt(k.size(-1))
        y = self.flash.causal_flash_attn_hip(q, k, v, scale)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
