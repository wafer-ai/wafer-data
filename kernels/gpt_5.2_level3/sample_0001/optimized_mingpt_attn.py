import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Compile HIP code with hipcc
os.environ.setdefault("CXX", "hipcc")

hip_source = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__device__ __forceinline__ float shfl_xor_32(float v, int laneMask) {
    return __shfl_xor(v, laneMask, 32);
}

__device__ __forceinline__ float shfl_32(float v, int srcLane) {
    return __shfl(v, srcLane, 32);
}

__device__ __forceinline__ float warp_reduce_max_32(float v) {
    v = fmaxf(v, shfl_xor_32(v, 16));
    v = fmaxf(v, shfl_xor_32(v, 8));
    v = fmaxf(v, shfl_xor_32(v, 4));
    v = fmaxf(v, shfl_xor_32(v, 2));
    v = fmaxf(v, shfl_xor_32(v, 1));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum_32(float v) {
    v += shfl_xor_32(v, 16);
    v += shfl_xor_32(v, 8);
    v += shfl_xor_32(v, 4);
    v += shfl_xor_32(v, 2);
    v += shfl_xor_32(v, 1);
    return v;
}

// FlashAttention-like forward for FP32 causal attention.
// Specialized for head_dim HS=96 (KernelBench fixed config: n_embd=768, n_head=8).

template<int BLOCK_M, int BLOCK_N, int HS>
__global__ void flash_attn_fwd_kernel_96(
    const float* __restrict__ qkv, // [B, T, 3*C]
    float* __restrict__ out,       // [B, T, C]
    int T, int C, int nhead, float scale)
{
    const int bhead = (int)blockIdx.x;
    const int b = bhead / nhead;
    const int h = bhead - b * nhead;

    const int qb = (int)blockIdx.y;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5; // 0..BLOCK_M-1
    const int lane = tid & 31; // 0..31

    extern __shared__ float smem[];
    float* sQ = smem;                                 // [BLOCK_M*HS]
    float* sK = sQ + (BLOCK_M * HS);                  // [BLOCK_N*HS]
    float* sV = sK + (BLOCK_N * HS);                  // [BLOCK_N*HS]
    float* sO = sV + (BLOCK_N * HS);                  // [BLOCK_M*HS]
    float* sMax = sO + (BLOCK_M * HS);                // [BLOCK_M]
    float* sSum = sMax + BLOCK_M;                     // [BLOCK_M]

    if (tid < BLOCK_M) {
        sMax[tid] = -INFINITY;
        sSum[tid] = 0.0f;
    }

    // Load Q tile + init O
    for (int idx = tid; idx < BLOCK_M * HS; idx += (int)blockDim.x) {
        const int m = idx / HS;
        const int d = idx - m * HS;
        const int qt = qb * BLOCK_M + m;
        float qv = 0.0f;
        if (qt < T) {
            const int64_t base = ((int64_t)b * T + qt) * (int64_t)(3 * C);
            qv = qkv[base + (int64_t)h * HS + d];
        }
        sQ[idx] = qv;
        sO[idx] = 0.0f;
    }
    __syncthreads();

    const int q_t = qb * BLOCK_M + warp;
    const bool valid_row = (q_t < T);

    const int max_key = min(T, (qb + 1) * BLOCK_M);
    const int n_kblocks = (max_key + BLOCK_N - 1) / BLOCK_N;

    for (int kb = 0; kb < n_kblocks; ++kb) {
        const int col_start = kb * BLOCK_N;

        // Load K and V blocks
        for (int idx = tid; idx < BLOCK_N * HS; idx += (int)blockDim.x) {
            const int n = idx / HS;
            const int d = idx - n * HS;
            const int kt = col_start + n;
            float kv = 0.0f;
            float vv = 0.0f;
            if (kt < max_key) {
                const int64_t base = ((int64_t)b * T + kt) * (int64_t)(3 * C);
                kv = qkv[base + (int64_t)C + (int64_t)h * HS + d];
                vv = qkv[base + (int64_t)(2 * C) + (int64_t)h * HS + d];
            }
            sK[idx] = kv;
            sV[idx] = vv;
        }
        __syncthreads();

        if (valid_row) {
            const int key_t = col_start + lane;
            float score = -INFINITY;
            if (key_t < max_key && key_t <= q_t) {
                float acc = 0.0f;
                const float* qrow = sQ + (int64_t)warp * HS;
                const float* krow = sK + (int64_t)lane * HS;
                #pragma unroll
                for (int d = 0; d < HS; ++d) {
                    acc = fmaf(qrow[d], krow[d], acc);
                }
                score = acc * scale;
            }

            const float row_block_max = warp_reduce_max_32(score);

            float old_max = 0.0f;
            float old_sum = 0.0f;
            float new_max = 0.0f;
            float scale_old = 0.0f;
            if (lane == 0) {
                old_max = sMax[warp];
                old_sum = sSum[warp];
                new_max = fmaxf(old_max, row_block_max);
                scale_old = (old_max == -INFINITY) ? 0.0f : expf(old_max - new_max);
            }
            old_sum = shfl_32(old_sum, 0);
            new_max = shfl_32(new_max, 0);
            scale_old = shfl_32(scale_old, 0);

            float exp_score = 0.0f;
            if (score != -INFINITY) {
                exp_score = expf(score - new_max);
            }
            const float row_block_sum = warp_reduce_sum_32(exp_score);

            const float* vrow = sV + (int64_t)lane * HS;

            // Update output accumulator
            #pragma unroll
            for (int d = 0; d < HS; ++d) {
                float contrib = exp_score * vrow[d];
                float numer = warp_reduce_sum_32(contrib);
                if (lane == 0) {
                    float old_o = sO[(int64_t)warp * HS + d];
                    sO[(int64_t)warp * HS + d] = old_o * scale_old + numer;
                }
            }

            if (lane == 0) {
                const float new_sum = old_sum * scale_old + row_block_sum;
                sMax[warp] = new_max;
                sSum[warp] = new_sum;
            }
        }

        __syncthreads();
    }

    // Normalize + write output
    for (int idx = tid; idx < BLOCK_M * HS; idx += (int)blockDim.x) {
        const int m = idx / HS;
        const int d = idx - m * HS;
        const int qt = qb * BLOCK_M + m;
        if (qt < T) {
            const float denom = sSum[m];
            float val = 0.0f;
            if (denom > 0.0f) val = sO[idx] / denom;
            out[((int64_t)b * T + qt) * (int64_t)C + (int64_t)h * HS + d] = val;
        }
    }
}

torch::Tensor flash_attn_fwd(torch::Tensor qkv, int64_t nhead) {
    CHECK_INPUT(qkv);
    TORCH_CHECK(qkv.dim() == 3, "qkv must be [B, T, 3*C]");
    const int64_t B = qkv.size(0);
    const int64_t T = qkv.size(1);
    const int64_t threeC = qkv.size(2);
    TORCH_CHECK(threeC % 3 == 0, "last dim must be 3*C");
    const int64_t C = threeC / 3;
    TORCH_CHECK(C % nhead == 0, "C must be divisible by nhead");
    const int64_t hs = C / nhead;
    TORCH_CHECK(hs == 96, "This optimized kernel is specialized for head_dim=96");

    auto out = torch::empty({B, T, C}, qkv.options());

    const float scale = 1.0f / sqrtf(96.0f);

    constexpr int BLOCK_M = 8;
    constexpr int BLOCK_N = 32;
    dim3 block(256, 1, 1);
    dim3 grid((uint32_t)(B * nhead), (uint32_t)((T + BLOCK_M - 1) / BLOCK_M), 1);

    const size_t shmem_floats = (size_t)(BLOCK_M * 96 + 2 * BLOCK_N * 96 + BLOCK_M * 96 + 2 * BLOCK_M);
    const size_t shmem_bytes = shmem_floats * sizeof(float);

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    hipLaunchKernelGGL((flash_attn_fwd_kernel_96<BLOCK_M, BLOCK_N, 96>), grid, block, shmem_bytes, stream,
        (const float*)qkv.data_ptr<float>(), (float*)out.data_ptr<float>(),
        (int)T, (int)C, (int)nhead, scale);

    return out;
}
'''

flash_attn = load_inline(
    name="flash_attn_mingpt_rocm",
    cpp_sources=hip_source,
    functions=["flash_attn_fwd"],
    extra_cflags=["-O3", "-x", "hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
        )
        self.n_head = n_head
        self.n_embd = n_embd
        self._flash_attn = flash_attn

    def forward(self, x):
        qkv = self.c_attn(x)  # [B, T, 3C]
        y = self._flash_attn.flash_attn_fwd(qkv.contiguous(), self.n_head)  # [B, T, C]
        y = self.resid_dropout(self.c_proj(y))
        return y


# KernelBench hooks
batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
