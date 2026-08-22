import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_add_ln_kernel(const float* attn, const float* res, const float* gamma, const float* beta, float* out, int64_t M, int64_t D, float eps) {
    int tid = threadIdx.x;
    int64_t sid = blockIdx.x;
    if (sid >= M) return;
    if (tid >= (int)D) return;
    int64_t offset = sid * D + tid;
    float val = attn[offset] + res[offset];
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    s_sum[tid] = val;
    __syncthreads();
    for (int s = (int)D / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)D;
    __syncthreads();
    float delta = val - mean;
    s_sum[tid] = delta * delta;
    __syncthreads();
    for (int s = (int)D / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float variance = s_sum[0] / (float)D;
    float invstd = 1.0f / sqrtf(variance + eps);
    out[offset] = gamma[tid] * (delta * invstd) + beta[tid];
}

torch::Tensor fused_add_ln_hip(torch::Tensor attn_output, torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto out = torch::empty_like(attn_output);
    int64_t seq_len = attn_output.size(0);
    int64_t N = attn_output.size(1);
    int64_t D = attn_output.size(2);
    int64_t num_samples = seq_len * N;
    dim3 blocks((unsigned int)num_samples);
    dim3 threads((unsigned int)D);
    size_t shared_mem_bytes = (size_t)D * sizeof(float);
    fused_add_ln_kernel<<<blocks, threads, shared_mem_bytes>>>(
        attn_output.data_ptr<float>(),
        residual.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        num_samples,
        D,
        eps
    );
    return out;
}
"""

fused_module = load_inline(
    name="fused_add_ln",
    cpp_sources=cpp_source,
    functions=["fused_add_ln_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.fused_add_ln = fused_module

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        orig_x = x.view(B, C, seq_len).permute(2, 0, 1).contiguous()
        attn_output, _ = self.attn(orig_x, orig_x, orig_x)
        res_norm = self.fused_add_ln.fused_add_ln_hip(
            attn_output, orig_x, self.norm.weight, self.norm.bias, float(self.norm.eps)
        )
        out = res_norm.permute(1, 2, 0).view(B, C, H, W)
        return out

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width)]

def get_init_inputs():
    return [embed_dim, num_heads]
