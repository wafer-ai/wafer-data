import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_addnorm_cpp = """
#include <hip/hip_runtime.h>

__global__ void compute_sums_kernel(const float *u, const float *r, float *means, float *vars, int num_rows, int D) {
  int row = blockIdx.x;
  if (row >= num_rows) return;
  extern __shared__ float shared[];
  float *s_sum = shared;
  float *s_sumsq = shared + blockDim.x;
  float priv_sum = 0.0f;
  float priv_sumsq = 0.0f;
  const float *u_row = u + row * D;
  const float *r_row = r + row * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float val = u_row[i] + r_row[i];
    priv_sum += val;
    priv_sumsq += val * val;
  }
  s_sum[threadIdx.x] = priv_sum;
  s_sumsq[threadIdx.x] = priv_sumsq;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
      s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float mean = s_sum[0] / D;
    float mean_sq = mean * mean;
    float var = (s_sumsq[0] / D) - mean_sq;
    means[row] = mean;
    vars[row] = var;
  }
}

__global__ void normalize_kernel(const float *u, const float *r, const float *means, const float *vars, const float *gamma, const float *beta, float *out, int num_rows, int D, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_rows * D) return;
  int row = idx / D;
  int d = idx % D;
  float mean = means[row];
  float var = vars[row] + eps;
  float inv_std = 1.0f / sqrtf(var);
  float val = u[idx] + r[idx];
  float normed = (val - mean) * inv_std;
  out[idx] = normed * gamma[d] + beta[d];
}

torch::Tensor fused_addnorm_hip(torch::Tensor update, torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta) {
  auto seq_len = update.size(0);
  auto batch = update.size(1);
  auto dim = update.size(2);
  int num_rows = static_cast<int>(seq_len * batch);
  auto out = torch::empty_like(update);
  auto options = torch::TensorOptions().dtype(torch::kFloat).device(update.device());
  auto means = torch::zeros({num_rows}, options);
  auto mvars = torch::zeros({num_rows}, options);
  int D_int = static_cast<int>(dim);
  int threads_red = 128;
  dim3 block_red(threads_red);
  dim3 grid_red(num_rows);
  compute_sums_kernel<<<grid_red, block_red, 2 * threads_red * sizeof(float)>>>(update.data_ptr<float>(), residual.data_ptr<float>(), means.data_ptr<float>(), mvars.data_ptr<float>(), num_rows, D_int);
  dim3 block_norm(256);
  dim3 grid_norm((num_rows * D_int + 255) / 256);
  float eps = 1e-5f;
  normalize_kernel<<<grid_norm, block_norm>>>(update.data_ptr<float>(), residual.data_ptr<float>(), means.data_ptr<float>(), mvars.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), out.data_ptr<float>(), num_rows, D_int, eps);
  return out;
}
"""

fused_addnorm = load_inline(
    name="fused_addnorm",
    cpp_sources=fused_addnorm_cpp,
    functions=["fused_addnorm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.fused_addnorm = fused_addnorm

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1).contiguous()
        attn_output, _ = self.attn(x, x, x)
        x = self.fused_addnorm.fused_addnorm_hip(attn_output, x, self.norm.weight, self.norm.bias)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x

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
