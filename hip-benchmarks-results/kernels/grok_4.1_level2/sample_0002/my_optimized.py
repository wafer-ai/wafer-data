import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void compute_partial_sums_sumsq_kernel(const float* raw, int B, int C, int rpb, double* p_sum, double* p_sumsq) {
  const int chans_per_block = 256;
  int num_chan_blocks = (C + chans_per_block - 1) / chans_per_block;
  int block_id = blockIdx.x;
  int partial_idx = block_id / num_chan_blocks;
  int chan_block_idx = block_id % num_chan_blocks;
  int chan_start = chan_block_idx * chans_per_block;
  int tid = threadIdx.x;
  int ch = chan_start + tid;
  if (ch >= C) return;
  int row_start = partial_idx * rpb;
  int row_end = row_start + rpb;
  if (row_end > B) row_end = B;
  double sumc = 0.0;
  double sumsqc = 0.0;
  for (int row = row_start; row < row_end; row++) {
    float rval = raw[row * C + ch];
    double rval_d = static_cast<double>(rval);
    sumc += rval_d;
    sumsqc += rval_d * rval_d;
  }
  p_sum[partial_idx * C + ch] = sumc;
  p_sumsq[partial_idx * C + ch] = sumsqc;
}

torch::Tensor compute_partials_hip(torch::Tensor raw, int64_t B_, int64_t C_, int64_t rpb_, torch::Tensor p_sum, torch::Tensor p_sumsq) {
  int B = static_cast<int>(B_);
  int C = static_cast<int>(C_);
  int rpb = static_cast<int>(rpb_);
  const int chans_per_block = 256;
  int num_chan_blocks = (C + chans_per_block - 1) / chans_per_block;
  int64_t num_partials_ = (B_ + rpb_ - 1) / rpb_;
  int num_partials = static_cast<int>(num_partials_);
  dim3 block(chans_per_block);
  dim3 grid(num_partials * num_chan_blocks);
  compute_partial_sums_sumsq_kernel<<<grid, block>>>(
    raw.data_ptr<float>(), B, C, rpb,
    p_sum.data_ptr<double>(), p_sumsq.data_ptr<double>());
  return torch::Tensor();
}

__global__ void reduce_partials_kernel(const double* partial, int num_rows, int C, int rpb, double* final_out) {
  const int chans_per_block = 256;
  int num_chan_blocks = (C + chans_per_block - 1) / chans_per_block;
  int block_id = blockIdx.x;
  int partial_idx = block_id / num_chan_blocks;
  int chan_block_idx = block_id % num_chan_blocks;
  int chan_start = chan_block_idx * chans_per_block;
  int tid = threadIdx.x;
  int ch = chan_start + tid;
  if (ch >= C) return;
  int row_start = partial_idx * rpb;
  int row_end = row_start + rpb;
  if (row_end > num_rows) row_end = num_rows;
  double sumc = 0.0;
  for (int row = row_start; row < row_end; row++) {
    sumc += partial[row * C + ch];
  }
  final_out[partial_idx * C + ch] = sumc;
}

torch::Tensor reduce_partials_hip(torch::Tensor partial, int64_t num_rows_, int64_t C_, torch::Tensor final_out) {
  int num_rows = static_cast<int>(num_rows_);
  int C = static_cast<int>(C_);
  int rpb = num_rows;  // loop all rows in each block
  int num_partials2 = 1;
  const int chans_per_block = 256;
  int num_chan_blocks = (C + chans_per_block - 1) / chans_per_block;
  dim3 block(chans_per_block);
  dim3 grid(num_partials2 * num_chan_blocks);
  reduce_partials_kernel<<<grid, block>>>(
    partial.data_ptr<double>(), num_rows, C, rpb,
    final_out.data_ptr<double>());
  return torch::Tensor();
}

__global__ void normalize_kernel(const float* raw, const double* group_means, const double* group_invstd, const float* norm_w, const float* norm_b, int B, int C, int num_groups, int gs, int rpb, float* out) {
  const int chans_per_block = 256;
  int num_chan_blocks = (C + chans_per_block - 1) / chans_per_block;
  int block_id = blockIdx.x;
  int partial_idx = block_id / num_chan_blocks;
  int chan_block_idx = block_id % num_chan_blocks;
  int chan_start = chan_block_idx * chans_per_block;
  int tid = threadIdx.x;
  int ch = chan_start + tid;
  if (ch >= C) return;
  int g = ch / gs;
  double mean_g = group_means[g];
  double invstd_g = group_invstd[g];
  float w = norm_w[ch];
  float nb = norm_b[ch];
  int row_start = partial_idx * rpb;
  int row_end = row_start + rpb;
  if (row_end > B) row_end = B;
  for (int row = row_start; row < row_end; row++) {
    float rval = raw[row * C + ch];
    float centered = rval - static_cast<float>(mean_g);
    float normed = static_cast<float>(centered * static_cast<float>(invstd_g) * w + nb);
    out[row * C + ch] = normed;
  }
}

torch::Tensor normalize_hip(torch::Tensor raw, torch::Tensor group_means, torch::Tensor group_invstd, torch::Tensor norm_w, torch::Tensor norm_b, int64_t B_, int64_t C_, int64_t num_groups_, int64_t gs_, int64_t rpb_, torch::Tensor out) {
  int B = static_cast<int>(B_);
  int C = static_cast<int>(C_);
  int num_groups = static_cast<int>(num_groups_);
  int gs = static_cast<int>(gs_);
  int rpb = static_cast<int>(rpb_);
  const int chans_per_block = 256;
  int num_chan_blocks = (C + chans_per_block - 1) / chans_per_block;
  int64_t num_partials_ = (B_ + rpb_ - 1) / rpb_;
  int num_partials = static_cast<int>(num_partials_);
  dim3 block(chans_per_block);
  dim3 grid(num_partials * num_chan_blocks);
  normalize_kernel<<<grid, block>>>(
    raw.data_ptr<float>(), group_means.data_ptr<double>(), group_invstd.data_ptr<double>(),
    norm_w.data_ptr<float>(), norm_b.data_ptr<float>(),
    B, C, num_groups, gs, rpb,
    out.data_ptr<float>());
  return torch::Tensor();
}
"""

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.out_features = out_features
        self.eps = 1e-5
        self.rpb_stats = 32
        self.rpb_apply = 1024
        self.fused_norm = load_inline(
            name="fused_norm",
            cpp_sources=cpp_source,
            functions=["compute_partials_hip", "reduce_partials_hip", "normalize_hip"],
            verbose=True,
        )

    def forward(self, x):
        x1 = self.matmul(x)
        raw = torch.sigmoid(x1) * x1 + self.bias.unsqueeze(0)
        B, C = raw.shape
        gs = C // self.num_groups
        rpb_stats = self.rpb_stats
        num_partials = (B + rpb_stats - 1) // rpb_stats
        p_sum = torch.empty((num_partials * C), dtype=torch.float64, device=x.device)
        p_sumsq = torch.empty((num_partials * C), dtype=torch.float64, device=x.device)
        self.fused_norm.compute_partials_hip(raw, B, C, rpb_stats, p_sum, p_sumsq)
        final_sum = torch.empty((C), dtype=torch.float64, device=x.device)
        final_sumsq = torch.empty((C), dtype=torch.float64, device=x.device)
        self.fused_norm.reduce_partials_hip(p_sum, num_partials, C, final_sum)
        self.fused_norm.reduce_partials_hip(p_sumsq, num_partials, C, final_sumsq)
        nelem_g = torch.tensor(B * gs, dtype=torch.float64, device=x.device)
        group_means = torch.zeros(self.num_groups, dtype=torch.float64, device=x.device)
        group_vars = torch.zeros(self.num_groups, dtype=torch.float64, device=x.device)
        for g in range(self.num_groups):
            s_ch = g * gs
            e_ch = (g + 1) * gs
            sum_g = torch.sum(final_sum[s_ch:e_ch])
            sumsq_g = torch.sum(final_sumsq[s_ch:e_ch])
            mean_g = sum_g / nelem_g
            var_g = (sumsq_g / nelem_g) - (mean_g * mean_g)
            group_means[g] = mean_g
            group_vars[g] = var_g
        group_invstd = 1.0 / torch.sqrt(group_vars + self.eps).to(torch.float64)
        out = torch.empty_like(raw)
        rpb_apply = self.rpb_apply
        self.fused_norm.normalize_hip(raw, group_means, group_invstd, self.group_norm.weight, self.group_norm.bias, B, C, self.num_groups, gs, rpb_apply, out)
        return out

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
