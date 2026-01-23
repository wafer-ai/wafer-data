import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__global__ void cel_loss_kernel(const float* __restrict__ logits,
                                const int64_t* __restrict__ targets,
                                float* __restrict__ losses,
                                int N, int C) {
  int bid = blockIdx.x;
  if (bid >= N) return;
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int ts = blockDim.x;
  const float* row = logits + static_cast<size_t>(bid) * static_cast<size_t>(C);
  const float NEG_INF = -1e38f;

  // compute max
  float maxv = NEG_INF;
  for (int pos = tid; pos < C; pos += ts) {
    maxv = fmaxf(maxv, row[pos]);
  }
  sdata[tid] = maxv;
  __syncthreads();
  for (int s = ts / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  float max_logit = sdata[0];

  // compute sum_exp
  float sev = 0.0f;
  for (int pos = tid; pos < C; pos += ts) {
    sev += expf(row[pos] - max_logit);
  }
  sdata[tid] = sev;
  __syncthreads();
  for (int s = ts / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  float sum_exp = sdata[0];
  float logsumexp = max_logit + logf(sum_exp);

  int tgt = static_cast<int>(targets[bid]);
  float tgt_val = row[tgt];
  float loss = - (tgt_val - logsumexp);
  if (tid == 0) {
    losses[bid] = loss;
  }
}

torch::Tensor cross_entropy_hip(torch::Tensor logits, torch::Tensor targets) {
  int64_t N = logits.size(0);
  int64_t C = logits.size(1);
  auto options = logits.options();
  auto losses = torch::empty({N}, options);
  const int ts = 128;
  dim3 blocks(static_cast<uint32_t>(N));
  dim3 threads(ts);
  size_t shmem = ts * sizeof(float);
  cel_loss_kernel<<<blocks, threads, shmem>>>(
    logits.data_ptr<float>(),
    targets.data_ptr<int64_t>(),
    losses.data_ptr<float>(),
    static_cast<int>(N),
    static_cast<int>(C)
  );
  auto sum_loss = torch::sum(losses);
  return sum_loss / static_cast<float>(N);
}
"""

cross_entropy_impl = load_inline(
    name="cross_entropy_impl",
    cpp_sources=cross_entropy_cpp,
    functions=["cross_entropy_hip"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy_impl

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_hip(predictions, targets)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    preds = torch.rand(batch_size, *input_shape, device='cuda')
    tgts = torch.randint(0, num_classes, (batch_size,), device='cuda')
    return [preds, tgts]

def get_init_inputs():
    return []
