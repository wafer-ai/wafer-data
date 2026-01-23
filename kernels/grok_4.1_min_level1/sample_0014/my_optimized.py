import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void compute_loss_per_row(const float* logits, const int64_t* targets, float* loss_per_row, int N, int K) {
    int row = blockIdx.x;
    if (row >= N) return;
    int row_start = row * K;
    int tid = threadIdx.x;
    __shared__ float sdata[256];

    // compute max
    float lmax = -1e30f;
    for (int j = tid; j < K; j += 256) {
        lmax = fmaxf(lmax, logits[row_start + j]);
    }
    sdata[tid] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    // compute sum exp(x - max)
    float lsum = 0.0f;
    for (int j = tid; j < K; j += 256) {
        float val = logits[row_start + j] - row_max;
        lsum += expf(val);
    }
    sdata[tid] = lsum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sumexp = sdata[0];

    if (tid == 0) {
        int64_t tgt = targets[row];
        float logit_tgt = logits[row_start + static_cast<int>(tgt)];
        float loss_i = row_max - logit_tgt + logf(row_sumexp);
        loss_per_row[row] = loss_i;
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor logits, torch::Tensor targets) {
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D");
    int64_t N = logits.size(0);
    int64_t K = logits.size(1);
    TORCH_CHECK(targets.size(0) == N, "batch size mismatch");

    auto options = logits.options();
    auto loss_per_row = torch::empty({N}, options);

    int BS = 256;
    dim3 blocks(static_cast<unsigned int>(N));
    dim3 threads(BS);
    size_t shmem = BS * sizeof(float);

    compute_loss_per_row<<<blocks, threads, shmem>>>(
        logits.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        loss_per_row.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(K)
    );

    hipDeviceSynchronize();

    auto sum_loss = torch::sum(loss_per_row);
    return sum_loss / static_cast<float>(N);
}
"""

cross_entropy = load_inline(
    name="cross_entropy",
    cpp_sources=cpp_source,
    functions=["cross_entropy_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_hip(predictions, targets)

def get_inputs():
    batch_size = 32768
    num_classes = 4096
    input_shape = (num_classes,)
    dim = 1
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []
