
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cross_entropy_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

#define WARP_SIZE 64

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__global__ void cross_entropy_kernel(
    const float4* __restrict__ predictions,
    const long* __restrict__ targets,
    float* __restrict__ total_loss,
    int batch_size,
    int num_classes_over_4) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    int target = targets[row];
    const float4* row_ptr = predictions + row * num_classes_over_4;

    // 1. Find max
    float max_val = -1e38f;
    for (int i = threadIdx.x; i < num_classes_over_4; i += blockDim.x) {
        float4 p4 = row_ptr[i];
        max_val = fmaxf(max_val, fmaxf(fmaxf(p4.x, p4.y), fmaxf(p4.z, p4.w)));
    }

    max_val = warpReduceMax(max_val);

    __shared__ float shared_val[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) shared_val[wid] = max_val;
    __syncthreads();

    if (wid == 0) {
        max_val = (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_val[lane] : -1e38f;
        max_val = warpReduceMax(max_val);
        shared_val[0] = max_val;
    }
    __syncthreads();
    max_val = shared_val[0];

    // 2. Compute sum of exps
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < num_classes_over_4; i += blockDim.x) {
        float4 p4 = row_ptr[i];
        sum_exp += expf(p4.x - max_val);
        sum_exp += expf(p4.y - max_val);
        sum_exp += expf(p4.z - max_val);
        sum_exp += expf(p4.w - max_val);
    }

    sum_exp = warpReduceSum(sum_exp);

    if (lane == 0) shared_val[wid] = sum_exp;
    __syncthreads();

    if (wid == 0) {
        sum_exp = (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_val[lane] : 0.0f;
        sum_exp = warpReduceSum(sum_exp);
        shared_val[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = shared_val[0];

    // 3. Compute final loss for the row and accumulate it
    if (threadIdx.x == 0) {
        const float* row_ptr_f = reinterpret_cast<const float*>(row_ptr);
        float log_sum_exp = max_val + logf(sum_exp);
        float loss = -row_ptr_f[target] + log_sum_exp;
        atomicAdd(total_loss, loss / batch_size);
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto num_classes = predictions.size(1);
    auto total_loss = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int num_blocks = batch_size;

    cross_entropy_kernel<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(predictions.data_ptr<float>()),
        targets.data_ptr<long>(),
        total_loss.data_ptr<float>(),
        batch_size,
        num_classes / 4);

    return total_loss.squeeze();
}
"""

cross_entropy_lib = load_inline(
    name="cross_entropy_lib",
    cpp_sources=cross_entropy_cpp_source,
    functions=["cross_entropy_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy_lib = cross_entropy_lib

    def forward(self, predictions, targets):
        return self.cross_entropy_lib.cross_entropy_hip(predictions, targets)

