
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cross_entropy_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset, 64));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset, 64);
    }
    return val;
}

__global__ void cross_entropy_forward_kernel_v4(
    const float4* predictions,
    const long* targets,
    float* losses,
    int batch_size,
    int num_classes_v4) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    int target = targets[row];
    const float4* row_preds_v4 = predictions + row * num_classes_v4;

    // Step 1: Find max for numerical stability
    float max_val = -1e38f;
    for (int i = threadIdx.x; i < num_classes_v4; i += blockDim.x) {
        float4 p = row_preds_v4[i];
        max_val = fmaxf(max_val, fmaxf(p.x, fmaxf(p.y, fmaxf(p.z, p.w))));
    }

    max_val = warpReduceMax(max_val);

    __shared__ float final_max;
    __shared__ float final_sum;
    __shared__ float temp_storage[16]; // 1024 / 64 = 16
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;

    if (lane == 0) temp_storage[wid] = max_val;
    __syncthreads();

    if (wid == 0) {
        float val = (threadIdx.x < (blockDim.x / 64)) ? temp_storage[threadIdx.x] : -1e38f;
        val = warpReduceMax(val);
        if (threadIdx.x == 0) final_max = val;
    }
    __syncthreads();
    max_val = final_max;

    // Step 2: Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < num_classes_v4; i += blockDim.x) {
        float4 p = row_preds_v4[i];
        sum_exp += expf(p.x - max_val);
        sum_exp += expf(p.y - max_val);
        sum_exp += expf(p.z - max_val);
        sum_exp += expf(p.w - max_val);
    }

    sum_exp = warpReduceSum(sum_exp);

    if (lane == 0) temp_storage[wid] = sum_exp;
    __syncthreads();

    if (wid == 0) {
        float val = (threadIdx.x < (blockDim.x / 64)) ? temp_storage[threadIdx.x] : 0.0f;
        val = warpReduceSum(val);
        if (threadIdx.x == 0) final_sum = val;
    }
    __syncthreads();
    sum_exp = final_sum;

    // Step 3: Compute final loss for the row
    if (threadIdx.x == 0) {
        const float* row_preds = reinterpret_cast<const float*>(row_preds_v4);
        float log_sum_exp = logf(sum_exp) + max_val;
        losses[row] = log_sum_exp - row_preds[target];
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    int num_classes_v4 = num_classes / 4;

    auto losses = torch::empty({batch_size}, predictions.options());

    // Use larger block size
    int threads_per_block = 512;

    cross_entropy_forward_kernel_v4<<<batch_size, threads_per_block>>>(
        reinterpret_cast<const float4*>(predictions.data_ptr<float>()),
        targets.data_ptr<long>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes_v4
    );

    return losses.mean();
}
"""

cross_entropy_lib = load_inline(
    name="cross_entropy_lib",
    cpp_sources=cross_entropy_kernel_source,
    functions=["cross_entropy_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy_lib = cross_entropy_lib

    def forward(self, predictions, targets):
        return self.cross_entropy_lib.cross_entropy_hip(predictions, targets)
