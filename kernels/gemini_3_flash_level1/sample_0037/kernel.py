
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cross_entropy_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <limits>

struct Result {
    float max_val;
    float sum_exp;
};

__device__ __forceinline__ Result merge(Result a, Result b) {
    if (a.max_val > b.max_val) {
        return {a.max_val, a.sum_exp + b.sum_exp * expf(b.max_val - a.max_val)};
    } else if (b.max_val > a.max_val) {
        return {b.max_val, b.sum_exp + a.sum_exp * expf(a.max_val - b.max_val)};
    } else {
        return {a.max_val, a.sum_exp + b.sum_exp};
    }
}

__global__ void cross_entropy_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ row_losses,
    int batch_size,
    int num_classes) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    int target_idx = targets[row];
    const float* row_ptr = predictions + row * num_classes;

    Result res = {-std::numeric_limits<float>::infinity(), 0.0f};

    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
    int num_classes4 = num_classes / 4;

    for (int i = threadIdx.x; i < num_classes4; i += blockDim.x) {
        float4 vals = row_ptr4[i];
        float v[4] = {vals.x, vals.y, vals.z, vals.w};
        for (int j = 0; j < 4; ++j) {
            float x = v[j];
            if (x > res.max_val) {
                res.sum_exp = res.sum_exp * expf(res.max_val - x) + 1.0f;
                res.max_val = x;
            } else {
                res.sum_exp += expf(x - res.max_val);
            }
        }
    }
    
    for (int i = num_classes4 * 4 + threadIdx.x; i < num_classes; i += blockDim.x) {
        float x = row_ptr[i];
        if (x > res.max_val) {
            res.sum_exp = res.sum_exp * expf(res.max_val - x) + 1.0f;
            res.max_val = x;
        } else {
            res.sum_exp += expf(x - res.max_val);
        }
    }

    // Warp-level reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor(res.max_val, offset);
        float other_sum = __shfl_xor(res.sum_exp, offset);
        res = merge(res, {other_max, other_sum});
    }

    // Block-level reduction
    __shared__ float shared_max[16];
    __shared__ float shared_sum[16];
    
    int warp_id = threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;

    if (lane_id == 0) {
        shared_max[warp_id] = res.max_val;
        shared_sum[warp_id] = res.sum_exp;
    }
    __syncthreads();

    if (threadIdx.x < 64) {
        int num_warps = blockDim.x / 64;
        float m = (threadIdx.x < num_warps) ? shared_max[threadIdx.x] : -std::numeric_limits<float>::infinity();
        float s = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
        
        for (int offset = 32; offset > 0; offset >>= 1) {
            float other_max = __shfl_xor(m, offset);
            float other_sum = __shfl_xor(s, offset);
            Result res_m = merge({m, s}, {other_max, other_sum});
            m = res_m.max_val;
            s = res_m.sum_exp;
        }
        
        if (threadIdx.x == 0) {
            float log_sum_exp = logf(s) + m;
            row_losses[row] = log_sum_exp - row_ptr[target_idx];
        }
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    auto row_losses = torch::empty({batch_size}, predictions.options());

    const int block_size = 512;
    const int num_blocks = batch_size;

    cross_entropy_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        row_losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    return row_losses.mean();
}
"""

cross_entropy_lib = load_inline(
    name="cross_entropy_lib_v4",
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
