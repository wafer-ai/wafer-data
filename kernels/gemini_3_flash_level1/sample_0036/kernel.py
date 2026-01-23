
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

mse_loss_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = 32; offset > 0; offset /= 2)
        val += __shfl_down(val, offset, 64);
    return val;
}

__device__ __forceinline__ double blockReduceSum(double val) {
    static __shared__ double shared[32];
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / 64) ? shared[lane] : 0.0;

    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__global__ void mse_loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, double* __restrict__ out, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double local_sum = 0.0;
    
    long long n4 = n / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* target4 = reinterpret_cast<const float4*>(targets);
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < n4; i += stride) {
        float4 p = pred4[i];
        float4 t = target4[i];
        float d0 = p.x - t.x;
        float d1 = p.y - t.y;
        float d2 = p.z - t.z;
        float d3 = p.w - t.w;
        local_sum += (double)(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
    }

    for (long long i = n4 * 4 + idx; i < n; i += stride) {
        float diff = predictions[i] - targets[i];
        local_sum += (double)(diff * diff);
    }

    local_sum = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        atomicAdd(out, local_sum);
    }
}

torch::Tensor mse_loss_hip(torch::Tensor predictions, torch::Tensor targets) {
    auto n = predictions.numel();
    auto out = torch::zeros({1}, predictions.options().dtype(torch::kFloat64));
    
    int block_size = 256;
    int num_blocks = 2048; 

    mse_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        out.data_ptr<double>(), 
        (long long)n
    );

    return (out / (double)n).to(predictions.dtype());
}
"""

mse_loss_lib = load_inline(
    name="mse_loss_lib",
    cpp_sources=mse_loss_cpp_source,
    functions=["mse_loss_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_loss_lib = mse_loss_lib

    def forward(self, predictions, targets):
        return self.mse_loss_lib.mse_loss_hip(predictions, targets)
