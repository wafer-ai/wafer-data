import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Build with ROCm/HIP
os.environ.setdefault("CXX", "hipcc")

# Fused cross entropy (logsumexp + gather) + separate mean-reduction to avoid global atomic contention
cross_entropy_cpp_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down(v, offset, warpSize);
        v = fmaxf(v, other);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, warpSize);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float smax[32];
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    v = warp_reduce_max(v);
    if (lane == 0) smax[warp] = v;
    __syncthreads();

    float out = -INFINITY;
    if (warp == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        out = (lane < num_warps) ? smax[lane] : -INFINITY;
        out = warp_reduce_max(out);
        if (lane == 0) smax[0] = out;
    }
    __syncthreads();
    return smax[0];
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float ssum[32];
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    v = warp_reduce_sum(v);
    if (lane == 0) ssum[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        out = (lane < num_warps) ? ssum[lane] : 0.0f;
        out = warp_reduce_sum(out);
        if (lane == 0) ssum[0] = out;
    }
    __syncthreads();
    return ssum[0];
}

__global__ void cross_entropy_losses_fused_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int B,
    int C)
{
    const int row = (int)blockIdx.x;
    if (row >= B) return;

    const float* row_ptr = logits + (size_t)row * (size_t)C;

    // Vectorized loads (C is divisible by 4: 4096)
    const int C4 = C / 4;
    const float4* row4 = reinterpret_cast<const float4*>(row_ptr);

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < C4; i += blockDim.x) {
        float4 v = row4[i];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }

    float m = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < C4; i += blockDim.x) {
        float4 v = row4[i];
        local_sum += __expf(v.x - m);
        local_sum += __expf(v.y - m);
        local_sum += __expf(v.z - m);
        local_sum += __expf(v.w - m);
    }

    float s = block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        int t = (int)targets[row];
        float x_t = row_ptr[t];
        losses[row] = (m + __logf(s)) - x_t;
    }
}

__global__ void reduce_mean_kernel(const float* __restrict__ in, float* __restrict__ out, int n, float inv_n) {
    float sum = 0.0f;
    // grid-stride loop
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += (int)(gridDim.x * blockDim.x)) {
        sum += in[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(out, sum * inv_n);
    }
}

torch::Tensor cross_entropy_mean_hip(torch::Tensor logits, torch::Tensor targets) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA/HIP tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA/HIP tensor");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [B, C]");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D [B]");
    TORCH_CHECK(logits.size(0) == targets.size(0), "batch size mismatch");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");

    const int B = (int)logits.size(0);
    const int C = (int)logits.size(1);
    TORCH_CHECK((C % 4) == 0, "C must be divisible by 4 for float4 loads");

    auto losses = torch::empty({B}, logits.options().dtype(torch::kFloat32));
    auto out = torch::zeros({}, logits.options().dtype(torch::kFloat32));

    const int threads = 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // 1) compute per-sample loss
    cross_entropy_losses_fused_kernel<<<(unsigned)B, threads, 0, stream>>>(
        (const float*)logits.data_ptr<float>(),
        (const int64_t*)targets.data_ptr<int64_t>(),
        (float*)losses.data_ptr<float>(),
        B, C);

    // 2) reduce to mean (few atomics)
    const float invB = 1.0f / (float)B;
    int blocks = (B + threads - 1) / threads; // 128 for B=32768
    blocks = blocks > 256 ? 256 : blocks;     // cap blocks
    reduce_mean_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const float*)losses.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, invB);

    return out;
}
"""

cross_entropy_ext = load_inline(
    name="cross_entropy_fused_ext_v2",
    cpp_sources=cross_entropy_cpp_source,
    functions=["cross_entropy_mean_hip"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.ext = cross_entropy_ext

    def forward(self, predictions, targets):
        if not predictions.is_cuda:
            return torch.nn.functional.cross_entropy(predictions, targets)
        if predictions.dtype != torch.float32:
            predictions = predictions.float()
        if targets.dtype != torch.int64:
            targets = targets.long()
        return self.ext.cross_entropy_mean_hip(predictions.contiguous(), targets.contiguous())


batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []
