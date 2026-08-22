import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

cpp_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset, 32));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 32);
    }
    return val;
}

// Block reduce and broadcast to all threads
__device__ __forceinline__ float block_allreduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    float out = -INFINITY;
    if (wid == 0) {
        out = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -INFINITY;
        out = warp_reduce_max(out);
        if (lane == 0) shared[0] = out;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_allreduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    float out = 0.0f;
    if (wid == 0) {
        out = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
        if (lane == 0) shared[0] = out;
    }
    __syncthreads();
    return shared[0];
}

__global__ void xent_per_sample_kernel(const float* __restrict__ logits,
                                      const int64_t* __restrict__ targets,
                                      float* __restrict__ losses,
                                      int B, int C) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    const float* row = logits + ((int64_t)b) * (int64_t)C;

    float tmax = -INFINITY;
    for (int j = (int)threadIdx.x; j < C; j += (int)blockDim.x) {
        tmax = fmaxf(tmax, row[j]);
    }
    float rmax = block_allreduce_max(tmax);

    float tsum = 0.0f;
    for (int j = (int)threadIdx.x; j < C; j += (int)blockDim.x) {
        tsum += __expf(row[j] - rmax);
    }
    float rsum = block_allreduce_sum(tsum);

    if (threadIdx.x == 0) {
        int64_t t = targets[b];
        if (t < 0) t = 0;
        if (t >= C) t = (int64_t)C - 1;
        float lse = logf(rsum + 1e-20f) + rmax;
        losses[b] = lse - row[t];
    }
}

__global__ void reduce_sum_kernel(const float* __restrict__ losses,
                                 float* __restrict__ out,
                                 int N) {
    float sum = 0.0f;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    for (int i = idx; i < N; i += stride) sum += losses[i];
    sum = block_allreduce_sum(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

torch::Tensor cross_entropy_mean_hip(torch::Tensor logits, torch::Tensor targets) {
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA/HIP tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA/HIP tensor");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [B, C]");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D [B]");

    const int B = (int)logits.size(0);
    const int C = (int)logits.size(1);
    TORCH_CHECK((int)targets.size(0) == B, "targets size mismatch");

    auto losses = torch::empty({B}, torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
    auto out = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));

    hipStream_t stream = 0;

    const int threads = 256;
    hipLaunchKernelGGL(xent_per_sample_kernel, dim3(B), dim3(threads), 0, stream,
                       logits.data_ptr<float>(), targets.data_ptr<int64_t>(), losses.data_ptr<float>(), B, C);

    const int threads2 = 256;
    int blocks2 = (B + threads2 - 1) / threads2;
    if (blocks2 > 256) blocks2 = 256;
    hipLaunchKernelGGL(reduce_sum_kernel, dim3(blocks2), dim3(threads2), 0, stream,
                       losses.data_ptr<float>(), out.data_ptr<float>(), B);

    return out / (float)B;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_entropy_mean_hip", &cross_entropy_mean_hip, "CrossEntropyLoss mean (HIP)");
}
"""

xent_ext = load_inline(
    name="xent_rocm_ext",
    cpp_sources=cpp_source,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent = xent_ext

    def forward(self, predictions, targets):
        if not predictions.is_contiguous():
            predictions = predictions.contiguous()
        if not targets.is_contiguous():
            targets = targets.contiguous()
        return self.xent.cross_entropy_mean_hip(predictions, targets)


batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape, device="cuda", dtype=torch.float32),
            torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)]

def get_init_inputs():
    return []
