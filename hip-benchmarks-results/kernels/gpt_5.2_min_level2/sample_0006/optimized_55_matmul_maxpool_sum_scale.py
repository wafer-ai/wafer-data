import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

cpp_source = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // AMD wavefront is 64, but HIP's __shfl_down works with warpSize
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    return v;
}

__global__ void maxpool2_sum_scale_kernel_opt(const float* __restrict__ x, float* __restrict__ out,
                                             int64_t N, float scale) {
    int b = (int)blockIdx.x;
    const float* row = x + (int64_t)b * N;
    int64_t pairs = N >> 1;

    float acc = 0.0f;

    // Use vectorized loads when possible
    // Each pooled element corresponds to 2 floats; load float2 and take max
    for (int64_t i = (int64_t)threadIdx.x; i < pairs; i += (int64_t)blockDim.x) {
        const float2 v = reinterpret_cast<const float2*>(row)[i];
        acc += (v.x > v.y ? v.x : v.y);
    }

    // Warp-level reduction
    acc = warp_reduce_sum(acc);

    __shared__ float warp_sums[32]; // up to 1024 threads -> 32 warps (assuming warpSize>=32)
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    // Final reduction by first warp
    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (threadIdx.x < (blockDim.x / warpSize)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) out[b] = block_sum * scale;
    }
}

torch::Tensor maxpool2_sum_scale_hip(torch::Tensor x, double scale_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B,N]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto B = (int64_t)x.size(0);
    auto N = (int64_t)x.size(1);
    TORCH_CHECK((N % 2) == 0, "N must be even for kernel_size=2 pooling");
    TORCH_CHECK((N % 2) == 0, "N must be divisible by 2");

    auto out = torch::empty({B}, x.options());

    // 512 threads often works well for reductions and is within limits.
    const int threads = 512;
    dim3 blocks((unsigned)B);
    auto stream = at::cuda::getDefaultCUDAStream();

    hipLaunchKernelGGL(maxpool2_sum_scale_kernel_opt, blocks, dim3(threads), 0, stream,
                      (const float*)x.data_ptr<float>(), (float*)out.data_ptr<float>(),
                      N, (float)scale_factor);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool2_sum_scale_hip", &maxpool2_sum_scale_hip, "Fused maxpool2+sum+scale (HIP, optimized)");
}
'''

ext = load_inline(
    name="fused_pool_sum_scale_ext",
    cpp_sources=cpp_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        assert kernel_size == 2, "This optimized kernel assumes kernel_size=2"
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        x = self.matmul(x)
        return ext.maxpool2_sum_scale_hip(x, self.scale_factor)


batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
