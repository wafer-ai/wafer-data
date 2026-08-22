import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

source = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float gelu_fast(float x) {
    // x * sigmoid(1.702x)
    float z = 1.702f * x;
    float s = 1.0f / (1.0f + __expf(-z));
    return x * s;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    // full mask assumed
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor(v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset);
    }
    return v;
}

template<int THREADS>
__global__ void gelu_softmax_row_kernel_vec4(const float* __restrict__ inp, float* __restrict__ out, int cols) {
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int base = row * cols;

    int cols4 = cols >> 2;
    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(inp + base);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out + base);

    // Compute GELU once into out
    for (int c4 = tid; c4 < cols4; c4 += THREADS) {
        float4 v = in4[c4];
        v.x = gelu_fast(v.x);
        v.y = gelu_fast(v.y);
        v.z = gelu_fast(v.z);
        v.w = gelu_fast(v.w);
        out4[c4] = v;
    }
    __syncthreads();

    const float* __restrict__ outp = out + base;

    // Max reduction (warp then block)
    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += THREADS) local_max = fmaxf(local_max, outp[c]);

    int lane = tid & 31;
    int warp = tid >> 5;
    float wmax = warp_reduce_max(local_max);

    __shared__ float warp_max[32]; // up to 1024/32 = 32 warps
    if (lane == 0) warp_max[warp] = wmax;
    __syncthreads();

    float row_max;
    if (warp == 0) {
        float v = (tid < (THREADS >> 5)) ? warp_max[lane] : -INFINITY;
        float r = warp_reduce_max(v);
        if (lane == 0) warp_max[0] = r;
    }
    __syncthreads();
    row_max = warp_max[0];

    // Sum reduction
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += THREADS) local_sum += __expf(outp[c] - row_max);
    float wsum = warp_reduce_sum(local_sum);

    __shared__ float warp_sum[32];
    if (lane == 0) warp_sum[warp] = wsum;
    __syncthreads();

    float denom;
    if (warp == 0) {
        float v = (tid < (THREADS >> 5)) ? warp_sum[lane] : 0.0f;
        float r = warp_reduce_sum(v);
        if (lane == 0) warp_sum[0] = r;
    }
    __syncthreads();
    denom = warp_sum[0];
    float inv_denom = 1.0f / denom;

    // Write outputs (vec4)
    for (int c4 = tid; c4 < cols4; c4 += THREADS) {
        float4 v = out4[c4];
        v.x = __expf(v.x - row_max) * inv_denom;
        v.y = __expf(v.y - row_max) * inv_denom;
        v.z = __expf(v.z - row_max) * inv_denom;
        v.w = __expf(v.w - row_max) * inv_denom;
        out4[c4] = v;
    }
}

torch::Tensor gelu_softmax_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, N]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK((x.size(1) % 4) == 0, "N must be multiple of 4");

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    auto out = torch::empty_like(x);

    constexpr int THREADS = 1024;
    dim3 block(THREADS);
    dim3 grid(B);
    hipLaunchKernelGGL((gelu_softmax_row_kernel_vec4<THREADS>), grid, block, 0, 0,
                      (const float*)x.data_ptr<float>(), (float*)out.data_ptr<float>(), N);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_softmax_hip", &gelu_softmax_hip, "Fused GELU(approx)+Softmax (HIP)");
}
'''

_ext = load_inline(
    name='gelu_softmax_ext_v3',
    cpp_sources='',
    cuda_sources=source,
    functions=None,
    extra_cuda_cflags=['-O3'],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._ext = _ext

    def forward(self, x):
        x = self.linear(x)
        return self._ext.gelu_softmax_hip(x)


batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [in_features, out_features]
