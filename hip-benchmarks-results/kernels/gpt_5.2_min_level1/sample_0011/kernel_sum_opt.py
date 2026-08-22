import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

hip_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

// Tuned for MI300-class GPUs: coalesced loads along D2, 2D block to split D1 reduction.
__global__ void sum_dim1_coalesced_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         int B, int D1, int D2) {
    constexpr int TX = 128;
    constexpr int TY = 4;

    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    int tile_k = (int)blockIdx.x;
    int b = (int)blockIdx.y;
    int k = tile_k * TX + tx;

    float acc = 0.0f;
    if (k < D2) {
        const float* base = x + ((long)b * D1 * D2 + k);
        // unroll by 2 to reduce loop overhead
        for (int i = ty; i < D1; i += TY * 2) {
            acc += base[(long)i * D2];
            int i2 = i + TY;
            if (i2 < D1) acc += base[(long)i2 * D2];
        }
    }

    __shared__ float partial[TY][TX];
    partial[ty][tx] = acc;
    __syncthreads();

    if (ty == 0 && k < D2) {
        float sum = partial[0][tx];
        #pragma unroll
        for (int r = 1; r < TY; r++) sum += partial[r][tx];
        out[(long)b * D2 + k] = sum;
    }
}

torch::Tensor sum_dim1_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor [B, D1, D2]");

    auto x_contig = x.contiguous();
    int64_t B = x_contig.size(0);
    int64_t D1 = x_contig.size(1);
    int64_t D2 = x_contig.size(2);

    auto out = torch::empty({B, 1, D2}, x_contig.options());

    constexpr int TX = 128;
    constexpr int TY = 4;
    dim3 block(TX, TY, 1);
    dim3 grid((unsigned int)((D2 + TX - 1) / TX), (unsigned int)B, 1);

    hipStream_t stream = at::hip::getDefaultHIPStream();
    hipLaunchKernelGGL(sum_dim1_coalesced_kernel, grid, block, 0, stream,
                       (const float*)x_contig.data_ptr<float>(),
                       (float*)out.data_ptr<float>(),
                       (int)B, (int)D1, (int)D2);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_hip", &sum_dim1_hip, "sum over dim=1 keepdim (HIP, coalesced)");
}
"""

sum_dim1_ext = load_inline(
    name="sum_dim1_ext_v4",
    cpp_sources="",
    cuda_sources=hip_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1 and x.is_cuda and x.dtype == torch.float32 and x.dim() == 3:
            return sum_dim1_ext.sum_dim1_hip(x)
        return torch.sum(x, dim=self.dim, keepdim=True)


def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    return [x]


def get_init_inputs():
    return [1]
