import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Compile HIP extension
os.environ.setdefault("CXX", "hipcc")

hip_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float gelu_exact(float x) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 0.7071067811865475244f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__global__ void gelu_scale_max_kernel(const float* __restrict__ inp,
                                     float* __restrict__ out,
                                     int ncols,
                                     float scale) {
    // One block per row.
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;

    float maxv = -INFINITY;
    const int row_off = row * ncols;

    // Assume ncols is multiple of 4 for vector loads (true for 512).
    const int idx4 = tid * 4;
    if (idx4 + 3 < ncols) {
        const float4 v = *reinterpret_cast<const float4*>(inp + row_off + idx4);
        float a0 = gelu_exact(v.x) * scale;
        float a1 = gelu_exact(v.y) * scale;
        float a2 = gelu_exact(v.z) * scale;
        float a3 = gelu_exact(v.w) * scale;
        maxv = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
    } else {
        // Tail (shouldn't happen for ncols=512)
        for (int j = idx4; j < ncols; ++j) {
            float a = gelu_exact(inp[row_off + j]) * scale;
            maxv = fmaxf(maxv, a);
        }
    }

    extern __shared__ float sdata[];
    sdata[tid] = maxv;
    __syncthreads();

    // Parallel reduction (max)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = sdata[0];
    }
}

torch::Tensor gelu_scale_max_hip(torch::Tensor input, double scale) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "FP32 only");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");

    auto inp = input.contiguous();
    const int64_t B = inp.size(0);
    const int64_t N = inp.size(1);

    auto out = torch::empty({B}, inp.options());

    // Use 128 threads so each thread processes 4 elements for N=512.
    const int threads = 128;
    TORCH_CHECK((threads * 4) >= N, "threads*4 must cover N");

    const dim3 blocks((uint32_t)B);
    const size_t shmem = threads * sizeof(float);

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream().stream();
    hipLaunchKernelGGL(gelu_scale_max_kernel, blocks, dim3(threads), shmem, stream,
                       (const float*)inp.data_ptr<float>(),
                       (float*)out.data_ptr<float>(),
                       (int)N,
                       (float)scale);

    return out;
}
"""

# Build extension once.
_gelu_scale_max = load_inline(
    name="gelu_scale_max_ext_98",
    cpp_sources=hip_src,
    functions=["gelu_scale_max_hip"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized model:

    Exploits algebraic equivalence:
      Linear(8192->8192) + AvgPool1d(k=16, stride=16)
    == Linear(8192->512) with weights/bias averaged over groups of 16 output rows.

    Then uses a fused HIP kernel for GELU + scale + max-reduction.
    """

    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features, bias=True)
        self.pool_kernel_size = int(pool_kernel_size)
        self.scale_factor = float(scale_factor)

        assert out_features % self.pool_kernel_size == 0
        self.pooled_features = out_features // self.pool_kernel_size

        # Cached aggregated weights/bias to avoid building them every forward.
        self.register_buffer("_w_agg", torch.empty(0), persistent=False)
        self.register_buffer("_b_agg", torch.empty(0), persistent=False)
        self._agg_ready = False

        self._ext = _gelu_scale_max

    def _rebuild_agg(self):
        k = self.pool_kernel_size
        w = self.matmul.weight  # (out_features, in_features)
        b = self.matmul.bias    # (out_features,)

        # AvgPool1d on length dimension corresponds to averaging over groups of k output channels.
        w_agg = w.view(self.pooled_features, k, w.size(1)).mean(dim=1)
        b_agg = b.view(self.pooled_features, k).mean(dim=1)

        # Keep contiguous for GEMM and kernel reads.
        self._w_agg = w_agg.contiguous()
        self._b_agg = b_agg.contiguous()
        self._agg_ready = True

    def load_state_dict(self, state_dict, strict: bool = True):
        out = super().load_state_dict(state_dict, strict=strict)
        self._agg_ready = False
        return out

    def forward(self, x):
        if (not self._agg_ready) or (self._w_agg.device != x.device):
            self._rebuild_agg()

        # Reduced GEMM: (B, in_features) x (pooled_features, in_features)^T
        y = F.linear(x, self._w_agg, self._b_agg)
        return self._ext.gelu_scale_max_hip(y, self.scale_factor)
