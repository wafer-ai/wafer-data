import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

# Key optimization (algorithmic but exact):
# AvgPool over the Linear output (kernel=16,stride=16) is linear, so we can pre-collapse
# the Linear weights/bias into a smaller Linear with out_features/16 outputs.
# Then we only need GELU+scale+max over 512 values per batch.

hip_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float u = k0 * (x + k1 * x3);
    float t = tanhf(u);
    return 0.5f * x * (1.0f + t);
}

__global__ void gelu_scale_max_kernel(const float* __restrict__ inp, float* __restrict__ out,
                                     int B, int S, float scale) {
    int b = blockIdx.x;
    if (b >= B) return;

    const float* base = inp + ((size_t)b) * S;
    float local_max = -INFINITY;

    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        float y = gelu_tanh(base[i]) * scale;
        local_max = fmaxf(local_max, y);
    }

    __shared__ float shmem[256];
    int tid = threadIdx.x;
    shmem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] = fmaxf(shmem[tid], shmem[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) out[b] = shmem[0];
}

torch::Tensor gelu_scale_max(torch::Tensor x, double scale_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (B, S)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t B64 = x.size(0);
    int64_t S64 = x.size(1);
    int B = (int)B64;
    int S = (int)S64;

    auto out = torch::empty({B64}, x.options());
    const int threads = 256;
    hipLaunchKernelGGL(gelu_scale_max_kernel, dim3(B), dim3(threads), 0, 0,
                       (const float*)x.data_ptr<float>(), (float*)out.data_ptr<float>(), B, S, (float)scale_factor);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_scale_max", &gelu_scale_max, "GELU+scale+max over last dim (HIP)");
}
'''

ext = load_inline(
    name="gelu_scale_max_ext",
    cpp_sources="",
    cuda_sources=hip_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        assert out_features % pool_kernel_size == 0
        assert pool_kernel_size == 16, "optimized path assumes kernel=16"
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.reduced_out = out_features // pool_kernel_size

        # Keep the original Linear so parameter initialization matches the reference.
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = float(scale_factor)

        # Cached reduced weights/bias (computed lazily on the current device)
        self.register_buffer("_w_red", None, persistent=False)
        self.register_buffer("_b_red", None, persistent=False)
        self._cache_device = None
        self._cache_dtype = None

    def _ensure_reduced_params(self):
        w = self.matmul.weight  # (out_features, in_features)
        b = self.matmul.bias    # (out_features,)
        if (
            self._w_red is None
            or self._cache_device != w.device
            or self._cache_dtype != w.dtype
        ):
            # Reduce along out_features in blocks of 16: exact equivalent to AvgPool1d(kernel=16,stride=16)
            w_red = w.view(self.reduced_out, self.pool_kernel_size, self.in_features).mean(dim=1).contiguous()
            b_red = b.view(self.reduced_out, self.pool_kernel_size).mean(dim=1).contiguous()
            self._w_red = w_red
            self._b_red = b_red
            self._cache_device = w.device
            self._cache_dtype = w.dtype

    def forward(self, x):
        self._ensure_reduced_params()
        # Equivalent to: y = AvgPool1d(Linear(x).unsqueeze(1)).squeeze(1)
        y = F.linear(x, self._w_red, self._b_red)
        # Fused GELU + scale + max over 512
        return ext.gelu_scale_max(y.contiguous(), self.scale_factor)


batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
