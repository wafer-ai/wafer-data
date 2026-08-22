import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <cstdint>

__global__ void compute_rms_kernel(const float* __restrict__ x, float* __restrict__ rms, int N, int C, int H, int W, float eps) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pos = N * H * W;
    if (gid >= num_pos) return;
    int hw = H * W;
    int n = gid / hw;
    int rem = gid % hw;
    int h = rem / W;
    int w = rem % W;
    float sum_sq = 0.0f;
    for (int c = 0; c < C; ++c) {
        int idx = ((n * C + c) * H + h) * W + w;
        float val = x[idx];
        sum_sq += val * val;
    }
    int rms_idx = (n * H + h) * W + w;
    rms[rms_idx] = sqrtf(sum_sq / static_cast<float>(C) + eps);
}

__global__ void normalize_kernel(const float* __restrict__ x, const float* __restrict__ rms, float* __restrict__ out, int N, int C, int H, int W) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (gid >= total) return;
    int W_stride = 1;
    int H_stride = W;
    int C_stride = H * W;
    int N_stride = C * H * W;
    // decode
    int w = gid % W;
    int tmp = gid / W;
    int h = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int n = tmp / C;
    int rms_idx = n * (H * W) + h * W + w;
    out[gid] = x[gid] / rms[rms_idx];
}

torch::Tensor rmsnorm_hip(torch::Tensor x, float eps) {
    TORCH_CHECK(x.dim() == 4, "Input must be 4D tensor");
    auto out = torch::empty_like(x);
    auto options = x.options();
    int64_t N_ = x.size(0);
    int64_t C_ = x.size(1);
    int64_t H_ = x.size(2);
    int64_t W_ = x.size(3);
    int N = static_cast<int>(N_);
    int C = static_cast<int>(C_);
    int H = static_cast<int>(H_);
    int W = static_cast<int>(W_);
    int64_t num_pos_ = N_ * H_ * W_;
    int64_t total_ = num_pos_ * C_;
    TORCH_CHECK(num_pos_ <= std::numeric_limits<int>::max(), "Too large");
    TORCH_CHECK(total_ <= std::numeric_limits<int>::max(), "Too large");
    int num_pos = static_cast<int>(num_pos_);
    int total = static_cast<int>(total_);

    std::vector<int64_t> rms_dims{N_, 1LL, H_, W_};
    auto rms = torch::zeros(rms_dims, options);

    // compute rms
    const int bs = 256;
    dim3 grid_rms((num_pos + bs - 1) / bs);
    dim3 block_rms(bs);
    compute_rms_kernel<<<grid_rms, block_rms>>>(x.data_ptr<float>(), rms.data_ptr<float>(), N, C, H, W, eps);

    // normalize
    dim3 grid_norm((total + bs - 1) / bs);
    dim3 block_norm(bs);
    normalize_kernel<<<grid_norm, block_norm>>>(x.data_ptr<float>(), rms.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W);

    return out;
}
"""

rmsnorm = load_inline(
    name="rmsnorm",
    cpp_sources=cpp_source,
    functions=["rmsnorm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm.rmsnorm_hip(x, self.eps)

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
