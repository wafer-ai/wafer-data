import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Force hipcc for ROCm builds
os.environ.setdefault("CXX", "hipcc")

src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float silu_f32(float x){
    // SiLU(x) = x * sigmoid(x)
    return x / (1.0f + expf(-x));
}

__global__ void silu_mul_kernel(const float* __restrict__ gate,
                               const float* __restrict__ up,
                               float* __restrict__ out,
                               int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float g = gate[i];
        float u = up[i];
        out[i] = silu_f32(g) * u;
    }
}

__global__ void scatter_add_weighted_kernel(const float* __restrict__ src, // (N,H)
                                           const int64_t* __restrict__ token_idx, // (N)
                                           const float* __restrict__ w, // (N)
                                           float* __restrict__ out, // (T,H)
                                           int64_t N, int64_t H) {
    int64_t row = (int64_t)blockIdx.x;
    int64_t tid = threadIdx.x;
    if (row >= N) return;

    int64_t t = token_idx[row];
    float ww = w[row];

    // vector over hidden dim
    for (int64_t j = tid; j < H; j += blockDim.x) {
        float v = src[row * H + j] * ww;
        atomicAdd(&out[t * H + j], v);
    }
}

torch::Tensor silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA/HIP tensor");
    TORCH_CHECK(up.is_cuda(), "up must be CUDA/HIP tensor");
    TORCH_CHECK(gate.scalar_type() == at::kFloat, "FP32 only");
    TORCH_CHECK(up.scalar_type() == at::kFloat, "FP32 only");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(), "up must be contiguous");
    TORCH_CHECK(gate.numel() == up.numel(), "size mismatch");

    auto out = torch::empty_like(gate);
    int64_t n = gate.numel();

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;

    hipLaunchKernelGGL(silu_mul_kernel, dim3(blocks), dim3(threads), 0, 0,
                       (const float*)gate.data_ptr<float>(),
                       (const float*)up.data_ptr<float>(),
                       (float*)out.data_ptr<float>(),
                       n);
    return out;
}

void scatter_add_weighted_hip(torch::Tensor src,
                             torch::Tensor token_idx,
                             torch::Tensor w,
                             torch::Tensor out) {
    TORCH_CHECK(src.is_cuda() && token_idx.is_cuda() && w.is_cuda() && out.is_cuda(), "all tensors must be CUDA/HIP");
    TORCH_CHECK(src.scalar_type() == at::kFloat, "src FP32 only");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w FP32 only");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out FP32 only");
    TORCH_CHECK(token_idx.scalar_type() == at::kLong, "token_idx must be int64");

    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(token_idx.is_contiguous(), "token_idx must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    int64_t N = src.size(0);
    int64_t H = src.size(1);
    TORCH_CHECK(token_idx.numel() == N, "token_idx shape mismatch");
    TORCH_CHECK(w.numel() == N, "w shape mismatch");
    TORCH_CHECK(out.size(1) == H, "out hidden mismatch");

    int threads = 256;
    int blocks = (int)N;
    hipLaunchKernelGGL(scatter_add_weighted_kernel, dim3(blocks), dim3(threads), 0, 0,
                       (const float*)src.data_ptr<float>(),
                       (const int64_t*)token_idx.data_ptr<int64_t>(),
                       (const float*)w.data_ptr<float>(),
                       (float*)out.data_ptr<float>(),
                       N, H);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_mul_hip", &silu_mul_hip, "SiLU(gate) * up (FP32, HIP)");
    m.def("scatter_add_weighted_hip", &scatter_add_weighted_hip, "scatter add weighted (FP32, HIP)");
}
"""

moe_fused = load_inline(
    name="moe_gated_gemm_fused",
    cpp_sources=src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.gate_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.02)
        self.up_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * 0.02)

        self._kern = moe_fused

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        # Force FP32 path for the custom kernels
        if x.dtype != torch.float32:
            x = x.float()
        if expert_weights.dtype != torch.float32:
            expert_weights = expert_weights.float()

        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]

        x_flat = x.reshape(-1, self.hidden_size).contiguous()
        num_tokens = x_flat.shape[0]

        output = torch.zeros((num_tokens, self.hidden_size), device=x.device, dtype=torch.float32)

        # Still loop per expert for GEMMs (rocBLAS-backed), but fuse SiLU+mul and fused scatter-add.
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx)
            if not bool(expert_mask.any()):
                continue

            batch_idx, seq_idx, slot_idx = torch.where(expert_mask)
            token_indices = (batch_idx * seq_len + seq_idx).to(torch.long).contiguous()
            weights = expert_weights[batch_idx, seq_idx, slot_idx].contiguous()

            expert_input = x_flat.index_select(0, token_indices).contiguous()

            gate_lin = F.linear(expert_input, self.gate_proj[expert_idx]).contiguous()
            up_lin = F.linear(expert_input, self.up_proj[expert_idx]).contiguous()

            intermediate = self._kern.silu_mul_hip(gate_lin, up_lin)
            expert_output = F.linear(intermediate, self.down_proj[expert_idx]).contiguous()

            # output[token_indices] += expert_output * weights
            self._kern.scatter_add_weighted_hip(expert_output, token_indices, weights, output)

        return output.view(batch, seq_len, self.hidden_size)


# KernelBench hooks
batch_size = 4
seq_len = 2048
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)

    expert_indices = torch.stack([
        torch.randperm(num_experts, device="cpu")[:top_k]
        for _ in range(batch_size * seq_len)
    ]).view(batch_size, seq_len, top_k).to("cuda")

    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k, device="cuda", dtype=torch.float32), dim=-1)
    return [x, expert_indices, expert_weights]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_experts]
