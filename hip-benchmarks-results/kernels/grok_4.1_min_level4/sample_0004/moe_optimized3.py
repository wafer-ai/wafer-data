import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

fused_gated_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

torch::Tensor fused_gated_hip(torch::Tensor input, torch::Tensor gate_weight, torch::Tensor up_weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(gate_weight.is_cuda(), "gate_weight must be a CUDA tensor");
    TORCH_CHECK(up_weight.is_cuda(), "up_weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must be 2D");
    TORCH_CHECK(up_weight.dim() == 2, "up_weight must be 2D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Only FP32 supported");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat, "Only FP32 supported");
    TORCH_CHECK(up_weight.scalar_type() == torch::kFloat, "Only FP32 supported");

    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = gate_weight.size(0);
    TORCH_CHECK(up_weight.size(0) == N, "up_weight must have same output dim");

    auto output = torch::empty({M, N}, input.options());

    if (M == 0 || N == 0 || K == 0) {
        return output;
    }

    const float *d_A = input.data_ptr<float>();
    const float *d_Bg = gate_weight.data_ptr<float>();
    const float *d_Bu = up_weight.data_ptr<float>();
    float *d_C = output.mutable_data_ptr<float>();

    constexpr int RM = 64;
    constexpr int RN = 16;
    constexpr int RK = 64;
    constexpr int block_size = RM * RN;
    dim3 block(block_size);
    dim3 grid((N + RN - 1) / RN, (M + RM - 1) / RM);

    size_t shmem_bytes = (RM * RK + 2 * RN * RK) * sizeof(float);

    hipLaunchKernelGpv(
        fused_gated_linear_kernel,
        grid,
        block,
        shmem_bytes,
        0,
        d_A, d_Bg, d_Bu, d_C, M, K, N
    );

    return output;
}
"""

__global__ void fused_gated_linear_kernel(
    const float *A, const float *Bg, const float *Bu, float *C,
    int M, int K, int N
) {
    constexpr int RM = 64;
    constexpr int RN = 16;
    constexpr int RK = 64;
    extern __shared__ float shmem[];
    float* sA = shmem;
    float* sBg = shmem + RM * RK;
    float* sBu = sBg + RN * RK;

    int tid = threadIdx.x;
    int row_tx = tid / RN;
    int col_tx = tid % RN;

    int m = blockIdx.y * RM + row_tx;
    int n = blockIdx.x * RN + col_tx;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int bk = 0; bk < K; bk += RK) {
        // Load sA: RM x RK
        for (int p = 0; p < 4; p++) {
            int aid = tid * 4 + p;
            if (aid < RM * RK) {
                int ra = aid / RK;
                int ca = aid % RK;
                int km = bk + ca;
                int mm = blockIdx.y * RM + ra;
                if (mm < M && km < K) {
                    sA[ra * RK + ca] = A[mm * K + km];
                } else {
                    sA[ra * RK + ca] = 0.0f;
                }
            }
        }

        // Load sBg: RN x RK
        {
            int bid = tid;
            if (bid < RN * RK) {
                int rb = bid / RK;
                int cb = bid % RK;
                int km = bk + cb;
                int nn = blockIdx.x * RN + rb;
                if (nn < N && km < K) {
                    sBg[rb * RK + cb] = Bg[nn * K + km];
                } else {
                    sBg[rb * RK + cb] = 0.0f;
                }
            }
        }

        // Load sBu: RN x RK
        {
            int bid = tid;
            if (bid < RN * RK) {
                int rb = bid / RK;
                int cb = bid % RK;
                int km = bk + cb;
                int nn = blockIdx.x * RN + rb;
                if (nn < N && km < K) {
                    sBu[rb * RK + cb] = Bu[nn * K + km];
                } else {
                    sBu[rb * RK + cb] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int kk = 0; kk < RK; kk++) {
            acc_g += sA[row_tx * RK + kk] * sBg[col_tx * RK + kk];
            acc_u += sA[row_tx * RK + kk] * sBu[col_tx * RK + kk];
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        float x = acc_g;
        float sigmoid = 1.0f / (1.0f + expf(-x));
        C[m * N + n] = x * sigmoid * acc_u;
    }
}

fused_gated = load_inline(
    name="fused_gated",
    cpp_sources=fused_gated_cpp,
    functions=["fused_gated_hip"],
    verbose=True,
    functions_are_extern=True,  # optional
)

class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )
        self.fused_gated = fused_gated

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]
        device = x.device
        dtype = x.dtype

        num_assignments = num_tokens * top_k
        positions = torch.arange(num_assignments, dtype=torch.long, device=device)
        token_assign = positions // top_k
        slot_assign = positions % top_k

        expert_flat = expert_indices.view(num_tokens, top_k)
        weight_flat = expert_weights.view(num_tokens, top_k)

        expert_assign = expert_flat[token_assign, slot_assign]
        weight_assign = weight_flat[token_assign, slot_assign]

        sort_idx = torch.argsort(expert_assign)
        sorted_expert = expert_assign[sort_idx]
        sorted_token = token_assign[sort_idx]
        sorted_weight = weight_assign[sort_idx]

        output = torch.zeros(num_tokens, self.hidden_size, device=device, dtype=dtype)

        for expert_idx in range(self.num_experts):
            start = torch.searchsorted(sorted_expert, expert_idx)
            end = torch.searchsorted(sorted_expert, expert_idx + 1)
            if start == end:
                continue

            token_indices = sorted_token[start:end]
            weights = sorted_weight[start:end]
            expert_input = x_flat[token_indices]

            intermediate = self.fused_gated.fused_gated_hip(expert_input, self.gate_proj[expert_idx], self.up_proj[expert_idx])

            expert_output = F.linear(intermediate, self.down_proj[expert_idx])

            output.index_add_(0, token_indices, expert_output * weights.unsqueeze(-1))

        return output.view(batch, seq_len, self.hidden_size)


def get_inputs():
    batch_size = 4
    seq_len = 2048
    hidden_size = 4096
    num_experts = 8
    top_k = 2
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()

    expert_indices = torch.stack([
        torch.randperm(num_experts)[:top_k]
        for _ in range(batch_size * seq_len)
    ]).view(batch_size, seq_len, top_k).cuda()

    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1).cuda()

    return [x, expert_indices, expert_weights]


def get_init_inputs():
    return [4096, 14336, 8]
