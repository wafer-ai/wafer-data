import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_gated_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void fused_gated_linear_kernel(const float *A, const float *Bg, const float *Bu, float *C, int M, int K, int N) {
    constexpr int RM = 64;
    constexpr int RN = 16;
    constexpr int RK = 128;
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
        // Load sA 64*128=8192 elems, 8 per thread
        for (int p = 0; p < 8; ++p) {
            int aid = tid * 8 + p;
            if (aid < RM * RK) {
                int ra = aid / RK;
                int ca = aid % RK;
                int km = bk + ca;
                int mm = blockIdx.y * RM + ra;
                sA[ra * RK + ca] = (mm < M && km < K) ? A[mm * K + km] : 0.0f;
            }
        }

        // Load sBg 16*128=2048 elems, 2 per thread
        for (int p = 0; p < 2; ++p) {
            int bid = tid * 2 + p;
            if (bid < RN * RK) {
                int rb = bid / RK;
                int cb = bid % RK;
                int km = bk + cb;
                int nn = blockIdx.x * RN + rb;
                sBg[rb * RK + cb] = (nn < N && km < K) ? Bg[nn * K + km] : 0.0f;
            }
        }

        // Load sBu 16*128=2048 elems, 2 per thread
        for (int p = 0; p < 2; ++p) {
            int bid = tid * 2 + p;
            if (bid < RN * RK) {
                int rb = bid / RK;
                int cb = bid % RK;
                int km = bk + cb;
                int nn = blockIdx.x * RN + rb;
                sBu[rb * RK + cb] = (nn < N && km < K) ? Bu[nn * K + km] : 0.0f;
            }
        }

        __syncthreads();

        for (int kk = 0; kk < RK; ++kk) {
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

torch::Tensor fused_gated_hip(torch::Tensor input, torch::Tensor gate_weight, torch::Tensor up_weight) {
    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = gate_weight.size(0);
    auto output = torch::empty({M, N}, input.options());

    const float *d_A = input.data_ptr<float>();
    const float *d_Bg = gate_weight.data_ptr<float>();
    const float *d_Bu = up_weight.data_ptr<float>();
    float *d_C = output.data_ptr<float>();

    constexpr int RM = 64;
    constexpr int RN = 16;
    constexpr int RK = 128;
    constexpr int block_size = RM * RN;
    dim3 block(block_size);
    dim3 grid((N + RN - 1) / RN, (M + RM - 1) / RM);

    size_t shmem_bytes = (RM * RK + 2 * RN * RK) * sizeof(float);

    fused_gated_linear_kernel<<<grid, block, shmem_bytes>>>(
        d_A, d_Bg, d_Bu, d_C,
        static_cast<int>(M), static_cast<int>(K), static_cast<int>(N)
    );

    return output;
}
"""

fused_gated = load_inline(
    name="fused_gated",
    cpp_sources=fused_gated_cpp,
    functions=["fused_gated_hip"],
    verbose=True,
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
    x = torch.randn(batch_size, seq_len, hidden_size)

    expert_indices = torch.stack([
        torch.randperm(num_experts)[:top_k]
        for _ in range(batch_size * seq_len)
    ]).view(batch_size, seq_len, top_k)

    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)

    return [x, expert_indices, expert_weights]


def get_init_inputs():
    hidden_size = 4096
    intermediate_size = 14336
    num_experts = 8
    return [hidden_size, intermediate_size, num_experts]
