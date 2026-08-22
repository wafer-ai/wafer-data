import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;

__global__ void matmul_kernel(
    const float *A, 
    const float *B, 
    float *C,
    int M, 
    int N, 
    int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = bx * TILE_M + ty;
    int col = by * TILE_N + tx;
    if (row >= M || col >= N) return;

    float acc = 0.0f;

    int num_tiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < num_tiles; ++t) {
        As[ty][tx] = 0.0f;
        if (tx < TILE_K && row < M && (t * TILE_K + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_K + tx];
        }
        Bs[ty][tx] = 0.0f;
        if (ty < TILE_K && (t * TILE_K + ty) < K) {
            Bs[ty][tx] = B[col * K + t * TILE_K + ty];
        }
        __syncthreads();
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = acc;
}

torch::Tensor fused_matmul_hip(
    torch::Tensor input,
    torch::Tensor weight
) {
    int64_t n_rows = input.size(0);
    int64_t k_dim = input.size(1);
    int64_t n_cols = weight.size(0);

    int M = static_cast<int>(n_rows);
    int N = static_cast<int>(n_cols);
    int K = static_cast<int>(k_dim);

    input = input.contiguous();
    weight = weight.contiguous();

    auto out = torch::zeros({n_rows, n_cols}, input.options());

    dim3 block(TILE_N, TILE_M);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

    size_t shmem_bytes = sizeof(float) * (TILE_M * TILE_K + TILE_K * TILE_N);

    hipLaunchKernelGGL(
        matmul_kernel,
        grid,
        block,
        shmem_bytes,
        0,
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        M, 
        N, 
        K
    );
    hipStreamSynchronize(0);
    return out;
}
"""

moe_fused = load_inline(
    name="moe_fused",
    cpp_sources=cpp_source,
    functions=["fused_matmul_hip"],
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

        self.fused_matmul = moe_fused.fused_matmul_hip

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

        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)

        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx)
            if not expert_mask.any():
                continue
            batch_idx, seq_idx, slot_idx = torch.where(expert_mask)
            token_indices = batch_idx * seq_len + seq_idx
            weights = expert_weights[batch_idx, seq_idx, slot_idx]
            expert_input = x_flat[token_indices]
            gate = F.silu(self.fused_matmul(expert_input, self.gate_proj[expert_idx]))
            up = self.fused_matmul(expert_input, self.up_proj[expert_idx])
            intermediate = gate * up
            expert_output = F.linear(intermediate, self.down_proj[expert_idx])
            output.index_add_(0, token_indices, expert_output * weights.unsqueeze(-1))

        return output.view(batch, seq_len, self.hidden_size)
