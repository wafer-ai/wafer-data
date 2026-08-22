import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

class MoEGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


cpp_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void sparse_mm_kernel(
    const float *A,
    int nexp,
    int M,
    int K,
    const int32_t *idx,
    int pairs,
    const float *x,
    int x_stride,
    float *out,
    int is_token_x_int,
    int top_k
) {
    const int TILE_M = 256;
    const int TILE_K = 64;
    int pair_id = blockIdx.x;
    int m_tile = blockIdx.y * TILE_M;
    int m = m_tile + threadIdx.x;
    if (pair_id >= pairs || m >= M) return;
    int e = idx[pair_id];
    int input_row = is_token_x_int ? pair_id / top_k : pair_id;
    size_t a_off = ((size_t)e * (size_t)M * (size_t)K) + ((size_t)m * (size_t)K);
    size_t x_off = ((size_t)input_row * (size_t)x_stride);
    __shared__ float x_sh[TILE_K];
    float sum = 0.0f;
    int nk = (K + TILE_K - 1) / TILE_K;
    for (int kt = 0; kt < nk; ++kt) {
        int kbase = kt * TILE_K;
        if (threadIdx.x < TILE_K) {
            int kl = kbase + threadIdx.x;
            x_sh[threadIdx.x] = (kl < K) ? x[x_off + kl] : 0.0f;
        }
        __syncthreads();
        for (int kk = 0; kk < TILE_K; ++kk) {
            int kl = kbase + kk;
            if (kl < K) {
                sum += A[a_off + kl] * x_sh[kk];
            }
        }
        __syncthreads();
    }
    out[(size_t)pair_id * M + m] = sum;
}

void sparse_mm_hip(
    torch::Tensor A,
    torch::Tensor idx,
    torch::Tensor x,
    bool is_token_x,
    int top_k,
    torch::Tensor out
) {
    int nexp = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int pairs = idx.numel();
    int x_stride = x.size(1);
    const int TILE_M = 256;
    dim3 tb(TILE_M);
    dim3 blocks(pairs, (M + TILE_M - 1) / TILE_M);
    int is_token_x_int = is_token_x ? 1 : 0;
    sparse_mm_kernel<<<blocks, tb>>>(
        A.data_ptr<float>(),
        nexp,
        M,
        K,
        idx.data_ptr<int32_t>(),
        pairs,
        x.data_ptr<float>(),
        x_stride,
        out.data_ptr<float>(),
        is_token_x_int,
        top_k
    );
}
"""

class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        n_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor

        self.gate = MoEGate(
            hidden_size,
            n_routed_experts,
            num_experts_per_tok,
            n_group,
            topk_group,
            routed_scaling_factor,
        )

        self.gate_proj = nn.Parameter(torch.randn(n_routed_experts, intermediate_size, hidden_size) * 0.02)
        self.up_proj = nn.Parameter(torch.randn(n_routed_experts, intermediate_size, hidden_size) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(n_routed_experts, hidden_size, intermediate_size) * 0.02)

        if n_shared_experts > 0:
            shared_intermediate = intermediate_size * n_shared_experts
            self.shared_gate_proj = nn.Linear(hidden_size, shared_intermediate, bias=False)
            self.shared_up_proj = nn.Linear(hidden_size, shared_intermediate, bias=False)
            self.shared_down_proj = nn.Linear(shared_intermediate, hidden_size, bias=False)
        else:
            self.shared_gate_proj = None

        self.sparse_mm = load_inline(
            name="moe_sparse_mm",
            cpp_sources=cpp_source,
            functions=["sparse_mm_hip"],
            verbose=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert not self.training, "DeepSeek MoE grouped selection is inference-only"

        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        ntok = hidden_states_flat.shape[0]
        top_k = self.num_experts_per_tok
        flat_topk_idx = topk_idx.flatten().to(torch.int32, device=hidden_states.device)

        device = hidden_states.device

        gate_out = torch.empty((ntok * top_k, self.intermediate_size), dtype=torch.float32, device=device)
        up_out = torch.empty_like(gate_out)

        self.sparse_mm.sparse_mm_hip(
            self.gate_proj.float(),
            flat_topk_idx,
            hidden_states_flat.float(),
            True,
            top_k,
            gate_out
        )

        self.sparse_mm.sparse_mm_hip(
            self.up_proj.float(),
            flat_topk_idx,
            hidden_states_flat.float(),
            True,
            top_k,
            up_out
        )

        inter = F.silu(gate_out) * up_out

        expert_out = torch.empty((ntok * top_k, self.hidden_size), dtype=torch.float32, device=device)

        self.sparse_mm.sparse_mm_hip(
            self.down_proj.float(),
            flat_topk_idx,
            inter,
            False,
            top_k,
            expert_out
        )

        expert_out = expert_out.view(ntok, top_k, self.hidden_size)
        y = (expert_out * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(*orig_shape)

        if self.shared_gate_proj is not None:
            shared_gate = self.shared_gate_proj(identity.float())
            shared_up = self.shared_up_proj(identity.float())
            shared_inter = F.silu(shared_gate) * shared_up
            shared_out = self.shared_down_proj(shared_inter)
            y = y + shared_out

        return y
