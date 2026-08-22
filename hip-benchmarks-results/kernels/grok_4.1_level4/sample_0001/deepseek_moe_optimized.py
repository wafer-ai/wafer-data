import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

silu_mul_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void silu_mul_kernel(const float* gate, const float* up, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float sig = 1.0f / (1.0f + __expf(-g));
        out[idx] = g * sig * up[idx];
    }
}

torch::Tensor silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    silu_mul_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    return out;
}
"""

silu_mul = load_inline(
    name="silu_mul",
    cpp_sources=silu_mul_cpp_source,
    functions=["silu_mul_hip"],
    verbose=True,
)

# DeepSeek-V3 Mixture of Experts (MoE) Layer
# Optimized with dispatch-based expert computation and custom HIP silu-mul kernel

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

        self.gate_proj = nn.Parameter(
            torch.randn(n_routed_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(n_routed_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(n_routed_experts, hidden_size, intermediate_size) * 0.02
        )

        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
        )

        if n_shared_experts > 0:
            shared_intermediate = intermediate_size * n_shared_experts
            self.shared_gate_proj = nn.Linear(hidden_size, shared_intermediate, bias=False)
            self.shared_up_proj = nn.Linear(hidden_size, shared_intermediate, bias=False)
            self.shared_down_proj = nn.Linear(shared_intermediate, hidden_size, bias=False)
        else:
            self.shared_gate_proj = None

        self.silu_mul = silu_mul

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert not self.training, "DeepSeek MoE grouped selection is inference-only"

        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states_flat.shape[0]
        device = hidden_states_flat.device
        num_experts_per_tok = self.num_experts_per_tok

        token_ids = torch.arange(num_tokens, dtype=torch.long, device=device).unsqueeze(1).expand(
            num_tokens, num_experts_per_tok
        ).reshape(-1)
        expert_ids = topk_idx.view(-1)
        assign_weights = topk_weight.view(-1)

        order = torch.argsort(expert_ids)
        token_ids = token_ids[order]
        expert_ids = expert_ids[order]
        assign_weights = assign_weights[order]

        counts = torch.zeros(self.n_routed_experts, dtype=torch.long, device=device)
        ones = torch.ones_like(expert_ids, dtype=torch.long)
        counts.scatter_add_(0, expert_ids, ones)
        offsets = torch.zeros(self.n_routed_experts + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)

        y = torch.zeros_like(hidden_states_flat)

        for e in range(self.n_routed_experts):
            start = offsets[e]
            end = offsets[e + 1]
            if start == end:
                continue

            pos = token_ids[start:end]
            ws = assign_weights[start:end]

            input_e = hidden_states_flat.index_select(0, pos)

            gate_proj_t = self.gate_proj[e].t()
            up_proj_t = self.up_proj[e].t()
            down_proj_t = self.down_proj[e].t()

            gate_out = torch.matmul(input_e.float(), gate_proj_t.float())
            up_out = torch.matmul(input_e.float(), up_proj_t.float())
            intermediate = self.silu_mul.silu_mul_hip(gate_out, up_out)
            expert_out = torch.matmul(intermediate, down_proj_t.float())

            contrib = expert_out * ws.unsqueeze(-1)
            y.index_add_(0, pos, contrib)

        y = y.view(*orig_shape)

        if self.shared_gate_proj is not None:
            shared_gate_out = self.shared_gate_proj(identity)
            shared_up_out = self.shared_up_proj(identity)
            shared_intermediate = F.silu(shared_gate_out) * shared_up_out
            shared_out = self.shared_down_proj(shared_intermediate)
            y = y + shared_out

        return y


# configuration
batch_size = 4
seq_len = 2048
hidden_size = 2048
intermediate_size = 1408
n_routed_experts = 64
num_experts_per_tok = 8
n_group = 8
topk_group = 4
n_shared_experts = 2
routed_scaling_factor = 2.5


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size).cuda()]


def get_init_inputs():
    return [
        hidden_size,
        intermediate_size,
        n_routed_experts,
        num_experts_per_tok,
        n_group,
        topk_group,
        n_shared_experts,
        routed_scaling_factor,
    ]