import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused SiLU multiply kernel
fused_silu_mul_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        out[idx] = g * sigmoid_g * up[idx];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_silu_mul_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}
"""

# Fused weighted sum kernel for expert outputs
fused_weighted_sum_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_weighted_sum_kernel(
    const float* __restrict__ expert_out,  // (num_tokens, top_k, hidden)
    const float* __restrict__ weights,      // (num_tokens, top_k)
    float* __restrict__ out,                // (num_tokens, hidden)
    int num_tokens,
    int top_k,
    int hidden_size
) {
    int token_idx = blockIdx.x;
    int hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (token_idx < num_tokens && hidden_idx < hidden_size) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            float w = weights[token_idx * top_k + k];
            float val = expert_out[token_idx * top_k * hidden_size + k * hidden_size + hidden_idx];
            sum += w * val;
        }
        out[token_idx * hidden_size + hidden_idx] = sum;
    }
}

torch::Tensor fused_weighted_sum_hip(torch::Tensor expert_out, torch::Tensor weights, int hidden_size) {
    int num_tokens = expert_out.size(0);
    int top_k = expert_out.size(1);
    
    auto out = torch::empty({num_tokens, hidden_size}, expert_out.options());
    
    const int block_size = 256;
    dim3 blocks(num_tokens, (hidden_size + block_size - 1) / block_size);
    
    fused_weighted_sum_kernel<<<blocks, block_size>>>(
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        out.data_ptr<float>(),
        num_tokens,
        top_k,
        hidden_size
    );
    
    return out;
}
"""

cpp_source = fused_silu_mul_source + fused_weighted_sum_source

fused_ops = load_inline(
    name="fused_moe_ops",
    cpp_sources=cpp_source,
    functions=["fused_silu_mul_hip", "fused_weighted_sum_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


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
        self.n_shared_experts = n_shared_experts
        self.fused_ops = fused_ops

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert not self.training, "DeepSeek MoE grouped selection is inference-only"

        identity = hidden_states
        orig_shape = hidden_states.shape
        bsz, seq_len, _ = orig_shape

        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]

        flat_topk_idx = topk_idx.view(-1)

        expanded_tokens = hidden_states.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1)
        expanded_tokens = expanded_tokens.reshape(-1, self.hidden_size)

        selected_gate = self.gate_proj[flat_topk_idx]
        selected_up = self.up_proj[flat_topk_idx]
        selected_down = self.down_proj[flat_topk_idx]

        x = expanded_tokens.unsqueeze(-1)

        gate_out = torch.bmm(selected_gate, x).squeeze(-1)
        up_out = torch.bmm(selected_up, x).squeeze(-1)

        # Use fused SiLU * multiply kernel
        intermediate = self.fused_ops.fused_silu_mul_hip(gate_out.contiguous(), up_out.contiguous())

        expert_out = torch.bmm(selected_down, intermediate.unsqueeze(-1)).squeeze(-1)

        expert_out = expert_out.view(num_tokens, self.num_experts_per_tok, self.hidden_size)

        # Use fused weighted sum kernel
        y = self.fused_ops.fused_weighted_sum_hip(
            expert_out.contiguous(), 
            topk_weight.contiguous(),
            self.hidden_size
        )

        y = y.view(*orig_shape)

        if self.shared_gate_proj is not None:
            shared_gate = self.shared_gate_proj(identity)
            shared_up = self.shared_up_proj(identity)
            shared_intermediate = self.fused_ops.fused_silu_mul_hip(
                shared_gate.contiguous(), 
                shared_up.contiguous()
            )
            shared_out = self.shared_down_proj(shared_intermediate)
            y = y + shared_out

        return y


def custom_kernel(inputs):
    hidden_states = inputs[0]
    
    # Initialize model parameters
    hidden_size = 2048
    intermediate_size = 1408
    n_routed_experts = 64
    num_experts_per_tok = 8
    n_group = 8
    topk_group = 4
    n_shared_experts = 2
    routed_scaling_factor = 2.5
    
    model = ModelNew(
        hidden_size,
        intermediate_size,
        n_routed_experts,
        num_experts_per_tok,
        n_group,
        topk_group,
        n_shared_experts,
        routed_scaling_factor,
    ).cuda().eval()
    
    with torch.no_grad():
        return model(hidden_states)
