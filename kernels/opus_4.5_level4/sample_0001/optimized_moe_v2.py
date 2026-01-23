import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused SiLU-multiply kernel and optimized MoE kernels
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused SiLU(gate) * up kernel
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
        float silu_g = g * sigmoid_g;
        out[idx] = silu_g * up[idx];
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

// Scatter add kernel: out[idx] += weight * value for each token
__global__ void weighted_scatter_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ expert_out,
    const float* __restrict__ weights,
    const long* __restrict__ token_indices,
    int num_items,
    int hidden_size
) {
    int item_idx = blockIdx.x;
    int h_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (item_idx < num_items && h_idx < hidden_size) {
        int token_idx = token_indices[item_idx];
        float w = weights[item_idx];
        float val = expert_out[item_idx * hidden_size + h_idx];
        atomicAdd(&out[token_idx * hidden_size + h_idx], w * val);
    }
}

void weighted_scatter_add_hip(
    torch::Tensor out,
    torch::Tensor expert_out,
    torch::Tensor weights,
    torch::Tensor token_indices
) {
    int num_items = expert_out.size(0);
    int hidden_size = expert_out.size(1);
    
    const int block_size = 256;
    dim3 grid(num_items, (hidden_size + block_size - 1) / block_size);
    
    weighted_scatter_add_kernel<<<grid, block_size>>>(
        out.data_ptr<float>(),
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        token_indices.data_ptr<long>(),
        num_items,
        hidden_size
    );
}
"""

cpp_source = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
void weighted_scatter_add_hip(torch::Tensor out, torch::Tensor expert_out, torch::Tensor weights, torch::Tensor token_indices);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v2",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip"],
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
    """
    Memory-efficient MoE implementation that processes tokens per expert
    instead of gathering all expert weights at once.
    """
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

        # Memory-efficient expert computation: process each expert separately
        # instead of gathering all weights at once
        
        # Create output tensor
        y = torch.zeros(num_tokens, self.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Process each expert - gather tokens that use this expert
        for expert_id in range(self.n_routed_experts):
            # Find which token-slot pairs use this expert
            # topk_idx: (num_tokens, top_k)
            mask = (topk_idx == expert_id)  # (num_tokens, top_k)
            
            if not mask.any():
                continue
            
            # Get token indices and slot indices
            token_ids, slot_ids = torch.where(mask)
            
            # Get the input tokens for this expert
            expert_input = hidden_states[token_ids]  # (num_selected, hidden_size)
            
            # Get weights for this expert
            expert_weights = topk_weight[token_ids, slot_ids]  # (num_selected,)
            
            # Expert computation
            gate_out = F.linear(expert_input, self.gate_proj[expert_id])  # (num_selected, intermediate)
            up_out = F.linear(expert_input, self.up_proj[expert_id])      # (num_selected, intermediate)
            
            # Fused SiLU * mul
            intermediate = self.fused_ops.fused_silu_mul_hip(gate_out, up_out)
            
            # Down projection
            expert_out = F.linear(intermediate, self.down_proj[expert_id])  # (num_selected, hidden)
            
            # Weighted scatter add using custom kernel
            self.fused_ops.weighted_scatter_add_hip(y, expert_out, expert_weights, token_ids)

        y = y.view(*orig_shape)

        if self.shared_gate_proj is not None:
            shared_gate_out = self.shared_gate_proj(identity)
            shared_up_out = self.shared_up_proj(identity)
            shared_intermediate = self.fused_ops.fused_silu_mul_hip(
                shared_gate_out.contiguous().view(-1), 
                shared_up_out.contiguous().view(-1)
            ).view(shared_gate_out.shape)
            shared_out = self.shared_down_proj(shared_intermediate)
            y = y + shared_out

        return y
