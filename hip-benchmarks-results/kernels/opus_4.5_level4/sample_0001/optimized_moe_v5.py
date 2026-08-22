import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized HIP kernels for MoE
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64

// Optimized fused SiLU(gate) * up kernel with warp-level processing
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float g = gate[idx];
        // Use fast reciprocal and exponential
        float sigmoid_g = __frcp_rn(1.0f + __expf(-g));
        out[idx] = g * sigmoid_g * up[idx];
    }
}

// Optimized weighted scatter-add with warp-level coalescing
__global__ void weighted_scatter_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ expert_out,
    const float* __restrict__ weights,
    const long* __restrict__ token_indices,
    int num_items,
    int hidden_size
) {
    int item_idx = blockIdx.x;
    if (item_idx >= num_items) return;
    
    int token_idx = token_indices[item_idx];
    float w = weights[item_idx];
    
    // Each thread handles multiple hidden dimensions
    for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
        float val = expert_out[item_idx * hidden_size + h_idx] * w;
        atomicAdd(&out[token_idx * hidden_size + h_idx], val);
    }
}

// Batched weighted scatter-add: handles multiple items with same target
// Reduces atomic contention by accumulating locally first
__global__ void batched_weighted_scatter_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ expert_out,
    const float* __restrict__ weights,
    const long* __restrict__ token_indices,
    const long* __restrict__ sorted_indices,  // Indices sorted by token
    int num_items,
    int hidden_size
) {
    int h_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (h_idx >= hidden_size) return;
    
    int item_idx = sorted_indices[blockIdx.x];
    if (item_idx >= num_items) return;
    
    int token_idx = token_indices[item_idx];
    float w = weights[item_idx];
    float val = expert_out[item_idx * hidden_size + h_idx] * w;
    atomicAdd(&out[token_idx * hidden_size + h_idx], val);
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

void weighted_scatter_add_hip(
    torch::Tensor out,
    torch::Tensor expert_out,
    torch::Tensor weights,
    torch::Tensor token_indices
) {
    int num_items = expert_out.size(0);
    int hidden_size = expert_out.size(1);
    
    const int block_size = 256;
    
    // Launch one block per item for better parallelism
    weighted_scatter_add_kernel<<<num_items, block_size>>>(
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
    name="fused_moe_ops_v5",
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
        self.experts_per_group = n_routed_experts // n_group

        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, h = hidden_states.shape
        num_tokens = bsz * seq_len
        hidden_states = hidden_states.view(-1, h)

        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Reshape for group selection
        scores_grouped = scores_for_choice.view(num_tokens, self.n_group, self.experts_per_group)
        
        # Get top-2 per group for group scoring
        group_topk = scores_grouped.topk(2, dim=-1)[0]
        group_scores = group_topk.sum(dim=-1)  # (num_tokens, n_group)
        
        # Select top groups
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        
        # Create mask for selected groups
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, True)
        
        # Expand mask to expert level
        score_mask = group_mask.unsqueeze(-1).expand(-1, -1, self.experts_per_group).reshape(num_tokens, -1)
        
        # Mask unselected experts
        tmp_scores = scores_for_choice.masked_fill(~score_mask, float('-inf'))
        
        # Select top-k experts
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class ModelNew(nn.Module):
    """
    Optimized MoE with batched expert processing using sorted token indices.
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

        # Create output tensor
        y = torch.zeros(num_tokens, self.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Flatten expert assignments for batched processing
        flat_expert_idx = topk_idx.view(-1)  # (num_tokens * top_k,)
        flat_weights = topk_weight.view(-1)   # (num_tokens * top_k,)
        
        # Create token indices for scatter
        token_indices = torch.arange(num_tokens, device=hidden_states.device)
        flat_token_indices = token_indices.unsqueeze(1).expand(-1, self.num_experts_per_tok).reshape(-1)
        
        # Sort by expert for better batching
        sorted_expert_idx, sort_indices = flat_expert_idx.sort()
        sorted_token_indices = flat_token_indices[sort_indices]
        sorted_weights = flat_weights[sort_indices]
        
        # Find expert boundaries
        expert_counts = torch.bincount(sorted_expert_idx, minlength=self.n_routed_experts)
        expert_offsets = torch.zeros(self.n_routed_experts + 1, device=hidden_states.device, dtype=torch.long)
        expert_offsets[1:] = expert_counts.cumsum(0)
        
        # Process each expert in batch
        for expert_id in range(self.n_routed_experts):
            start = expert_offsets[expert_id].item()
            end = expert_offsets[expert_id + 1].item()
            
            if start >= end:
                continue
            
            # Get batch of token indices and weights for this expert
            batch_token_indices = sorted_token_indices[start:end]
            batch_weights = sorted_weights[start:end]
            
            # Gather input tokens for this expert
            expert_input = hidden_states[batch_token_indices]  # (batch, hidden)
            
            # Expert MLP computation using efficient batch GEMM
            gate_out = F.linear(expert_input, self.gate_proj[expert_id])
            up_out = F.linear(expert_input, self.up_proj[expert_id])
            
            # Fused SiLU * mul
            intermediate = self.fused_ops.fused_silu_mul_hip(gate_out, up_out)
            
            # Down projection
            expert_out = F.linear(intermediate, self.down_proj[expert_id])
            
            # Weighted scatter add
            self.fused_ops.weighted_scatter_add_hip(
                y, 
                expert_out.contiguous(), 
                batch_weights.contiguous(), 
                batch_token_indices.contiguous()
            )

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
