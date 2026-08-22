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

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Fused SiLU(gate) * up kernel - optimized with better memory coalescing
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread if possible
    int base_idx = idx * 4;
    if (base_idx + 3 < size) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int curr_idx = base_idx + i;
            float g = gate[curr_idx];
            float sigmoid_g = __frcp_rn(1.0f + __expf(-g));
            out[curr_idx] = g * sigmoid_g * up[curr_idx];
        }
    } else {
        // Handle boundary
        for (int i = 0; i < 4 && base_idx + i < size; i++) {
            int curr_idx = base_idx + i;
            float g = gate[curr_idx];
            float sigmoid_g = __frcp_rn(1.0f + __expf(-g));
            out[curr_idx] = g * sigmoid_g * up[curr_idx];
        }
    }
}

// Weighted scatter-add with improved atomics
__global__ void weighted_scatter_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ expert_out,
    const float* __restrict__ weights,
    const long* __restrict__ token_indices,
    int num_items,
    int hidden_size
) {
    // Each warp handles one item
    int item_idx = blockIdx.x;
    if (item_idx >= num_items) return;
    
    int token_idx = token_indices[item_idx];
    float w = weights[item_idx];
    
    // Threads collaborate to scatter add hidden dimensions
    for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
        float val = expert_out[item_idx * hidden_size + h_idx];
        atomicAdd(&out[token_idx * hidden_size + h_idx], w * val);
    }
}

// Fused gate projection + up projection + SiLU * up for batched tokens
// This combines three operations into one kernel launch
__global__ void fused_expert_mlp_first_half_kernel(
    const float* __restrict__ input,      // (batch, hidden_size)
    const float* __restrict__ gate_weight, // (intermediate, hidden_size)
    const float* __restrict__ up_weight,   // (intermediate, hidden_size)
    float* __restrict__ output,            // (batch, intermediate)
    int batch_size,
    int hidden_size,
    int intermediate_size
) {
    // Use tiling for better cache utilization
    int batch_idx = blockIdx.x;
    int inter_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && inter_idx < intermediate_size) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        
        #pragma unroll 8
        for (int h = 0; h < hidden_size; h++) {
            float inp = input[batch_idx * hidden_size + h];
            gate_sum += inp * gate_weight[inter_idx * hidden_size + h];
            up_sum += inp * up_weight[inter_idx * hidden_size + h];
        }
        
        // Apply SiLU to gate and multiply by up
        float sigmoid_gate = __frcp_rn(1.0f + __expf(-gate_sum));
        float silu_gate = gate_sum * sigmoid_gate;
        output[batch_idx * intermediate_size + inter_idx] = silu_gate * up_sum;
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = BLOCK_SIZE;
    // Each thread processes 4 elements
    const int num_blocks = (size + block_size * 4 - 1) / (block_size * 4);
    
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
    
    // Launch one block per item
    weighted_scatter_add_kernel<<<num_items, BLOCK_SIZE>>>(
        out.data_ptr<float>(),
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        token_indices.data_ptr<long>(),
        num_items,
        hidden_size
    );
}

torch::Tensor fused_expert_mlp_first_half_hip(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight
) {
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    int intermediate_size = gate_weight.size(0);
    
    auto output = torch::empty({batch_size, intermediate_size}, input.options());
    
    const int block_size = BLOCK_SIZE;
    dim3 grid(batch_size, (intermediate_size + block_size - 1) / block_size);
    
    fused_expert_mlp_first_half_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        gate_weight.data_ptr<float>(),
        up_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        intermediate_size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
void weighted_scatter_add_hip(torch::Tensor out, torch::Tensor expert_out, torch::Tensor weights, torch::Tensor token_indices);
torch::Tensor fused_expert_mlp_first_half_hip(torch::Tensor input, torch::Tensor gate_weight, torch::Tensor up_weight);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v4",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip", "fused_expert_mlp_first_half_hip"],
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
    Optimized MoE with fused expert computation kernels.
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
        
        # Process each expert
        for expert_id in range(self.n_routed_experts):
            mask = (topk_idx == expert_id)
            
            if not mask.any():
                continue
            
            token_ids, slot_ids = torch.where(mask)
            expert_input = hidden_states[token_ids]
            expert_weights = topk_weight[token_ids, slot_ids]
            
            # Use F.linear for matrix ops (cuBLAS optimized)
            # Gate and Up projections
            gate_out = F.linear(expert_input, self.gate_proj[expert_id])
            up_out = F.linear(expert_input, self.up_proj[expert_id])
            
            # Fused SiLU * mul kernel
            intermediate = self.fused_ops.fused_silu_mul_hip(
                gate_out.contiguous(), 
                up_out.contiguous()
            )
            
            # Down projection
            expert_out = F.linear(intermediate, self.down_proj[expert_id])
            
            # Optimized weighted scatter add
            self.fused_ops.weighted_scatter_add_hip(
                y, 
                expert_out.contiguous(), 
                expert_weights.contiguous(), 
                token_ids.contiguous()
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
