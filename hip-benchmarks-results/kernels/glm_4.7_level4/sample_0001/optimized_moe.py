import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# Fused kernel for MoE expert computation
# Each token is processed by multiple threads in parallel
moe_expert_fused_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void moe_expert_fused_kernel(
    const float* hidden_states,       // (num_tokens, hidden_size)
    const int64_t* topk_idx,         // (num_tokens, top_k) expert indices
    const float* gate_weight,         // (n_experts, intermediate, hidden)
    const float* up_weight,           // (n_experts, intermediate, hidden)
    const float* down_weight,         // (n_experts, hidden, intermediate)
    const float* topk_weight,         // (num_tokens, top_k)
    float* output,                    // (num_tokens, hidden_size)
    int num_tokens,
    int top_k,
    int hidden_size,
    int intermediate_size,
    int n_experts
) {
    // Each thread block processes one token
    int token = blockIdx.x;
    if (token >= num_tokens) return;
    
    // Each thread in the block could process multiple hidden dimensions
    // Global thread index
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Each thread processes hidden_size / num_threads elements
    for (int h_start = tid; h_start < hidden_size; h_start += num_threads) {
        float token_out = 0.0f;
        
        // Process each expert for this token
        for (int k = 0; k < top_k; k++) {
            int expert_idx = static_cast<int>(topk_idx[token * top_k + k]);
            float weight = topk_weight[token * top_k + k];
            
            // Get pointers to this expert's weights
            const float* this_gate = gate_weight + expert_idx * intermediate_size * hidden_size;
            const float* this_up = up_weight + expert_idx * intermediate_size * hidden_size;
            const float* this_down = down_weight + expert_idx * hidden_size * intermediate_size;
            const float* x = hidden_states + token * hidden_size;
            
            // Compute contribution to output[h_start]:
            float sum = 0.0f;
            
            for (int inter = 0; inter < intermediate_size; inter++) {
                // gate[inter] @ x
                float gate_val = 0.0f;
                for (int j = 0; j < hidden_size; j++) {
                    gate_val += this_gate[inter * hidden_size + j] * x[j];
                }
                
                // up[inter] @ x
                float up_val = 0.0f;
                for (int j = 0; j < hidden_size; j++) {
                    up_val += this_up[inter * hidden_size + j] * x[j];
                }
                
                // SiLU activation: x / (1 + exp(-x))
                float silu_val = gate_val / (1.0f + expf(-gate_val));
                
                // Element-wise multiply
                float intermediate_val = silu_val * up_val;
                
                // Down projection for hidden dimension h_start
                sum += this_down[h_start * intermediate_size + inter] * intermediate_val;
            }
            
            token_out += weight * sum;
        }
        
        output[token * hidden_size + h_start] = token_out;
    }
}

torch::Tensor moe_expert_fused_hip(
    torch::Tensor hidden_states,
    torch::Tensor topk_idx,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight,
    torch::Tensor topk_weight
) {
    auto num_tokens = hidden_states.size(0);
    auto hidden_size = hidden_states.size(1);
    auto top_k = topk_idx.size(1);
    auto intermediate_size = gate_weight.size(1);
    auto n_experts = gate_weight.size(0);
    
    auto output = torch::zeros_like(hidden_states);
    
    // Use 256 threads per block for efficiency
    const int block_size = 256;
    
    dim3 grid(num_tokens);
    dim3 block(block_size);
    
    moe_expert_fused_kernel<<<grid, block>>>(
        hidden_states.data_ptr<float>(),
        topk_idx.data_ptr<int64_t>(),
        gate_weight.data_ptr<float>(),
        up_weight.data_ptr<float>(),
        down_weight.data_ptr<float>(),
        topk_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        num_tokens,
        top_k,
        hidden_size,
        intermediate_size,
        n_experts
    );
    
    return output;
}
"""

moe_expert_fused = load_inline(
    name="moe_expert_fused",
    cpp_sources=moe_expert_fused_cpp_source,
    functions=["moe_expert_fused_hip"],
    verbose=True,
)


class MoEGate(nn.Module):
    """
    DeepSeek-V3 MoE gating with grouped expert selection.
    """

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
    DeepSeek-V3 Mixture of Experts Layer - Optimized with fused HIP kernel
    Avoids large memory allocations
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
        assert not self.training

        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)

        # Use fused kernel to compute expert outputs
        expert_out = moe_expert_fused.moe_expert_fused_hip(
            hidden_states_flat,
            topk_idx,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            topk_weight,
        )

        expert_out = expert_out.view(*orig_shape)

        if self.shared_gate_proj is not None:
            shared_out = self.shared_down_proj(
                F.silu(self.shared_gate_proj(identity)) * self.shared_up_proj(identity)
            )
            expert_out = expert_out + shared_out

        return expert_out