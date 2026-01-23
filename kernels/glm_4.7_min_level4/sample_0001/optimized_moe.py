import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_expert_cpp_source = """
#include <hip/hip_runtime.h>

template <typename T>
__device__ __forceinline__ T silu(T x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__global__ void fused_expert_mlp_kernel(
    const float* gate_weights,
    const float* up_weights,
    const float* down_weights,
    const float* expanded_tokens,
    const int32_t* topk_idx_flat,
    const float* topk_weights,
    float* output,
    int num_tokens_original,
    int top_k,
    int hidden_size,
    int intermediate_size,
    int n_experts
) {
    // Use dynamic shared memory - partition it properly
    extern __shared__ char shared_data[];
    
    float* shared_gate = reinterpret_cast<float*>(shared_data);
    float* shared_up = shared_gate + static_cast<size_t>(intermediate_size) * hidden_size;
    float* shared_down = shared_up + static_cast<size_t>(intermediate_size) * hidden_size;
    float* shared_intermediate = shared_down + static_cast<size_t>(hidden_size) * intermediate_size;
    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    // Map block to token-expert pair
    int global_idx = bx;
    int token_idx = global_idx / top_k;
    int expert_rank = global_idx % top_k;
    
    if (token_idx >= num_tokens_original) return;
    
    // Get expert index and weight
    int expert_idx = topk_idx_flat[global_idx];
    float weight = topk_weights[token_idx * top_k + expert_rank];
    
    // Cooperative load of gate and up weights
    size_t weight_elems = static_cast<size_t>(intermediate_size) * hidden_size;
    
    for (size_t i = tx; i < weight_elems; i += blockDim.x) {
        size_t linear_idx = static_cast<size_t>(expert_idx) * weight_elems + i;
        shared_gate[i] = gate_weights[linear_idx];
        shared_up[i] = up_weights[linear_idx];
    }
    
    // Cooperative load of down weights
    for (size_t i = tx; i < weight_elems; i += blockDim.x) {
        size_t linear_idx = static_cast<size_t>(expert_idx) * weight_elems + i;
        shared_down[i] = down_weights[linear_idx];
    }
    
    __syncthreads();
    
    // Get input pointer
    const float* input_ptr = expanded_tokens + static_cast<size_t>(global_idx) * hidden_size;
    
    // Each thread computes one intermediate element
    int inter_elem = tx;
    if (inter_elem < intermediate_size) {
        float gate_val = 0.0f;
        float up_val = 0.0f;
        
        for (int j = 0; j < hidden_size; j++) {
            gate_val += shared_gate[static_cast<size_t>(inter_elem) * hidden_size + j] * input_ptr[j];
            up_val += shared_up[static_cast<size_t>(inter_elem) * hidden_size + j] * input_ptr[j];
        }
        
        shared_intermediate[inter_elem] = silu(gate_val) * up_val;
    }
    
    __syncthreads();
    
    // Each thread computes one output element
    int out_elem = tx;
    if (out_elem < hidden_size) {
        float down_val = 0.0f;
        
        for (int i = 0; i < intermediate_size; i++) {
            down_val += shared_down[static_cast<size_t>(out_elem) * intermediate_size + i] * shared_intermediate[i];
        }
        
        // Atomic add to output
        size_t out_linear_idx = static_cast<size_t>(token_idx) * hidden_size + out_elem;
        atomicAdd(&output[out_linear_idx], down_val * weight);
    }
}

torch::Tensor fused_expert_mlp_hip(
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    torch::Tensor expanded_tokens,
    torch::Tensor topk_idx_flat,
    torch::Tensor topk_weights,
    int num_tokens_original,
    int top_k
) {
    auto hidden_size = gate_weights.size(2);
    auto intermediate_size = gate_weights.size(1);
    auto n_experts = gate_weights.size(0);
    
    // Initialize output
    auto output = torch::zeros({num_tokens_original, static_cast<long>(hidden_size)}, torch::kFloat32).cuda();
    
    int block_size = 256;
    int num_blocks = topk_idx_flat.size(0);
    
    // Calculate shared memory needed
    size_t gate_size = intermediate_size * hidden_size;
    size_t up_size = gate_size;
    size_t down_size = hidden_size * intermediate_size;
    size_t inter_size = intermediate_size;
    size_t total_shared = (gate_size + up_size + down_size + inter_size) * sizeof(float);
    
    fused_expert_mlp_kernel<<<num_blocks, block_size, total_shared>>>(
        gate_weights.data_ptr<float>(),
        up_weights.data_ptr<float>(),
        down_weights.data_ptr<float>(),
        expanded_tokens.data_ptr<float>(),
        topk_idx_flat.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        output.data_ptr<float>(),
        num_tokens_original,
        top_k,
        hidden_size,
        intermediate_size,
        n_experts
    );
    
    return output;
}
"""

fused_expert_mlp = load_inline(
    name="fused_expert_mlp",
    cpp_sources=fused_expert_cpp_source,
    functions=["fused_expert_mlp_hip"],
    verbose=True,
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

        self.fused_expert_mlp = fused_expert_mlp

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
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states_flat.shape[0]

        flat_topk_idx = topk_idx.view(-1)
        
        expanded_tokens = hidden_states_flat.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1)
        expanded_tokens = expanded_tokens.reshape(-1, self.hidden_size)

        expert_out = self.fused_expert_mlp.fused_expert_mlp_hip(
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            expanded_tokens,
            flat_topk_idx,
            topk_weight,
            num_tokens,
            self.num_experts_per_tok
        )

        y = expert_out.view(*orig_shape)

        if self.shared_gate_proj is not None:
            shared_out = self.shared_down_proj(
                F.silu(self.shared_gate_proj(identity)) * self.shared_up_proj(identity)
            )
            y = y + shared_out

        return y