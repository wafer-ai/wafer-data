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

// Vectorized fused SiLU(gate) * up kernel with float4
__global__ void fused_silu_mul_vec4_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 g = *reinterpret_cast<const float4*>(gate + idx);
        float4 u = *reinterpret_cast<const float4*>(up + idx);
        
        float4 result;
        result.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        result.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        result.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        result.w = (g.w / (1.0f + expf(-g.w))) * u.w;
        
        *reinterpret_cast<float4*>(out + idx) = result;
    } else if (idx < size) {
        // Handle remainder
        for (int i = idx; i < size && i < idx + 4; i++) {
            float gv = gate[i];
            float sigmoid_g = 1.0f / (1.0f + expf(-gv));
            out[i] = gv * sigmoid_g * up[i];
        }
    }
}

// Scatter add kernel with vectorized loads where possible
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

// Vectorized scatter add with float4
__global__ void weighted_scatter_add_vec4_kernel(
    float* __restrict__ out,
    const float* __restrict__ expert_out,
    const float* __restrict__ weights,
    const long* __restrict__ token_indices,
    int num_items,
    int hidden_size
) {
    int item_idx = blockIdx.x;
    int h_idx_base = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    
    if (item_idx < num_items && h_idx_base + 3 < hidden_size) {
        int token_idx = token_indices[item_idx];
        float w = weights[item_idx];
        
        float4 val = *reinterpret_cast<const float4*>(expert_out + item_idx * hidden_size + h_idx_base);
        
        atomicAdd(&out[token_idx * hidden_size + h_idx_base + 0], w * val.x);
        atomicAdd(&out[token_idx * hidden_size + h_idx_base + 1], w * val.y);
        atomicAdd(&out[token_idx * hidden_size + h_idx_base + 2], w * val.z);
        atomicAdd(&out[token_idx * hidden_size + h_idx_base + 3], w * val.w);
    } else if (item_idx < num_items && h_idx_base < hidden_size) {
        int token_idx = token_indices[item_idx];
        float w = weights[item_idx];
        for (int i = h_idx_base; i < hidden_size && i < h_idx_base + 4; i++) {
            float val = expert_out[item_idx * hidden_size + i];
            atomicAdd(&out[token_idx * hidden_size + i], w * val);
        }
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = 256;
    // Each thread handles 4 elements
    const int num_blocks = (size + block_size * 4 - 1) / (block_size * 4);
    
    fused_silu_mul_vec4_kernel<<<num_blocks, block_size>>>(
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
    
    // Use vectorized version if hidden_size is multiple of 4
    if (hidden_size % 4 == 0) {
        int num_vec4 = hidden_size / 4;
        dim3 grid(num_items, (num_vec4 + block_size - 1) / block_size);
        
        weighted_scatter_add_vec4_kernel<<<grid, block_size>>>(
            out.data_ptr<float>(),
            expert_out.data_ptr<float>(),
            weights.data_ptr<float>(),
            token_indices.data_ptr<long>(),
            num_items,
            hidden_size
        );
    } else {
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
}
"""

cpp_source = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
void weighted_scatter_add_hip(torch::Tensor out, torch::Tensor expert_out, torch::Tensor weights, torch::Tensor token_indices);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v3",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
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
    Optimized MoE with vectorized kernels and better memory access patterns.
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
            
            # Expert MLP with fused SiLU*up
            gate_out = F.linear(expert_input, self.gate_proj[expert_id])
            up_out = F.linear(expert_input, self.up_proj[expert_id])
            
            # Fused SiLU * mul kernel
            intermediate = self.fused_ops.fused_silu_mul_hip(
                gate_out.contiguous(), 
                up_out.contiguous()
            )
            
            expert_out = F.linear(intermediate, self.down_proj[expert_id])
            
            # Vectorized weighted scatter add
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
