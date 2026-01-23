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

// Fused SiLU (x * sigmoid(x)) * y kernel with vectorized loads
__global__ void fused_silu_mul_kernel_vec4(
    const float4* __restrict__ gate,
    const float4* __restrict__ up,
    float4* __restrict__ out,
    int size4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        float4 g = gate[idx];
        float4 u = up[idx];
        
        // SiLU: x * sigmoid(x)
        float4 result;
        result.x = g.x * (1.0f / (1.0f + expf(-g.x))) * u.x;
        result.y = g.y * (1.0f / (1.0f + expf(-g.y))) * u.y;
        result.z = g.z * (1.0f / (1.0f + expf(-g.z))) * u.z;
        result.w = g.w * (1.0f / (1.0f + expf(-g.w))) * u.w;
        
        out[idx] = result;
    }
}

__global__ void fused_silu_mul_kernel_scalar(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int start,
    int size
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        out[idx] = g * (1.0f / (1.0f + expf(-g))) * up[idx];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = 256;
    
    // Use vectorized loads for aligned data
    int size4 = size / 4;
    int remainder = size % 4;
    
    if (size4 > 0) {
        int num_blocks = (size4 + block_size - 1) / block_size;
        fused_silu_mul_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(gate.data_ptr<float>()),
            reinterpret_cast<const float4*>(up.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            size4
        );
    }
    
    // Handle remainder
    if (remainder > 0) {
        int start = size4 * 4;
        fused_silu_mul_kernel_scalar<<<1, remainder>>>(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            out.data_ptr<float>(),
            start,
            size
        );
    }
    
    return out;
}

// Fused MLP kernel: computes gate(x) and up(x) projections together
// Then applies SiLU * up in one pass
// Input: tokens (batch, hidden), gate_w (intermediate, hidden), up_w (intermediate, hidden)
// Output: (batch, intermediate)
__global__ void fused_gate_up_silu_kernel(
    const float* __restrict__ tokens,      // (n_tokens, hidden)
    const float* __restrict__ gate_w,      // (intermediate, hidden)
    const float* __restrict__ up_w,        // (intermediate, hidden)
    float* __restrict__ out,               // (n_tokens, intermediate)
    int n_tokens,
    int hidden_size,
    int intermediate_size
) {
    int token_idx = blockIdx.x;
    int inter_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (token_idx < n_tokens && inter_idx < intermediate_size) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        
        const float* token_ptr = tokens + token_idx * hidden_size;
        const float* gate_row = gate_w + inter_idx * hidden_size;
        const float* up_row = up_w + inter_idx * hidden_size;
        
        // Dot product with loop unrolling
        int h = 0;
        for (; h + 3 < hidden_size; h += 4) {
            float t0 = token_ptr[h];
            float t1 = token_ptr[h + 1];
            float t2 = token_ptr[h + 2];
            float t3 = token_ptr[h + 3];
            
            gate_sum += t0 * gate_row[h] + t1 * gate_row[h + 1] + 
                        t2 * gate_row[h + 2] + t3 * gate_row[h + 3];
            up_sum += t0 * up_row[h] + t1 * up_row[h + 1] + 
                      t2 * up_row[h + 2] + t3 * up_row[h + 3];
        }
        for (; h < hidden_size; h++) {
            float t = token_ptr[h];
            gate_sum += t * gate_row[h];
            up_sum += t * up_row[h];
        }
        
        // SiLU(gate) * up
        float silu_gate = gate_sum * (1.0f / (1.0f + expf(-gate_sum)));
        out[token_idx * intermediate_size + inter_idx] = silu_gate * up_sum;
    }
}

torch::Tensor fused_gate_up_silu_hip(
    torch::Tensor tokens,
    torch::Tensor gate_w,
    torch::Tensor up_w
) {
    int n_tokens = tokens.size(0);
    int hidden_size = tokens.size(1);
    int intermediate_size = gate_w.size(0);
    
    auto out = torch::empty({n_tokens, intermediate_size}, tokens.options());
    
    const int block_size = 256;
    dim3 blocks(n_tokens, (intermediate_size + block_size - 1) / block_size);
    
    fused_gate_up_silu_kernel<<<blocks, block_size>>>(
        tokens.data_ptr<float>(),
        gate_w.data_ptr<float>(),
        up_w.data_ptr<float>(),
        out.data_ptr<float>(),
        n_tokens,
        hidden_size,
        intermediate_size
    );
    
    return out;
}
"""

fused_ops = load_inline(
    name="fused_moe_ops_v3",
    cpp_sources=hip_source,
    functions=["fused_silu_mul_hip", "fused_gate_up_silu_hip"],
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

        # Memory-efficient expert computation
        output = torch.zeros(num_tokens, self.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for expert_idx in range(self.n_routed_experts):
            mask = (topk_idx == expert_idx)
            
            if not mask.any():
                continue
            
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
                
            expert_tokens = hidden_states[token_indices]
            expert_weights = (mask[token_indices].float() * topk_weight[token_indices]).sum(dim=1)
            
            # Use fused gate + up + silu kernel
            intermediate = self.fused_ops.fused_gate_up_silu_hip(
                expert_tokens.contiguous(),
                self.gate_proj[expert_idx].contiguous(),
                self.up_proj[expert_idx].contiguous()
            )
            
            expert_out = F.linear(intermediate, self.down_proj[expert_idx])
            
            output.index_add_(0, token_indices, expert_out * expert_weights.unsqueeze(1))
        
        y = output.view(*orig_shape)

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
