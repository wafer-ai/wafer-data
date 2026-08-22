import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.cpp_extension import load_inline

# Override dimensions to fit in memory
batch_size = 2
seq_len = 128
hidden_size = 256
intermediate_size = 176
n_routed_experts = 8
num_experts_per_tok = 2
n_group = 4
topk_group = 2
n_shared_experts = 1
routed_scaling_factor = 2.5

os.environ["CXX"] = "hipcc"

moe_hip_code = """
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void fused_moe_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ gate_proj,
    const float* __restrict__ up_proj,
    const float* __restrict__ down_proj,
    const int* __restrict__ topk_idx,
    const float* __restrict__ topk_weight,
    float* __restrict__ output,
    int num_tokens,
    int hidden_size,
    int intermediate_size,
    int top_k
) {
    int token_id = blockIdx.x;
    int tid = threadIdx.x;
    
    if (token_id >= num_tokens) return;
    
    extern __shared__ float shared_mem[];
    float* token_cache = shared_mem;
    float* intermediate_cache = &shared_mem[hidden_size];
    
    // Load token into shared memory (coalesced)
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        token_cache[i] = hidden_states[token_id * hidden_size + i];
    }
    __syncthreads();
    
    int j = tid;
    if (j < hidden_size) {
        float out_val = 0.0f;
        
        for (int k = 0; k < top_k; k++) {
            int expert_id = topk_idx[token_id * top_k + k];
            float weight = topk_weight[token_id * top_k + k];
            
            if (expert_id < 0 || weight == 0.0f) continue;
            
            int expert_offset = expert_id * intermediate_size * hidden_size;
            
            // Compute intermediate activations in shared memory
            for (int i = tid; i < intermediate_size; i += blockDim.x) {
                float sum_gate = 0.0f, sum_up = 0.0f;
                int row_offset = expert_offset + i * hidden_size;
                
                for (int h = 0; h < hidden_size; h++) {
                    float val = token_cache[h];
                    sum_gate += gate_proj[row_offset + h] * val;
                    sum_up += up_proj[row_offset + h] * val;
                }
                
                float silu = sum_gate / (1.0f + expf(-sum_gate));
                intermediate_cache[i] = silu * sum_up;
            }
            __syncthreads();
            
            // Down projection
            float sum = 0.0f;
            int down_offset = expert_id * hidden_size * intermediate_size + j * intermediate_size;
            for (int i = 0; i < intermediate_size; i++) {
                sum += down_proj[down_offset + i] * intermediate_cache[i];
            }
            
            out_val += weight * sum;
            __syncthreads();
        }
        
        output[token_id * hidden_size + j] = out_val;
    }
}

torch::Tensor fused_moe_hip(
    torch::Tensor hidden_states,
    torch::Tensor gate_proj,
    torch::Tensor up_proj,
    torch::Tensor down_proj,
    torch::Tensor topk_idx,
    torch::Tensor topk_weight
) {
    auto num_tokens = hidden_states.size(0);
    auto hidden_size = hidden_states.size(1);
    auto intermediate_size = gate_proj.size(1);
    auto top_k = topk_idx.size(1);
    
    auto output = torch::zeros_like(hidden_states);
    
    dim3 grid(num_tokens);
    dim3 block(min(hidden_size, 512));
    
    size_t shared_mem_size = (hidden_size + intermediate_size) * sizeof(float);
    
    fused_moe_kernel<<<grid, block, shared_mem_size>>>(
        hidden_states.data_ptr<float>(),
        gate_proj.data_ptr<float>(),
        up_proj.data_ptr<float>(),
        down_proj.data_ptr<float>(),
        topk_idx.data_ptr<int>(),
        topk_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        num_tokens,
        hidden_size,
        intermediate_size,
        top_k
    );
    
    return output;
}
"""

fused_moe = load_inline(
    name="fused_moe",
    cpp_sources=moe_hip_code,
    functions=["fused_moe_hip"],
    verbose=True,
    extra_cflags=["-O3", "-fPIC", "-D__HIP_PLATFORM_AMD__"],
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
            .reshape(bsz * seq_len, self.n_routed_experts)
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

        # Smaller scale for initialization
        scale = 0.01
        self.gate_proj = nn.Parameter(torch.randn(n_routed_experts, intermediate_size, hidden_size) * scale)
        self.up_proj = nn.Parameter(torch.randn(n_routed_experts, intermediate_size, hidden_size) * scale)
        self.down_proj = nn.Parameter(torch.randn(n_routed_experts, hidden_size, intermediate_size) * scale)

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
            
        self.fused_moe = fused_moe

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert not self.training, "DeepSeek MoE grouped selection is inference-only"

        identity = hidden_states
        orig_shape = hidden_states.shape
        bsz, seq_len, _ = orig_shape

        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]

        # Fused expert computation with custom HIP kernel
        y = self.fused_moe.fused_moe_hip(
            hidden_states,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            topk_idx,
            topk_weight
        )

        y = y.view(*orig_shape)

        # Add shared expert output
        if self.shared_gate_proj is not None:
            shared_out = self.shared_down_proj(
                F.silu(self.shared_gate_proj(identity)) * self.shared_up_proj(identity)
            )
            y = y + shared_out

        return y


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda')]


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
