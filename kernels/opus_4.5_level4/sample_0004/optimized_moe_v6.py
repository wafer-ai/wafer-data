import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernels optimized for MI300X
fused_moe_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused SiLU * up with high occupancy
__global__ __launch_bounds__(1024)
void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        // Fast approximation of SiLU
        float sigmoid_g = __frcp_rn(1.0f + __expf(-g));
        out[idx] = g * sigmoid_g * up[idx];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_silu_mul_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Expert forward pass: computes intermediate = silu(x @ gate.T) * (x @ up.T), then x @ down.T
// with weighted accumulation to output
__global__ __launch_bounds__(256)
void weighted_accumulate_kernel(
    const float* __restrict__ expert_out,  // (N, hidden)
    const float* __restrict__ weights,     // (N,)
    const long* __restrict__ dst_indices,  // (N,)
    float* __restrict__ output,            // (total_tokens, hidden)
    int N,
    int hidden_size
) {
    int token_idx = blockIdx.x;
    if (token_idx >= N) return;
    
    float w = weights[token_idx];
    long dst_idx = dst_indices[token_idx];
    
    const float* src_row = expert_out + token_idx * hidden_size;
    float* dst_row = output + dst_idx * hidden_size;
    
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        atomicAdd(&dst_row[h], src_row[h] * w);
    }
}

void weighted_accumulate_hip(
    torch::Tensor expert_out,
    torch::Tensor weights,
    torch::Tensor dst_indices,
    torch::Tensor output
) {
    int N = expert_out.size(0);
    int hidden_size = expert_out.size(1);
    
    if (N == 0) return;
    
    weighted_accumulate_kernel<<<N, 256>>>(
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        dst_indices.data_ptr<long>(),
        output.data_ptr<float>(),
        N,
        hidden_size
    );
}
"""

fused_ops = load_inline(
    name="fused_moe_ops_v6",
    cpp_sources="""
    torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
    void weighted_accumulate_hip(torch::Tensor expert_out, torch::Tensor weights, torch::Tensor dst_indices, torch::Tensor output);
    """,
    cuda_sources=fused_moe_source,
    functions=["fused_silu_mul_hip", "weighted_accumulate_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "-munsafe-fp-atomics"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE with sorted token processing.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        # Store weights contiguously for better cache utilization
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )
        
        self.fused_ops = fused_ops

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        device = x.device
        dtype = x.dtype

        x_flat = x.view(-1, self.hidden_size)  # (B*S, H)
        num_tokens = x_flat.shape[0]

        output = torch.zeros(num_tokens, self.hidden_size, device=device, dtype=dtype)

        # Flatten the expert selections
        # expert_indices: (B, S, K) -> (B*S*K,)
        # expert_weights: (B, S, K) -> (B*S*K,)
        expert_indices_flat = expert_indices.view(-1)  # (B*S*K,)
        expert_weights_flat = expert_weights.view(-1)  # (B*S*K,)
        
        # Create token indices for each expert selection
        # Token i with top_k=2 generates token indices [i, i] for its 2 experts
        token_indices_expanded = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        
        # Sort by expert for better batching
        sorted_expert_indices, sort_order = torch.sort(expert_indices_flat)
        sorted_token_indices = token_indices_expanded[sort_order]
        sorted_weights = expert_weights_flat[sort_order]
        
        # Find boundaries between experts
        expert_counts = torch.bincount(sorted_expert_indices.int(), minlength=self.num_experts)
        expert_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), expert_counts.cumsum(0)])
        
        # Process each expert's tokens
        for expert_idx in range(self.num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            count = end - start
            
            if count == 0:
                continue
            
            # Get this expert's tokens
            token_idx = sorted_token_indices[start:end]
            weights = sorted_weights[start:end]
            
            expert_input = x_flat[token_idx]  # (count, hidden)
            
            # Gate and up projections
            gate = torch.mm(expert_input, self.gate_proj[expert_idx].t())
            up = torch.mm(expert_input, self.up_proj[expert_idx].t())
            
            # Fused SiLU + multiply
            intermediate = self.fused_ops.fused_silu_mul_hip(gate, up)
            
            # Down projection
            expert_output = torch.mm(intermediate, self.down_proj[expert_idx].t())
            
            # Weighted accumulation
            self.fused_ops.weighted_accumulate_hip(
                expert_output.contiguous(),
                weights.contiguous(),
                token_idx.contiguous(),
                output
            )

        return output.view(batch, seq_len, self.hidden_size)


def get_inputs():
    batch_size = 4
    seq_len = 2048
    hidden_size = 4096
    num_experts = 8
    top_k = 2

    x = torch.randn(batch_size, seq_len, hidden_size)
    expert_indices = torch.stack([
        torch.randperm(num_experts)[:top_k]
        for _ in range(batch_size * seq_len)
    ]).view(batch_size, seq_len, top_k)
    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)

    return [x, expert_indices, expert_weights]


def get_init_inputs():
    hidden_size = 4096
    intermediate_size = 14336
    num_experts = 8
    return [hidden_size, intermediate_size, num_experts]
