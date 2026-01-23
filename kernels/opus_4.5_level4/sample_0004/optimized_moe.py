import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused SiLU * up kernel - combines SiLU activation with element-wise multiply
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
        float silu_g = g / (1.0f + expf(-g));  // SiLU = x * sigmoid(x)
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

// Fused weighted scatter-add kernel
__global__ void weighted_scatter_add_kernel(
    const float* __restrict__ src,      // (num_tokens, hidden)
    const float* __restrict__ weights,  // (num_tokens,)
    const int64_t* __restrict__ indices, // (num_tokens,)
    float* __restrict__ dst,            // (total_tokens, hidden)
    int num_tokens,
    int hidden_size
) {
    int token_idx = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx < num_tokens && h < hidden_size) {
        int64_t dst_idx = indices[token_idx];
        float w = weights[token_idx];
        float val = src[token_idx * hidden_size + h] * w;
        atomicAdd(&dst[dst_idx * hidden_size + h], val);
    }
}

void weighted_scatter_add_hip(
    torch::Tensor src,
    torch::Tensor weights,
    torch::Tensor indices,
    torch::Tensor dst
) {
    int num_tokens = src.size(0);
    int hidden_size = src.size(1);
    
    const int block_x = 256;
    int grid_y = (hidden_size + block_x - 1) / block_x;
    dim3 grid(num_tokens, grid_y);
    dim3 block(block_x);
    
    weighted_scatter_add_kernel<<<grid, block>>>(
        src.data_ptr<float>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dst.data_ptr<float>(),
        num_tokens,
        hidden_size
    );
}
"""

fused_ops = load_inline(
    name="fused_moe_ops",
    cpp_sources="""
    torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
    void weighted_scatter_add_hip(torch::Tensor src, torch::Tensor weights, torch::Tensor indices, torch::Tensor dst);
    """,
    cuda_sources=fused_silu_mul_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE Expert with Gated GEMM (SiLU-gated FFN).
    
    Optimizations:
    1. Fused SiLU activation with element-wise multiplication
    2. Batched expert operations where possible
    3. Efficient weighted scatter-add
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

        # Expert weights
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
        x: torch.Tensor,              # (batch, seq_len, hidden_size)
        expert_indices: torch.Tensor, # (batch, seq_len, top_k)
        expert_weights: torch.Tensor, # (batch, seq_len, top_k)
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]

        x_flat = x.view(-1, self.hidden_size)  # (batch * seq_len, hidden)
        num_tokens = x_flat.shape[0]

        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)

        # Pre-compute masks and indices for all experts
        expert_indices_flat = expert_indices.view(-1, top_k)  # (num_tokens, top_k)
        expert_weights_flat = expert_weights.view(-1, top_k)  # (num_tokens, top_k)
        
        for expert_idx in range(self.num_experts):
            # Find which (token, slot) pairs use this expert
            expert_mask = (expert_indices_flat == expert_idx)  # (num_tokens, top_k)
            
            if not expert_mask.any():
                continue
            
            # Get indices
            token_indices, slot_indices = torch.where(expert_mask)
            weights = expert_weights_flat[token_indices, slot_indices]
            
            # Get tokens for this expert
            expert_input = x_flat[token_indices]  # (num_selected, hidden)
            
            # Compute gate and up projections
            gate = F.linear(expert_input, self.gate_proj[expert_idx])
            up = F.linear(expert_input, self.up_proj[expert_idx])
            
            # Fused SiLU + multiply
            intermediate = self.fused_ops.fused_silu_mul_hip(gate, up)
            
            # Down projection
            expert_output = F.linear(intermediate, self.down_proj[expert_idx])
            
            # Weighted scatter add using custom kernel
            self.fused_ops.weighted_scatter_add_hip(
                expert_output.contiguous(),
                weights.contiguous(),
                token_indices.contiguous(),
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
