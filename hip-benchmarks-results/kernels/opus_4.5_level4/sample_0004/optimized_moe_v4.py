import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernels for MoE
fused_moe_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused SiLU * up kernel with high throughput vectorization
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float g = gate[i];
        float silu_g = g / (1.0f + __expf(-g));
        out[i] = silu_g * up[i];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = 512;
    const int num_blocks = min((size + block_size - 1) / block_size, 1024);
    
    fused_silu_mul_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Batched version: process multiple experts' silu_mul at once
__global__ void batched_silu_mul_kernel(
    const float* __restrict__ gate,     // (total_tokens, intermediate)
    const float* __restrict__ up,       // (total_tokens, intermediate)
    float* __restrict__ out,            // (total_tokens, intermediate)
    const int64_t* __restrict__ offsets, // (num_experts+1,) cumulative token counts
    int num_experts,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = offsets[num_experts] * intermediate_size;
    
    if (idx < total_size) {
        float g = gate[idx];
        float silu_g = g / (1.0f + __expf(-g));
        out[idx] = silu_g * up[idx];
    }
}

// Efficient weighted scatter-add with coalesced memory access
__global__ void weighted_scatter_add_kernel(
    const float* __restrict__ src,
    const float* __restrict__ weights,
    const int64_t* __restrict__ indices,
    float* __restrict__ dst,
    int num_tokens,
    int hidden_size
) {
    int token_idx = blockIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    float w = weights[token_idx];
    int64_t dst_idx = indices[token_idx];
    
    // Each thread handles multiple elements
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
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
    
    if (num_tokens == 0) return;
    
    const int block_size = 512;
    
    weighted_scatter_add_kernel<<<num_tokens, block_size>>>(
        src.data_ptr<float>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dst.data_ptr<float>(),
        num_tokens,
        hidden_size
    );
}

// Grouped operations for better efficiency
struct ExpertTokenInfo {
    int start;
    int count;
};
"""

fused_ops = load_inline(
    name="fused_moe_ops_v4",
    cpp_sources="""
    torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
    void weighted_scatter_add_hip(torch::Tensor src, torch::Tensor weights, torch::Tensor indices, torch::Tensor dst);
    """,
    cuda_sources=fused_moe_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE Expert with Gated GEMM.
    
    Key optimizations:
    1. Sorted token processing for contiguous memory access
    2. Fused SiLU + multiply kernel
    3. Efficient weighted scatter-add
    4. Minimized synchronization overhead
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

        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]

        output = torch.zeros(num_tokens, self.hidden_size, device=device, dtype=dtype)

        # Flatten expert indices and weights
        expert_indices_flat = expert_indices.view(-1, top_k)  # (num_tokens, top_k)
        expert_weights_flat = expert_weights.view(-1, top_k)  # (num_tokens, top_k)
        
        # Pre-compute masks for all experts to avoid repeated torch.where calls
        # Sort tokens by expert for better cache utilization
        all_expert_data = []
        
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)
            if expert_mask.any():
                token_indices, slot_indices = torch.where(expert_mask)
                weights = expert_weights_flat[token_indices, slot_indices]
                all_expert_data.append((expert_idx, token_indices, weights))
        
        # Process each expert
        for expert_idx, token_indices, weights in all_expert_data:
            # Gather tokens for this expert
            expert_input = x_flat[token_indices]  # (num_selected, hidden)
            
            # Gate projection: (num_selected, hidden) @ (hidden, intermediate) -> (num_selected, intermediate)
            gate = torch.mm(expert_input, self.gate_proj[expert_idx].t())
            
            # Up projection
            up = torch.mm(expert_input, self.up_proj[expert_idx].t())
            
            # Fused SiLU + element-wise multiply
            intermediate = self.fused_ops.fused_silu_mul_hip(gate, up)
            
            # Down projection
            expert_output = torch.mm(intermediate, self.down_proj[expert_idx].t())
            
            # Weighted scatter-add back to output
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
