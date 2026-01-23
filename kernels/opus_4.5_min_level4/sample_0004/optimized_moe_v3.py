import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused SiLU + elementwise multiply kernel
fused_silu_mul_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// Vectorized SiLU * mul kernel with grid-stride loop
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time when possible
    for (int i = idx * 4; i + 3 < size; i += stride * 4) {
        float4 g = *reinterpret_cast<const float4*>(gate + i);
        float4 u = *reinterpret_cast<const float4*>(up + i);
        float4 result;
        result.x = fast_silu(g.x) * u.x;
        result.y = fast_silu(g.y) * u.y;
        result.z = fast_silu(g.z) * u.z;
        result.w = fast_silu(g.w) * u.w;
        *reinterpret_cast<float4*>(out + i) = result;
    }
    
    // Handle remaining elements
    int remaining_start = (size / 4) * 4;
    for (int i = remaining_start + idx; i < size; i += stride) {
        out[i] = fast_silu(gate[i]) * up[i];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "Inputs must be contiguous");
    
    int size = gate.numel();
    auto out = torch::empty_like(gate);
    
    int block_size = 256;
    int num_blocks = std::min((size + block_size * 4 - 1) / (block_size * 4), 1024);
    
    fused_silu_mul_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Batched version for processing multiple expert groups
__global__ void fused_silu_mul_batched_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int* __restrict__ expert_offsets,  // Start offset for each expert
    const int* __restrict__ expert_counts,   // Number of tokens per expert
    int intermediate_size,
    int num_experts
) {
    int expert_idx = blockIdx.y;
    int offset = expert_offsets[expert_idx];
    int count = expert_counts[expert_idx];
    
    if (count == 0) return;
    
    int total_elements = count * intermediate_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const float* gate_ptr = gate + offset * intermediate_size;
    const float* up_ptr = up + offset * intermediate_size;
    float* out_ptr = out + offset * intermediate_size;
    
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        out_ptr[i] = fast_silu(gate_ptr[i]) * up_ptr[i];
    }
}
"""

fused_ops_cpp = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v3",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_silu_mul_source,
    functions=["fused_silu_mul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE with batched processing per expert.
    Key optimization: Pre-sort tokens by expert to enable larger batch GEMMs.
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
        self.fused_ops = fused_ops

        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        
        x_flat = x.view(-1, self.hidden_size)  # (N, H)
        num_tokens = x_flat.shape[0]
        
        # Flatten indices and weights
        # expert_indices: (batch, seq, top_k) -> (N * top_k)
        expert_indices_flat = expert_indices.view(-1)  # (N * top_k)
        expert_weights_flat = expert_weights.view(-1)  # (N * top_k)
        
        # Create token indices for each assignment
        # token_ids[i] tells us which token assignment i belongs to
        token_ids = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        
        # Sort by expert for efficient batched processing
        sorted_expert_indices, sort_order = expert_indices_flat.sort()
        sorted_token_ids = token_ids[sort_order]
        sorted_weights = expert_weights_flat[sort_order]
        
        # Compute expert boundaries
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        for i in range(self.num_experts):
            expert_counts[i] = (sorted_expert_indices == i).sum()
        
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x.device)
        expert_offsets[1:] = expert_counts.cumsum(0)
        
        # Gather tokens in expert order
        sorted_x = x_flat[sorted_token_ids]  # (N * top_k, H)
        
        # Process all assignments
        total_assignments = num_tokens * top_k
        
        # Allocate buffers for intermediate results
        gate_results = torch.empty(total_assignments, self.intermediate_size, device=x.device, dtype=x.dtype)
        up_results = torch.empty(total_assignments, self.intermediate_size, device=x.device, dtype=x.dtype)
        
        # Process each expert's tokens with batched GEMM
        for expert_idx in range(self.num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            
            if start == end:
                continue
            
            expert_x = sorted_x[start:end]  # (count, H)
            
            # Batched GEMMs for this expert
            gate_results[start:end] = torch.mm(expert_x, self.gate_proj[expert_idx].t())
            up_results[start:end] = torch.mm(expert_x, self.up_proj[expert_idx].t())
        
        # Fused SiLU * up for all experts at once
        intermediate = self.fused_ops.fused_silu_mul_hip(
            gate_results.contiguous(), 
            up_results.contiguous()
        )
        
        # Down projection for each expert
        down_results = torch.empty(total_assignments, self.hidden_size, device=x.device, dtype=x.dtype)
        
        for expert_idx in range(self.num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            
            if start == end:
                continue
            
            down_results[start:end] = torch.mm(intermediate[start:end], self.down_proj[expert_idx].t())
        
        # Apply weights and scatter back
        weighted_results = down_results * sorted_weights.unsqueeze(-1)
        
        # Scatter add back to original token positions
        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)
        output.index_add_(0, sorted_token_ids, weighted_results)
        
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
