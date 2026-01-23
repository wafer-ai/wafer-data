import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized MoE with combined weight approach
fused_moe_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// High-performance fused SiLU*up kernel optimized for large tensors
__global__ void fused_silu_mul_large_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t size
) {
    int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = int64_t(blockDim.x) * gridDim.x;
    
    // Process 4 elements per iteration for better memory bandwidth
    for (int64_t i = idx * 4; i < size - 3; i += stride * 4) {
        float g0 = gate[i], g1 = gate[i+1], g2 = gate[i+2], g3 = gate[i+3];
        float u0 = up[i], u1 = up[i+1], u2 = up[i+2], u3 = up[i+3];
        
        out[i]   = (g0 / (1.0f + __expf(-g0))) * u0;
        out[i+1] = (g1 / (1.0f + __expf(-g1))) * u1;
        out[i+2] = (g2 / (1.0f + __expf(-g2))) * u2;
        out[i+3] = (g3 / (1.0f + __expf(-g3))) * u3;
    }
    
    // Handle remainder
    int64_t remainder_start = ((size / 4) / stride * stride) * 4;
    for (int64_t i = remainder_start + idx; i < size; i += stride) {
        float g = gate[i];
        out[i] = (g / (1.0f + __expf(-g))) * up[i];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = 256;
    const int num_blocks = min((int)((size + block_size * 4 - 1) / (block_size * 4)), 2048);
    
    fused_silu_mul_large_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Optimized weighted scatter-add with better memory coalescing
__global__ void weighted_scatter_kernel(
    const float* __restrict__ src,
    const float* __restrict__ weights,
    const int64_t* __restrict__ indices,
    float* __restrict__ dst,
    int num_tokens,
    int hidden_size
) {
    // 2D grid: x = hidden dimension chunks, y = tokens
    int token_idx = blockIdx.y;
    if (token_idx >= num_tokens) return;
    
    float w = weights[token_idx];
    int64_t dst_idx = indices[token_idx];
    
    int h_start = blockIdx.x * blockDim.x + threadIdx.x;
    int h_stride = gridDim.x * blockDim.x;
    
    for (int h = h_start; h < hidden_size; h += h_stride) {
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
    
    const int block_size = 256;
    int grid_x = (hidden_size + block_size - 1) / block_size;
    grid_x = min(grid_x, 16);  // Limit grid size
    
    dim3 grid(grid_x, num_tokens);
    dim3 block(block_size);
    
    weighted_scatter_kernel<<<grid, block>>>(
        src.data_ptr<float>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dst.data_ptr<float>(),
        num_tokens,
        hidden_size
    );
}

// Combined gate+up linear with fused activation
// Computes: out = SiLU(x @ gate_W.T) * (x @ up_W.T)
// This is memory-bound, so fusing reads x once
torch::Tensor fused_gate_up_forward(
    torch::Tensor x,           // (N, hidden)
    torch::Tensor gate_weight, // (intermediate, hidden)
    torch::Tensor up_weight    // (intermediate, hidden)
) {
    // Use torch's efficient mm and our fused silu_mul
    auto gate = torch::mm(x, gate_weight.t());
    auto up = torch::mm(x, up_weight.t());
    
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = 256;
    const int num_blocks = min((int)((size + block_size * 4 - 1) / (block_size * 4)), 2048);
    
    fused_silu_mul_large_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}
"""

fused_ops = load_inline(
    name="fused_moe_ops_v5",
    cpp_sources="""
    torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
    void weighted_scatter_add_hip(torch::Tensor src, torch::Tensor weights, torch::Tensor indices, torch::Tensor dst);
    torch::Tensor fused_gate_up_forward(torch::Tensor x, torch::Tensor gate_weight, torch::Tensor up_weight);
    """,
    cuda_sources=fused_moe_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip", "fused_gate_up_forward"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE Expert with Gated GEMM.
    
    Key optimizations:
    1. Combined gate+up projection with fused silu_mul
    2. 2D grid scatter-add for better memory parallelism
    3. Unrolled kernel loops
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

        expert_indices_flat = expert_indices.view(-1, top_k)
        expert_weights_flat = expert_weights.view(-1, top_k)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)
            
            if not expert_mask.any():
                continue
            
            token_indices, slot_indices = torch.where(expert_mask)
            weights = expert_weights_flat[token_indices, slot_indices]
            
            expert_input = x_flat[token_indices]
            
            # Fused gate+up+silu+mul 
            intermediate = self.fused_ops.fused_gate_up_forward(
                expert_input,
                self.gate_proj[expert_idx],
                self.up_proj[expert_idx]
            )
            
            # Down projection
            expert_output = torch.mm(intermediate, self.down_proj[expert_idx].t())
            
            # Weighted scatter-add
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
