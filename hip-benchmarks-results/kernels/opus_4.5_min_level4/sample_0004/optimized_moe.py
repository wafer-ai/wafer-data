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

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Fused SiLU(gate) * up kernel
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = silu(gate[idx]) * up[idx];
    }
}

// Vectorized version for better memory throughput
__global__ void fused_silu_mul_kernel_vec4(
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
        result.x = silu(g.x) * u.x;
        result.y = silu(g.y) * u.y;
        result.z = silu(g.z) * u.z;
        result.w = silu(g.w) * u.w;
        *reinterpret_cast<float4*>(out + idx) = result;
    } else if (idx < size) {
        for (int i = idx; i < size && i < idx + 4; i++) {
            out[i] = silu(gate[i]) * up[i];
        }
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(), "up must be contiguous");
    
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    if (size % 4 == 0 && size >= 256) {
        const int block_size = 256;
        const int num_blocks = (size / 4 + block_size - 1) / block_size;
        fused_silu_mul_kernel_vec4<<<num_blocks, block_size>>>(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            out.data_ptr<float>(),
            size
        );
    } else {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        fused_silu_mul_kernel<<<num_blocks, block_size>>>(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            out.data_ptr<float>(),
            size
        );
    }
    
    return out;
}

// Fused multiply-add for weighted accumulation
__global__ void weighted_scatter_add_kernel(
    const float* __restrict__ expert_output,  // (num_selected, hidden)
    const float* __restrict__ weights,        // (num_selected,)
    const int64_t* __restrict__ token_indices, // (num_selected,)
    float* __restrict__ output,               // (num_tokens, hidden)
    int num_selected,
    int hidden_size
) {
    int selected_idx = blockIdx.x;
    int hidden_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (selected_idx < num_selected && hidden_idx < hidden_size) {
        int64_t token_idx = token_indices[selected_idx];
        float weight = weights[selected_idx];
        float val = expert_output[selected_idx * hidden_size + hidden_idx] * weight;
        atomicAdd(&output[token_idx * hidden_size + hidden_idx], val);
    }
}

torch::Tensor weighted_scatter_add_hip(
    torch::Tensor expert_output,
    torch::Tensor weights,
    torch::Tensor token_indices,
    torch::Tensor output
) {
    int num_selected = expert_output.size(0);
    int hidden_size = expert_output.size(1);
    
    const int block_x = 256;
    dim3 block(block_x);
    dim3 grid(num_selected, (hidden_size + block_x - 1) / block_x);
    
    weighted_scatter_add_kernel<<<grid, block>>>(
        expert_output.data_ptr<float>(),
        weights.data_ptr<float>(),
        token_indices.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        num_selected,
        hidden_size
    );
    
    return output;
}
"""

fused_ops_cpp = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
torch::Tensor weighted_scatter_add_hip(torch::Tensor expert_output, torch::Tensor weights, torch::Tensor token_indices, torch::Tensor output);
"""

fused_ops = load_inline(
    name="fused_moe_ops",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_silu_mul_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE Expert with Gated GEMM using fused HIP kernels.
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

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]
        
        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)
        
        # Flatten expert_indices and expert_weights for easier processing
        expert_indices_flat = expert_indices.view(-1, top_k)  # (num_tokens, top_k)
        expert_weights_flat = expert_weights.view(-1, top_k)  # (num_tokens, top_k)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which (token, slot) pairs use this expert
            expert_mask = (expert_indices_flat == expert_idx)  # (num_tokens, top_k)
            
            if not expert_mask.any():
                continue
            
            # Get token indices and weights
            token_idx, slot_idx = torch.where(expert_mask)
            weights = expert_weights_flat[token_idx, slot_idx]
            
            # Get tokens for this expert
            expert_input = x_flat[token_idx]  # (num_selected, hidden)
            
            # Gated dual GEMM with fused SiLU*up
            gate_out = F.linear(expert_input, self.gate_proj[expert_idx])
            up_out = F.linear(expert_input, self.up_proj[expert_idx])
            
            # Fused SiLU(gate) * up
            intermediate = self.fused_ops.fused_silu_mul_hip(gate_out.contiguous(), up_out.contiguous())
            
            # Down projection
            expert_output = F.linear(intermediate, self.down_proj[expert_idx])
            
            # Weighted scatter add
            output.index_add_(0, token_idx, expert_output * weights.unsqueeze(-1))
        
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
