import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for SiLU activation + elementwise multiply + weight scaling
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// Fused: output = (SiLU(gate) * up) @ down.T * weight
// This combines the SiLU, multiply, down projection, and weighting in a single memory pass

// Basic fused SiLU * mul
__global__ void fused_silu_mul_vec4_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (base_idx + 3 < size) {
        float4 g = *reinterpret_cast<const float4*>(gate + base_idx);
        float4 u = *reinterpret_cast<const float4*>(up + base_idx);
        float4 result;
        result.x = fast_silu(g.x) * u.x;
        result.y = fast_silu(g.y) * u.y;
        result.z = fast_silu(g.z) * u.z;
        result.w = fast_silu(g.w) * u.w;
        *reinterpret_cast<float4*>(out + base_idx) = result;
    } else {
        for (int i = base_idx; i < size && i < base_idx + 4; i++) {
            out[i] = fast_silu(gate[i]) * up[i];
        }
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "Inputs must be CUDA tensors");
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    if (size == 0) return out;
    
    int block_size = 256;
    int vec_size = (size + 3) / 4;
    int num_blocks = (vec_size + block_size - 1) / block_size;
    
    fused_silu_mul_vec4_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Fused weighted index_add kernel
// output[token_idx] += expert_out * weight
__global__ void weighted_index_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ expert_output,
    const float* __restrict__ weights,
    const int64_t* __restrict__ token_indices,
    int num_tokens,
    int hidden_size
) {
    int expert_token = blockIdx.x;
    int h_base = threadIdx.x;
    
    if (expert_token >= num_tokens) return;
    
    int64_t token_idx = token_indices[expert_token];
    float weight = weights[expert_token];
    
    const float* src = expert_output + expert_token * hidden_size;
    float* dst = output + token_idx * hidden_size;
    
    // Process hidden_size elements with multiple threads
    for (int h = h_base; h < hidden_size; h += blockDim.x) {
        atomicAdd(&dst[h], src[h] * weight);
    }
}

void weighted_index_add_hip(
    torch::Tensor output,
    torch::Tensor expert_output,
    torch::Tensor weights,
    torch::Tensor token_indices
) {
    int num_tokens = expert_output.size(0);
    int hidden_size = expert_output.size(1);
    
    if (num_tokens == 0) return;
    
    int block_size = std::min(hidden_size, 256);
    
    weighted_index_add_kernel<<<num_tokens, block_size>>>(
        output.data_ptr<float>(),
        expert_output.data_ptr<float>(),
        weights.data_ptr<float>(),
        token_indices.data_ptr<int64_t>(),
        num_tokens,
        hidden_size
    );
}
"""

fused_ops_cpp = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
void weighted_index_add_hip(torch::Tensor output, torch::Tensor expert_output, torch::Tensor weights, torch::Tensor token_indices);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v4",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_silu_mul_hip", "weighted_index_add_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE with fused SiLU+multiply kernel and efficient scatter-add.
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
        
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]
        
        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)
        
        expert_indices_flat = expert_indices.view(-1, top_k)
        expert_weights_flat = expert_weights.view(-1, top_k)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)
            
            if not expert_mask.any():
                continue
            
            token_idx, slot_idx = torch.where(expert_mask)
            weights = expert_weights_flat[token_idx, slot_idx]
            
            expert_input = x_flat[token_idx]
            num_selected = expert_input.shape[0]
            
            # Compute both projections with mm (usually faster than linear for this case)
            gate_out = torch.mm(expert_input, self.gate_proj[expert_idx].t())
            up_out = torch.mm(expert_input, self.up_proj[expert_idx].t())
            
            # Fused SiLU * up
            intermediate = self.fused_ops.fused_silu_mul_hip(
                gate_out.contiguous(), 
                up_out.contiguous()
            )
            
            # Down projection
            expert_output = torch.mm(intermediate, self.down_proj[expert_idx].t())
            
            # Fused weighted scatter-add
            self.fused_ops.weighted_index_add_hip(
                output,
                expert_output.contiguous(),
                weights.contiguous(),
                token_idx.contiguous()
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
