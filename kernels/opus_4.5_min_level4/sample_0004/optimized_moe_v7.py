import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fixed and optimized fused kernels
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// Optimized SiLU * mul with vectorized access
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int size
) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (base + 3 < size) {
        float4 g = *reinterpret_cast<const float4*>(gate + base);
        float4 u = *reinterpret_cast<const float4*>(up + base);
        float4 res;
        res.x = fast_silu(g.x) * u.x;
        res.y = fast_silu(g.y) * u.y;
        res.z = fast_silu(g.z) * u.z;
        res.w = fast_silu(g.w) * u.w;
        *reinterpret_cast<float4*>(out + base) = res;
    } else {
        // Handle tail
        for (int i = base; i < size && i < base + 4; i++) {
            out[i] = fast_silu(gate[i]) * up[i];
        }
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    const int size = gate.numel();
    auto out = torch::empty_like(gate);
    
    if (size == 0) return out;
    
    const int block_size = 256;
    const int num_vec4 = (size + 3) / 4;
    const int num_blocks = (num_vec4 + block_size - 1) / block_size;
    
    fused_silu_mul_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Fixed weighted scatter-add kernel
// Each thread block handles one token's output
__global__ void weighted_scatter_add_kernel(
    float* __restrict__ output,             // (num_total_tokens, hidden)
    const float* __restrict__ expert_out,   // (num_selected, hidden)
    const float* __restrict__ weights,      // (num_selected,)
    const int64_t* __restrict__ token_indices, // (num_selected,)
    const int num_selected,
    const int hidden_size
) {
    const int token_local = blockIdx.x;  // Which token in this expert batch
    if (token_local >= num_selected) return;
    
    const int64_t token_global = token_indices[token_local];
    const float weight = weights[token_local];
    
    // Each thread processes multiple hidden elements
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float val = expert_out[token_local * hidden_size + h] * weight;
        atomicAdd(&output[token_global * hidden_size + h], val);
    }
}

void weighted_scatter_add_hip(
    torch::Tensor output,
    torch::Tensor expert_out,
    torch::Tensor weights,
    torch::Tensor token_indices
) {
    const int num_selected = expert_out.size(0);
    const int hidden_size = expert_out.size(1);
    
    if (num_selected == 0) return;
    
    const int block_size = 256;
    
    weighted_scatter_add_kernel<<<num_selected, block_size>>>(
        output.data_ptr<float>(),
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        token_indices.data_ptr<int64_t>(),
        num_selected,
        hidden_size
    );
}

// Alternative: Process weighted output inline to reduce kernel calls
// Combines: expert_output = mm(intermediate, down_proj) then weighted scatter
__global__ void fused_down_proj_scatter_kernel(
    float* __restrict__ output,
    const float* __restrict__ intermediate,
    const float* __restrict__ down_proj,
    const float* __restrict__ weights,
    const int64_t* __restrict__ token_indices,
    const int num_selected,
    const int hidden_size,
    const int intermediate_size
) {
    const int token_local = blockIdx.x;
    if (token_local >= num_selected) return;
    
    const int64_t token_global = token_indices[token_local];
    const float weight = weights[token_local];
    
    // Each thread computes one output hidden dimension
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float sum = 0.0f;
        const float* inter_row = intermediate + token_local * intermediate_size;
        const float* down_col = down_proj + h * intermediate_size;
        
        // Dot product
        for (int i = 0; i < intermediate_size; i++) {
            sum += inter_row[i] * down_col[i];
        }
        
        atomicAdd(&output[token_global * hidden_size + h], sum * weight);
    }
}
"""

fused_ops_cpp = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
void weighted_scatter_add_hip(torch::Tensor output, torch::Tensor expert_out, torch::Tensor weights, torch::Tensor token_indices);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v7",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
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
        
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)
            
            if not expert_mask.any():
                continue
            
            token_idx, slot_idx = torch.where(expert_mask)
            weights = expert_weights_flat[token_idx, slot_idx]
            
            expert_input = x_flat[token_idx]
            
            # Parallel GEMMs
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
            self.fused_ops.weighted_scatter_add_hip(
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
