import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized kernels
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// Fused SiLU * mul with maximum vectorization
__global__ void fused_silu_mul_vec4_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vec4 = size / 4;
    
    if (idx < total_vec4) {
        int base = idx * 4;
        float4 g = *reinterpret_cast<const float4*>(gate + base);
        float4 u = *reinterpret_cast<const float4*>(up + base);
        float4 res;
        res.x = fast_silu(g.x) * u.x;
        res.y = fast_silu(g.y) * u.y;
        res.z = fast_silu(g.z) * u.z;
        res.w = fast_silu(g.w) * u.w;
        *reinterpret_cast<float4*>(out + base) = res;
    }
    
    // Handle remainder (only needed for threads in last wavefront)
    int remainder_start = total_vec4 * 4;
    int remainder_idx = remainder_start + idx;
    if (idx < (size - remainder_start) && remainder_idx < size) {
        out[remainder_idx] = fast_silu(gate[remainder_idx]) * up[remainder_idx];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    const int size = gate.numel();
    auto out = torch::empty_like(gate);
    
    if (size == 0) return out;
    
    const int block_size = 256;
    const int vec4_count = (size + 3) / 4;
    const int num_blocks = (vec4_count + block_size - 1) / block_size;
    
    fused_silu_mul_vec4_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Fused multiply-weighted-add: output[indices] += src * weights
// Optimized for coalesced memory access patterns
__global__ void weighted_index_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ src,
    const float* __restrict__ weights,
    const int64_t* __restrict__ indices,
    const int num_tokens,
    const int hidden_size
) {
    // Grid-stride loop for better occupancy
    for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
        const int64_t out_idx = indices[token];
        const float weight = weights[token];
        
        float* dst_row = output + out_idx * hidden_size;
        const float* src_row = src + token * hidden_size;
        
        // Process hidden dimensions with multiple threads
        for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            atomicAdd(&dst_row[h], src_row[h] * weight);
        }
    }
}

void weighted_index_add_hip(
    torch::Tensor output,
    torch::Tensor src,
    torch::Tensor weights,
    torch::Tensor indices
) {
    const int num_tokens = src.size(0);
    const int hidden_size = src.size(1);
    
    if (num_tokens == 0) return;
    
    const int block_size = 256;
    const int num_blocks = std::min(num_tokens, 4096);
    
    weighted_index_add_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        src.data_ptr<float>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        num_tokens,
        hidden_size
    );
}
"""

fused_ops_cpp = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
void weighted_index_add_hip(torch::Tensor output, torch::Tensor src, torch::Tensor weights, torch::Tensor indices);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v8",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_silu_mul_hip", "weighted_index_add_hip"],
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
        
        # Reshape for efficient indexing
        expert_indices_flat = expert_indices.reshape(-1, top_k)
        expert_weights_flat = expert_weights.reshape(-1, top_k)
        
        # Pre-compute which tokens go to each expert
        # This reduces overhead in the loop
        for expert_idx in range(self.num_experts):
            # Vectorized comparison
            mask = (expert_indices_flat == expert_idx)
            
            if not mask.any():
                continue
            
            # Get token indices for this expert
            token_idx, slot_idx = torch.where(mask)
            
            if token_idx.numel() == 0:
                continue
            
            # Gather inputs and weights
            expert_input = x_flat[token_idx]
            routing_weights = expert_weights_flat[token_idx, slot_idx]
            
            # Compute dual GEMM for gate and up projections
            # Using torch.mm which is highly optimized
            gate_result = torch.mm(expert_input, self.gate_proj[expert_idx].t())
            up_result = torch.mm(expert_input, self.up_proj[expert_idx].t())
            
            # Fused SiLU activation and element-wise multiply
            intermediate = self.fused_ops.fused_silu_mul_hip(
                gate_result.contiguous(), 
                up_result.contiguous()
            )
            
            # Down projection
            expert_output = torch.mm(intermediate, self.down_proj[expert_idx].t())
            
            # Weighted scatter-add back to output
            self.fused_ops.weighted_index_add_hip(
                output,
                expert_output.contiguous(),
                routing_weights.contiguous(),
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
