import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Comprehensive fused kernels for MoE
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// Optimized SiLU * mul with coalesced memory access
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        float4 g = *reinterpret_cast<const float4*>(gate + idx);
        float4 u = *reinterpret_cast<const float4*>(up + idx);
        float4 res;
        res.x = fast_silu(g.x) * u.x;
        res.y = fast_silu(g.y) * u.y;
        res.z = fast_silu(g.z) * u.z;
        res.w = fast_silu(g.w) * u.w;
        *reinterpret_cast<float4*>(out + idx) = res;
    } else {
        for (int i = idx; i < size && i < idx + 4; i++) {
            out[i] = fast_silu(gate[i]) * up[i];
        }
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    int size = gate.numel();
    auto out = torch::empty_like(gate);
    
    if (size == 0) return out;
    
    int block_size = 256;
    int num_blocks = ((size + 3) / 4 + block_size - 1) / block_size;
    
    fused_silu_mul_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Process multiple experts in parallel using streams would require more complex setup
// Instead, optimize the single-expert path with better memory patterns

// Fused gather + GEMM + SiLU*mul kernel (specialized for this MoE pattern)
// This gathers tokens for an expert and computes both gate and up projections
__global__ void fused_gather_dual_gemm_silu_kernel(
    const float* __restrict__ x,           // (total_tokens, hidden)
    const int64_t* __restrict__ token_indices, // (num_selected,)
    const float* __restrict__ gate_proj,   // (intermediate, hidden)
    const float* __restrict__ up_proj,     // (intermediate, hidden)
    float* __restrict__ intermediate,      // (num_selected, intermediate)
    int num_selected,
    int hidden_size,
    int intermediate_size
) {
    // Each block handles one token
    // Within the block, threads collaborate to compute dot products
    extern __shared__ float smem[];
    float* x_shared = smem;  // hidden_size floats
    
    int token = blockIdx.x;
    if (token >= num_selected) return;
    
    int64_t src_idx = token_indices[token];
    
    // Load x into shared memory cooperatively
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        x_shared[i] = x[src_idx * hidden_size + i];
    }
    __syncthreads();
    
    // Each thread computes one intermediate dimension
    for (int out_dim = threadIdx.x; out_dim < intermediate_size; out_dim += blockDim.x) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        
        // Compute dot product with both gate and up projections
        for (int k = 0; k < hidden_size; k++) {
            float x_val = x_shared[k];
            gate_sum += x_val * gate_proj[out_dim * hidden_size + k];
            up_sum += x_val * up_proj[out_dim * hidden_size + k];
        }
        
        // Fused SiLU * up
        intermediate[token * intermediate_size + out_dim] = fast_silu(gate_sum) * up_sum;
    }
}

torch::Tensor fused_gather_dual_gemm_silu_hip(
    torch::Tensor x,
    torch::Tensor token_indices,
    torch::Tensor gate_proj,
    torch::Tensor up_proj
) {
    int num_selected = token_indices.size(0);
    int hidden_size = x.size(1);
    int intermediate_size = gate_proj.size(0);
    
    auto intermediate = torch::empty({num_selected, intermediate_size}, x.options());
    
    if (num_selected == 0) return intermediate;
    
    int block_size = 256;
    int shared_mem = hidden_size * sizeof(float);
    
    fused_gather_dual_gemm_silu_kernel<<<num_selected, block_size, shared_mem>>>(
        x.data_ptr<float>(),
        token_indices.data_ptr<int64_t>(),
        gate_proj.data_ptr<float>(),
        up_proj.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        num_selected,
        hidden_size,
        intermediate_size
    );
    
    return intermediate;
}

// Weighted scatter-add with vectorized access
__global__ void weighted_scatter_add_vec_kernel(
    float* __restrict__ output,
    const float* __restrict__ expert_out,
    const float* __restrict__ weights,
    const int64_t* __restrict__ token_indices,
    int num_selected,
    int hidden_size
) {
    int token = blockIdx.x;
    if (token >= num_selected) return;
    
    int64_t out_idx = token_indices[token];
    float weight = weights[token];
    
    const float* src = expert_out + token * hidden_size;
    float* dst = output + out_idx * hidden_size;
    
    // Vectorized accumulation
    int idx = threadIdx.x * 4;
    if (idx + 3 < hidden_size) {
        float4 v = *reinterpret_cast<const float4*>(src + idx);
        v.x *= weight;
        v.y *= weight;
        v.z *= weight;
        v.w *= weight;
        atomicAdd(&dst[idx], v.x);
        atomicAdd(&dst[idx + 1], v.y);
        atomicAdd(&dst[idx + 2], v.z);
        atomicAdd(&dst[idx + 3], v.w);
    } else {
        for (int i = idx; i < hidden_size && i < idx + 4; i++) {
            atomicAdd(&dst[i], src[i] * weight);
        }
    }
    
    // Handle remaining elements
    int remaining_start = (hidden_size / (blockDim.x * 4)) * (blockDim.x * 4);
    for (int i = remaining_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        atomicAdd(&dst[i], src[i] * weight);
    }
}

void weighted_scatter_add_hip(
    torch::Tensor output,
    torch::Tensor expert_out,
    torch::Tensor weights,
    torch::Tensor token_indices
) {
    int num_selected = expert_out.size(0);
    int hidden_size = expert_out.size(1);
    
    if (num_selected == 0) return;
    
    int block_size = std::min((hidden_size + 3) / 4, 256);
    
    weighted_scatter_add_vec_kernel<<<num_selected, block_size>>>(
        output.data_ptr<float>(),
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        token_indices.data_ptr<int64_t>(),
        num_selected,
        hidden_size
    );
}
"""

fused_ops_cpp = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
torch::Tensor fused_gather_dual_gemm_silu_hip(torch::Tensor x, torch::Tensor token_indices, torch::Tensor gate_proj, torch::Tensor up_proj);
void weighted_scatter_add_hip(torch::Tensor output, torch::Tensor expert_out, torch::Tensor weights, torch::Tensor token_indices);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v6",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_silu_mul_hip", "fused_gather_dual_gemm_silu_hip", "weighted_scatter_add_hip"],
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
            
            # Use torch.mm for GEMM (well optimized on AMD)
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
