import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernels for MoE - focus on batching experts
fused_moe_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fast vectorized SiLU + multiply
__global__ void fused_silu_mul_vec4_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        float4 g = *reinterpret_cast<const float4*>(&gate[idx]);
        float4 u = *reinterpret_cast<const float4*>(&up[idx]);
        
        float4 result;
        result.x = (g.x / (1.0f + __expf(-g.x))) * u.x;
        result.y = (g.y / (1.0f + __expf(-g.y))) * u.y;
        result.z = (g.z / (1.0f + __expf(-g.z))) * u.z;
        result.w = (g.w / (1.0f + __expf(-g.w))) * u.w;
        
        *reinterpret_cast<float4*>(&out[idx]) = result;
    } else if (idx < size) {
        // Handle remainder
        for (int i = idx; i < size && i < idx + 4; i++) {
            float g = gate[i];
            out[i] = (g / (1.0f + __expf(-g))) * up[i];
        }
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int block_size = 256;
    const int num_blocks = ((size + 3) / 4 + block_size - 1) / block_size;
    
    fused_silu_mul_vec4_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Weighted accumulation into output buffer
__global__ void weighted_index_add_kernel(
    const float* __restrict__ src,
    const float* __restrict__ weights,
    const int64_t* __restrict__ indices,
    float* __restrict__ dst,
    int num_tokens,
    int hidden_size
) {
    extern __shared__ float s_weight[];
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    // Load weight to shared memory (one per block)
    if (threadIdx.x == 0) {
        s_weight[0] = weights[token_idx];
    }
    __syncthreads();
    
    float w = s_weight[0];
    int64_t dst_idx = indices[token_idx];
    
    // Vectorized access for better memory bandwidth
    int h4 = threadIdx.x;
    int stride = blockDim.x;
    int hidden4 = hidden_size / 4;
    
    const float4* src4 = reinterpret_cast<const float4*>(&src[token_idx * hidden_size]);
    float4* dst4 = reinterpret_cast<float4*>(&dst[dst_idx * hidden_size]);
    
    for (int i = h4; i < hidden4; i += stride) {
        float4 val = src4[i];
        val.x *= w; val.y *= w; val.z *= w; val.w *= w;
        atomicAdd(&dst4[i].x, val.x);
        atomicAdd(&dst4[i].y, val.y);
        atomicAdd(&dst4[i].z, val.z);
        atomicAdd(&dst4[i].w, val.w);
    }
    
    // Handle remainder
    int start = hidden4 * 4;
    for (int h = start + threadIdx.x; h < hidden_size; h += blockDim.x) {
        float val = src[token_idx * hidden_size + h] * w;
        atomicAdd(&dst[dst_idx * hidden_size + h], val);
    }
}

void weighted_index_add_hip(
    torch::Tensor src,
    torch::Tensor weights,
    torch::Tensor indices,
    torch::Tensor dst
) {
    int num_tokens = src.size(0);
    int hidden_size = src.size(1);
    
    const int block_size = 256;
    
    weighted_index_add_kernel<<<num_tokens, block_size, sizeof(float)>>>(
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
    name="fused_moe_ops_v3",
    cpp_sources="""
    torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
    void weighted_index_add_hip(torch::Tensor src, torch::Tensor weights, torch::Tensor indices, torch::Tensor dst);
    """,
    cuda_sources=fused_moe_source,
    functions=["fused_silu_mul_hip", "weighted_index_add_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE Expert with Gated GEMM.
    
    Optimizations:
    1. Vectorized SiLU + multiply kernel
    2. Optimized weighted scatter-add with shared memory
    3. Pre-sorted expert processing for better cache utilization
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

        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]

        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)

        expert_indices_flat = expert_indices.view(-1, top_k)
        expert_weights_flat = expert_weights.view(-1, top_k)
        
        # Pre-compute sorted token assignments to improve cache efficiency
        # Group all tokens by their expert assignment
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)
            
            if not expert_mask.any():
                continue
            
            token_indices, slot_indices = torch.where(expert_mask)
            weights = expert_weights_flat[token_indices, slot_indices]
            
            expert_input = x_flat[token_indices]
            
            # Compute gate and up projections using optimized BLAS
            gate = torch.mm(expert_input, self.gate_proj[expert_idx].t())
            up = torch.mm(expert_input, self.up_proj[expert_idx].t())
            
            # Fused SiLU + multiply
            intermediate = self.fused_ops.fused_silu_mul_hip(gate, up)
            
            # Down projection
            expert_output = torch.mm(intermediate, self.down_proj[expert_idx].t())
            
            # Weighted scatter add
            self.fused_ops.weighted_index_add_hip(
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
