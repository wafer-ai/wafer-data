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

// Fused SiLU * up kernel with vectorized loads
__global__ void fused_silu_mul_kernel_vec4(
    const float4* __restrict__ gate,
    const float4* __restrict__ up,
    float4* __restrict__ out,
    int size4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        float4 g = gate[idx];
        float4 u = up[idx];
        
        float4 result;
        result.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        result.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        result.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        result.w = (g.w / (1.0f + expf(-g.w))) * u.w;
        
        out[idx] = result;
    }
}

__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = silu_g * up[idx];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.numel();
    auto out = torch::empty_like(gate);
    
    // Use vectorized version if aligned
    if (size % 4 == 0 && ((uintptr_t)gate.data_ptr<float>() % 16) == 0 &&
        ((uintptr_t)up.data_ptr<float>() % 16) == 0) {
        int size4 = size / 4;
        const int block_size = 256;
        const int num_blocks = (size4 + block_size - 1) / block_size;
        
        fused_silu_mul_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(gate.data_ptr<float>()),
            reinterpret_cast<const float4*>(up.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            size4
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

// Optimized weighted scatter-add kernel using more parallelism
__global__ void weighted_scatter_add_kernel(
    const float* __restrict__ src,
    const float* __restrict__ weights,
    const int64_t* __restrict__ indices,
    float* __restrict__ dst,
    int num_tokens,
    int hidden_size
) {
    int token_idx = blockIdx.x;
    
    if (token_idx < num_tokens) {
        int64_t dst_idx = indices[token_idx];
        float w = weights[token_idx];
        
        // Process multiple elements per thread using vectorized access
        for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            float val = src[token_idx * hidden_size + h] * w;
            atomicAdd(&dst[dst_idx * hidden_size + h], val);
        }
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
    
    dim3 grid(num_tokens);
    dim3 block(min(1024, hidden_size));
    
    weighted_scatter_add_kernel<<<grid, block>>>(
        src.data_ptr<float>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dst.data_ptr<float>(),
        num_tokens,
        hidden_size
    );
}

// Fused dual linear + silu + mul kernel
// Computes: SiLU(x @ gate_proj.T) * (x @ up_proj.T)
// This reads x once and writes intermediate once
__global__ void fused_gate_up_silu_mul_kernel(
    const float* __restrict__ x,          // (num_tokens, hidden_size)
    const float* __restrict__ gate_proj,  // (intermediate_size, hidden_size)
    const float* __restrict__ up_proj,    // (intermediate_size, hidden_size)
    float* __restrict__ out,              // (num_tokens, intermediate_size)
    int num_tokens,
    int hidden_size,
    int intermediate_size
) {
    // Each block computes one output element
    int token_idx = blockIdx.y;
    int out_idx = blockIdx.x;
    
    if (token_idx >= num_tokens || out_idx >= intermediate_size) return;
    
    // Shared memory for partial sums
    __shared__ float gate_partial[256];
    __shared__ float up_partial[256];
    
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    
    // Each thread accumulates partial dot products
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float x_val = x[token_idx * hidden_size + h];
        gate_sum += x_val * gate_proj[out_idx * hidden_size + h];
        up_sum += x_val * up_proj[out_idx * hidden_size + h];
    }
    
    gate_partial[threadIdx.x] = gate_sum;
    up_partial[threadIdx.x] = up_sum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gate_partial[threadIdx.x] += gate_partial[threadIdx.x + stride];
            up_partial[threadIdx.x] += up_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float g = gate_partial[0];
        float silu_g = g / (1.0f + expf(-g));
        out[token_idx * intermediate_size + out_idx] = silu_g * up_partial[0];
    }
}

torch::Tensor fused_gate_up_silu_mul_hip(
    torch::Tensor x,
    torch::Tensor gate_proj,
    torch::Tensor up_proj
) {
    int num_tokens = x.size(0);
    int hidden_size = x.size(1);
    int intermediate_size = gate_proj.size(0);
    
    auto out = torch::empty({num_tokens, intermediate_size}, x.options());
    
    // Only use fused kernel for small token counts (otherwise cublas is faster)
    if (num_tokens <= 32) {
        dim3 grid(intermediate_size, num_tokens);
        dim3 block(256);
        
        fused_gate_up_silu_mul_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            gate_proj.data_ptr<float>(),
            up_proj.data_ptr<float>(),
            out.data_ptr<float>(),
            num_tokens,
            hidden_size,
            intermediate_size
        );
        return out;
    }
    
    // For larger token counts, use separate matmuls (rocBLAS is optimized)
    // But still use fused silu_mul
    auto gate = torch::mm(x, gate_proj.t());
    auto up = torch::mm(x, up_proj.t());
    
    int size = gate.numel();
    if (size % 4 == 0) {
        int size4 = size / 4;
        const int block_size = 256;
        const int num_blocks = (size4 + block_size - 1) / block_size;
        
        fused_silu_mul_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(gate.data_ptr<float>()),
            reinterpret_cast<const float4*>(up.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            size4
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
"""

fused_ops = load_inline(
    name="fused_moe_ops_v2",
    cpp_sources="""
    torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
    void weighted_scatter_add_hip(torch::Tensor src, torch::Tensor weights, torch::Tensor indices, torch::Tensor dst);
    torch::Tensor fused_gate_up_silu_mul_hip(torch::Tensor x, torch::Tensor gate_proj, torch::Tensor up_proj);
    """,
    cuda_sources=fused_moe_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_add_hip", "fused_gate_up_silu_mul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE Expert with Gated GEMM.
    
    Optimizations:
    1. Fused SiLU + multiply with vectorized loads
    2. Fused gate + up projection for small token counts
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
        
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)
            
            if not expert_mask.any():
                continue
            
            token_indices, slot_indices = torch.where(expert_mask)
            weights = expert_weights_flat[token_indices, slot_indices]
            
            expert_input = x_flat[token_indices]
            
            # Use fused gate+up+silu+mul
            intermediate = self.fused_ops.fused_gate_up_silu_mul_hip(
                expert_input.contiguous(),
                self.gate_proj[expert_idx].contiguous(),
                self.up_proj[expert_idx].contiguous()
            )
            
            # Down projection
            expert_output = F.linear(intermediate, self.down_proj[expert_idx])
            
            # Weighted scatter add
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
