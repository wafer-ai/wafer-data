import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Ultra-optimized fused kernels
fused_moe_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused gate*up with silu - optimized for MI300X memory subsystem
__global__ __launch_bounds__(256)
void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    // Unroll to process 8 elements per thread when possible
    #pragma unroll 8
    for (int i = idx; i < size; i += gridDim.x * 256) {
        float g = gate[i];
        float u = up[i];
        float sig = 1.0f / (1.0f + __expf(-g));
        out[i] = g * sig * u;
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    int size = gate.numel();
    auto out = torch::empty_like(gate);
    
    int num_blocks = min((size + 255) / 256, 4096);
    
    fused_silu_mul_kernel<<<num_blocks, 256>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Weighted scatter with vectorized stores
__global__ __launch_bounds__(256)
void weighted_scatter_kernel(
    const float* __restrict__ src,
    const float* __restrict__ weights,
    const long* __restrict__ indices,
    float* __restrict__ dst,
    int N,
    int hidden
) {
    int tok = blockIdx.x;
    if (tok >= N) return;
    
    float w = weights[tok];
    long dst_idx = indices[tok];
    
    const float* src_ptr = src + tok * hidden;
    float* dst_ptr = dst + dst_idx * hidden;
    
    for (int h = threadIdx.x; h < hidden; h += 256) {
        atomicAdd(dst_ptr + h, src_ptr[h] * w);
    }
}

void weighted_scatter_hip(
    torch::Tensor src,
    torch::Tensor weights,
    torch::Tensor indices,
    torch::Tensor dst
) {
    int N = src.size(0);
    int hidden = src.size(1);
    if (N == 0) return;
    
    weighted_scatter_kernel<<<N, 256>>>(
        src.data_ptr<float>(),
        weights.data_ptr<float>(),
        indices.data_ptr<long>(),
        dst.data_ptr<float>(),
        N,
        hidden
    );
}

// Full expert forward: computes expert_output = silu(gate) * up matmul'd with down
// And weighted scatter to output
void expert_forward_fused(
    torch::Tensor input,         // (N, hidden_in)
    torch::Tensor gate_weight,   // (intermediate, hidden_in)
    torch::Tensor up_weight,     // (intermediate, hidden_in)  
    torch::Tensor down_weight,   // (hidden_out, intermediate)
    torch::Tensor weights,       // (N,)
    torch::Tensor indices,       // (N,)
    torch::Tensor output         // (total_tokens, hidden_out)
) {
    int N = input.size(0);
    if (N == 0) return;
    
    // Gate projection
    auto gate = torch::mm(input, gate_weight.t());
    
    // Up projection  
    auto up = torch::mm(input, up_weight.t());
    
    // Fused silu * up
    int size = gate.numel();
    int num_blocks = min((size + 255) / 256, 4096);
    
    auto intermediate = torch::empty_like(gate);
    fused_silu_mul_kernel<<<num_blocks, 256>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        size
    );
    
    // Down projection
    auto expert_out = torch::mm(intermediate, down_weight.t());
    
    // Weighted scatter
    int hidden = expert_out.size(1);
    weighted_scatter_kernel<<<N, 256>>>(
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        indices.data_ptr<long>(),
        output.data_ptr<float>(),
        N,
        hidden
    );
}
"""

fused_ops = load_inline(
    name="fused_moe_ops_v7",
    cpp_sources="""
    torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
    void weighted_scatter_hip(torch::Tensor src, torch::Tensor weights, torch::Tensor indices, torch::Tensor dst);
    void expert_forward_fused(torch::Tensor input, torch::Tensor gate_weight, torch::Tensor up_weight, torch::Tensor down_weight, torch::Tensor weights, torch::Tensor indices, torch::Tensor output);
    """,
    cuda_sources=fused_moe_source,
    functions=["fused_silu_mul_hip", "weighted_scatter_hip", "expert_forward_fused"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "-munsafe-fp-atomics"],
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
        
        # Process experts
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)
            
            if not expert_mask.any():
                continue
            
            token_indices, slot_indices = torch.where(expert_mask)
            weights = expert_weights_flat[token_indices, slot_indices]
            
            expert_input = x_flat[token_indices]
            
            # Use fused expert forward
            self.fused_ops.expert_forward_fused(
                expert_input.contiguous(),
                self.gate_proj[expert_idx].contiguous(),
                self.up_proj[expert_idx].contiguous(),
                self.down_proj[expert_idx].contiguous(),
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
