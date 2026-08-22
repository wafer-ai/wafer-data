import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

moe_gated_gemm_source = """
#include <torch/extension.h>

#define BLOCK_SIZE 256

__device__ __forceinline__ float silu(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__global__ void moe_gated_gemm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gate_weights,
    const float* __restrict__ up_weights,
    const float* __restrict__ down_weights,
    float* __restrict__ output,
    int hidden_size,
    int intermediate_size,
    int num_tokens
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int out_idx = tid; out_idx < hidden_size; out_idx += stride) {
        float down_sum = 0.0f;
        
        for (int k = 0; k < intermediate_size; k++) {
            float gate_val = 0.0f;
            float up_val = 0.0f;
            
            // Compute dot product
            for (int j = 0; j < hidden_size; j++) {
                float x_val = x[token_idx * hidden_size + j];
                gate_val += x_val * gate_weights[k * hidden_size + j];
                up_val += x_val * up_weights[k * hidden_size + j];
            }
            
            // SiLU activation and element-wise multiply
            gate_val = silu(gate_val);
            down_sum += gate_val * up_val * down_weights[out_idx * intermediate_size + k];
        }
        
        output[token_idx * hidden_size + out_idx] = down_sum;
    }
}

torch::Tensor moe_gated_gemm_hip(
    torch::Tensor x,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights
) {
    int num_tokens = x.size(0);
    int hidden_size = x.size(1);
    int intermediate_size = gate_weights.size(0);
    
    auto output = torch::zeros_like(x);
    
    moe_gated_gemm_kernel<<<num_tokens, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        gate_weights.data_ptr<float>(),
        up_weights.data_ptr<float>(),
        down_weights.data_ptr<float>(),
        output.data_ptr<float>(),
        hidden_size,
        intermediate_size,
        num_tokens
    );
    
    return output;
}
"""

moe_gated_gemm = load_inline(
    name="moe_gated_gemm",
    cuda_sources=moe_gated_gemm_source,
    functions=["moe_gated_gemm_hip"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
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
        
        self.moe_gated_gemm = moe_gated_gemm

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

        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx)

            if not expert_mask.any():
                continue

            batch_idx, seq_idx, slot_idx = torch.where(expert_mask)
            token_indices = batch_idx * seq_len + seq_idx
            weights = expert_weights[batch_idx, seq_idx, slot_idx]

            expert_input = x_flat[token_indices]

            expert_output = self.moe_gated_gemm.moe_gated_gemm_hip(
                expert_input,
                self.gate_proj[expert_idx],
                self.up_proj[expert_idx],
                self.down_proj[expert_idx]
            )

            output.index_add_(0, token_indices, expert_output * weights.unsqueeze(-1))

        return output.view(batch, seq_len, self.hidden_size)