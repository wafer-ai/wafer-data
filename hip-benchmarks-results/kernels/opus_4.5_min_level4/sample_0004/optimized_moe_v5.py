import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# More aggressive kernel fusion - combine gate+up projections
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <vector>

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// Optimized vectorized SiLU * mul
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int total_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per iteration for better memory bandwidth
    for (int i = idx * 4; i + 3 < total_size; i += stride * 4) {
        float4 g = *reinterpret_cast<const float4*>(gate + i);
        float4 u = *reinterpret_cast<const float4*>(up + i);
        float4 res;
        res.x = fast_silu(g.x) * u.x;
        res.y = fast_silu(g.y) * u.y;
        res.z = fast_silu(g.z) * u.z;
        res.w = fast_silu(g.w) * u.w;
        *reinterpret_cast<float4*>(out + i) = res;
    }
    
    // Handle remainder
    const int remainder_start = (total_size / 4) * 4;
    for (int i = remainder_start + idx; i < total_size; i += stride) {
        out[i] = fast_silu(gate[i]) * up[i];
    }
}

torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up) {
    const int size = gate.numel();
    auto out = torch::empty_like(gate);
    
    if (size == 0) return out;
    
    const int block_size = 256;
    const int num_vec4 = (size + 3) / 4;
    const int num_blocks = std::min((num_vec4 + block_size - 1) / block_size, 1024);
    
    fused_silu_mul_kernel<<<num_blocks, block_size>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Fused dual GEMM: computes both gate and up projections simultaneously
// This reads x once and writes both outputs
// gate_out = x @ gate_proj.T
// up_out = x @ up_proj.T
// Then computes: intermediate = SiLU(gate_out) * up_out
__global__ void fused_dual_gemm_silu_kernel(
    const float* __restrict__ x,           // (N, H)
    const float* __restrict__ gate_proj,   // (I, H) - stored transposed
    const float* __restrict__ up_proj,     // (I, H) - stored transposed
    float* __restrict__ intermediate,      // (N, I)
    int N,  // num tokens
    int H,  // hidden size
    int I   // intermediate size
) {
    // This is a simplified version - for large matrices, we should use tiled GEMM
    // But for demonstration, we compute one output element per thread
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (row < N && col < I) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        
        const float* x_row = x + row * H;
        const float* gate_col = gate_proj + col * H;
        const float* up_col = up_proj + col * H;
        
        // Compute both dot products simultaneously
        for (int k = 0; k < H; k += 4) {
            if (k + 3 < H) {
                float4 xv = *reinterpret_cast<const float4*>(x_row + k);
                float4 gv = *reinterpret_cast<const float4*>(gate_col + k);
                float4 uv = *reinterpret_cast<const float4*>(up_col + k);
                
                gate_sum += xv.x * gv.x + xv.y * gv.y + xv.z * gv.z + xv.w * gv.w;
                up_sum += xv.x * uv.x + xv.y * uv.y + xv.z * uv.z + xv.w * uv.w;
            } else {
                for (int kk = k; kk < H; kk++) {
                    float xval = x_row[kk];
                    gate_sum += xval * gate_col[kk];
                    up_sum += xval * up_col[kk];
                }
            }
        }
        
        // Apply SiLU and multiply
        intermediate[row * I + col] = fast_silu(gate_sum) * up_sum;
    }
}
"""

fused_ops_cpp = """
torch::Tensor fused_silu_mul_hip(torch::Tensor gate, torch::Tensor up);
"""

fused_ops = load_inline(
    name="fused_moe_ops_v5",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_silu_mul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized MoE using fused SiLU+multiply and efficient PyTorch operations.
    Key insight: The GEMM operations dominate - focus on reducing overhead elsewhere.
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
        
        # Flatten indices and weights
        expert_indices_flat = expert_indices.view(-1, top_k)
        expert_weights_flat = expert_weights.view(-1, top_k)
        
        # Pre-compute all expert masks to reduce per-expert torch.where overhead
        expert_masks = []
        expert_data = []
        
        for expert_idx in range(self.num_experts):
            mask = (expert_indices_flat == expert_idx)
            if mask.any():
                token_idx, slot_idx = torch.where(mask)
                weights = expert_weights_flat[token_idx, slot_idx]
                expert_masks.append((expert_idx, token_idx, slot_idx, weights))
        
        # Process each active expert
        for expert_idx, token_idx, slot_idx, weights in expert_masks:
            expert_input = x_flat[token_idx]
            
            # Combined GEMM + SiLU + multiply using torch operations
            # The fused kernel helps with the SiLU*up part
            gate_out = torch.mm(expert_input, self.gate_proj[expert_idx].t())
            up_out = torch.mm(expert_input, self.up_proj[expert_idx].t())
            
            # Fused SiLU(gate) * up
            intermediate = self.fused_ops.fused_silu_mul_hip(
                gate_out.contiguous(), 
                up_out.contiguous()
            )
            
            # Down projection
            expert_output = torch.mm(intermediate, self.down_proj[expert_idx].t())
            
            # Apply weights and accumulate
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
