import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os
import math

# Set hipcc as the compiler for ROCm/HIP
os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Vectorized kernel using float4
__global__ void silu_mul_kernel_vec(const float4* __restrict__ input, float4* __restrict__ output, int total_vecs, int vec_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vecs) {
        int row = idx / vec_cols;
        int col = idx % vec_cols;
        
        // Input stride is 2 * vec_cols
        int gate_idx = row * (2 * vec_cols) + col;
        int up_idx = gate_idx + vec_cols;
        
        float4 g = input[gate_idx];
        float4 u = input[up_idx];
        float4 out;
        
        out.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        out.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        out.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        out.w = (g.w / (1.0f + expf(-g.w))) * u.w;
        
        output[idx] = out;
    }
}

// Fallback scalar kernel
__global__ void silu_mul_kernel_scalar(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        int g_idx = row * (2 * cols) + col;
        int u_idx = g_idx + cols;
        float g = input[g_idx];
        output[idx] = silu(g) * input[u_idx];
    }
}

torch::Tensor silu_mul_hip(torch::Tensor input) {
    auto rows = input.size(0);
    auto double_cols = input.size(1);
    
    if (double_cols % 2 != 0) {
        return torch::zeros({rows, double_cols/2}, input.options());
    }
    
    auto cols = double_cols / 2;
    auto output = torch::empty({rows, cols}, input.options());

    bool can_vectorize = (cols % 4 == 0) && 
                         (reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0);

    if (can_vectorize) {
        int vec_cols = cols / 4;
        int total_vecs = rows * vec_cols;
        const int block_size = 256;
        const int num_blocks = (total_vecs + block_size - 1) / block_size;
        
        silu_mul_kernel_vec<<<num_blocks, block_size>>>(
            reinterpret_cast<float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            total_vecs,
            vec_cols
        );
    } else {
        int total = rows * cols;
        const int block_size = 256;
        const int num_blocks = (total + block_size - 1) / block_size;
        silu_mul_kernel_scalar<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            rows,
            cols
        );
    }

    return output;
}
"""

module = load_inline(
    name="moe_kernels_v4",
    cpp_sources=cpp_source,
    functions=["silu_mul_hip"],
    verbose=False,
    extra_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        # Fuse gate_proj and up_proj
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        
        with torch.no_grad():
             gate = torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
             up = torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
             self.w13.data.copy_(torch.cat([gate, up], dim=1))
             self.down_proj.data.copy_(torch.randn(num_experts, hidden_size, intermediate_size) * 0.02)

        self.silu_mul = module.silu_mul_hip

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape
        top_k = expert_indices.shape[-1]
        
        x_flat = x.view(-1, hidden_dim)
        expert_indices_flat = expert_indices.view(-1)
        expert_weights_flat = expert_weights.view(-1)
        
        num_tokens = batch * seq_len
        token_indices_flat = torch.arange(num_tokens, device=x.device).unsqueeze(-1).expand(-1, top_k).reshape(-1)
        
        # Sort tokens by expert assignment to enable batched processing per expert
        _, sort_idx = expert_indices_flat.sort()
        sorted_token_indices = token_indices_flat[sort_idx]
        sorted_weights = expert_weights_flat[sort_idx]
        
        # Gather inputs
        x_gathered = x_flat[sorted_token_indices]
        results_flat = torch.empty_like(x_gathered)
        
        # Count tokens per expert
        counts = torch.bincount(expert_indices_flat, minlength=self.num_experts)
        counts_cpu = counts.tolist()
        
        start_idx = 0
        
        # Process each expert sequentially
        # Sequential processing is chosen over streams because:
        # 1. The GEMM sizes are large enough to saturate the GPU
        # 2. Stream overhead (creation, events, syncing) outweighs the benefits of overlapping tails
        for expert_id in range(self.num_experts):
            count = counts_cpu[expert_id]
            if count == 0:
                continue
            
            end_idx = start_idx + count
            
            # Slice for current expert
            inp_slice = x_gathered[start_idx:end_idx]
            
            # 1. Fused Gate+Up GEMM
            # (count, H) @ (2I, H).T -> (count, 2I)
            gemm1 = F.linear(inp_slice, self.w13[expert_id])
            
            # 2. Fused Activation (HIP Kernel)
            # (count, 2I) -> (count, I)
            act = self.silu_mul(gemm1)
            
            # 3. Down GEMM
            # (count, I) @ (H, I).T -> (count, H)
            gemm2 = F.linear(act, self.down_proj[expert_id])
            
            # 4. Scale
            w_slice = sorted_weights[start_idx:end_idx].unsqueeze(-1)
            gemm2.mul_(w_slice)
            
            # Store result
            results_flat[start_idx:end_idx] = gemm2
            
            start_idx = end_idx
            
        # Scatter results back
        output = torch.zeros(num_tokens, hidden_dim, device=x.device, dtype=x.dtype)
        output.index_add_(0, sorted_token_indices, results_flat)
        
        return output.view(batch, seq_len, hidden_dim)
