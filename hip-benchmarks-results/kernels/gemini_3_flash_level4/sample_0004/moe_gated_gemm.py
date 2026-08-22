
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

moe_gated_gemm_kernels_code = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_silu_mul_v7(
    const float* __restrict__ gate_up,
    float* __restrict__ out,
    int total_rows,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = total_rows * intermediate_size;
    if (idx < size) {
        int row = idx / intermediate_size;
        int col = idx % intermediate_size;
        float g = gate_up[(int64_t)row * 2 * intermediate_size + col];
        float u = gate_up[(int64_t)row * 2 * intermediate_size + intermediate_size + col];
        out[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

void fused_silu_mul_hip(torch::Tensor gate_up, torch::Tensor out) {
    int size = out.numel();
    fused_silu_mul_v7<<<(size + 256 - 1) / 256, 256>>>(gate_up.data_ptr<float>(), out.data_ptr<float>(), out.size(0), out.size(1));
}
"""

moe_kernels = load_inline(
    name="moe_kernels_v7",
    cpp_sources=moe_gated_gemm_kernels_code,
    functions=["fused_silu_mul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.gate_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.02)
        self.up_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * 0.02)
        
        # We'll combine them in the first forward call to avoid issues with loading weights
        self.combined_proj = None

    def forward(self, x, expert_indices, expert_weights):
        if self.combined_proj is None:
            self.combined_proj = torch.cat([self.gate_proj, self.up_proj], dim=1)
            
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        x_flat = x.view(-1, self.hidden_size)
        indices_flat = expert_indices.view(-1)
        weights_flat = expert_weights.view(-1)
        
        perm = indices_flat.argsort()
        sorted_indices = indices_flat[perm]
        
        # Gather
        gathered_x = x_flat.index_select(0, perm // top_k)
        sorted_weights = weights_flat[perm]
        
        counts = torch.bincount(sorted_indices, minlength=self.num_experts).cpu().tolist()
        
        final_output = torch.zeros_like(x_flat)
        
        start = 0
        for i in range(self.num_experts):
            count = counts[i]
            if count == 0: continue
            
            end = start + count
            # GEMM 1
            gate_up = torch.mm(gathered_x[start:end], self.combined_proj[i].t())
            
            # Fused SiLU & Mul
            inter_out = torch.empty(count, self.intermediate_size, device=x.device, dtype=x.dtype)
            moe_kernels.fused_silu_mul_hip(gate_up, inter_out)
            
            # GEMM 2
            expert_out = torch.mm(inter_out, self.down_proj[i].t())
            
            # Weighted scatter
            final_output.index_add_(0, perm[start:end] // top_k, expert_out * sorted_weights[start:end].unsqueeze(-1))
            
            start = end
            
        return final_output.view(batch, seq_len, self.hidden_size)

def get_inputs():
    batch_size, seq_len, hidden_size, num_experts, top_k = 4, 2048, 4096, 8, 2
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    expert_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k)).cuda()
    expert_weights = torch.randn(batch_size, seq_len, top_k).softmax(-1).cuda()
    return [x, expert_indices, expert_weights]

def get_init_inputs():
    return [4096, 14336, 8]
