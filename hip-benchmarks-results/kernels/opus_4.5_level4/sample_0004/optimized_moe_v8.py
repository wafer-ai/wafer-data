import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized MoE kernels
fused_moe_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Ultra-fast fused silu*mul with coalesced accesses
__global__ __launch_bounds__(256, 8)
void fast_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < size; i += stride) {
        const float g = gate[i];
        const float u = up[i];
        // SiLU(g) * u = g * sigmoid(g) * u
        out[i] = g * __frcp_rn(1.0f + __expf(-g)) * u;
    }
}

torch::Tensor fast_silu_mul(torch::Tensor gate, torch::Tensor up) {
    const int size = gate.numel();
    auto out = torch::empty_like(gate);
    
    const int threads = 256;
    const int blocks = min((size + threads - 1) / threads, 4096);
    
    fast_silu_mul_kernel<<<blocks, threads>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

// Efficient weighted accumulation with reduced atomic contention
__global__ __launch_bounds__(256, 4)
void fast_weighted_accum_kernel(
    const float* __restrict__ src,
    const float* __restrict__ weights,
    const long* __restrict__ indices,
    float* __restrict__ dst,
    const int N,
    const int hidden
) {
    const int tok = blockIdx.x;
    if (tok >= N) return;
    
    const float w = weights[tok];
    const long dst_idx = indices[tok];
    const float* src_row = src + tok * hidden;
    float* dst_row = dst + dst_idx * hidden;
    
    // Process 4 elements at a time
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        atomicAdd(dst_row + h, src_row[h] * w);
    }
}

void fast_weighted_accum(
    torch::Tensor src,
    torch::Tensor weights,
    torch::Tensor indices,
    torch::Tensor dst
) {
    const int N = src.size(0);
    const int hidden = src.size(1);
    if (N == 0) return;
    
    fast_weighted_accum_kernel<<<N, 256>>>(
        src.data_ptr<float>(),
        weights.data_ptr<float>(),
        indices.data_ptr<long>(),
        dst.data_ptr<float>(),
        N,
        hidden
    );
}
"""

fused_ops = load_inline(
    name="fused_moe_ops_v8",
    cpp_sources="""
    torch::Tensor fast_silu_mul(torch::Tensor gate, torch::Tensor up);
    void fast_weighted_accum(torch::Tensor src, torch::Tensor weights, torch::Tensor indices, torch::Tensor dst);
    """,
    cuda_sources=fused_moe_source,
    functions=["fast_silu_mul", "fast_weighted_accum"],
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
        
        # Pre-allocate expert weight views
        self._gate_views = None
        self._up_views = None
        self._down_views = None

    def _get_weight_views(self):
        if self._gate_views is None:
            self._gate_views = [self.gate_proj[i] for i in range(self.num_experts)]
            self._up_views = [self.up_proj[i] for i in range(self.num_experts)]
            self._down_views = [self.down_proj[i] for i in range(self.num_experts)]
        return self._gate_views, self._up_views, self._down_views

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
        
        gate_views, up_views, down_views = self._get_weight_views()
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)
            
            if not expert_mask.any():
                continue
            
            token_indices, slot_indices = torch.where(expert_mask)
            weights = expert_weights_flat[token_indices, slot_indices]
            
            # Gather input for this expert
            expert_input = x_flat.index_select(0, token_indices)
            
            # Compute gate and up projections together
            # Using transpose is more efficient than .t() as it's a view
            gate = torch.mm(expert_input, gate_views[expert_idx].T)
            up = torch.mm(expert_input, up_views[expert_idx].T)
            
            # Fused SiLU + multiply
            intermediate = self.fused_ops.fast_silu_mul(gate, up)
            
            # Down projection
            expert_output = torch.mm(intermediate, down_views[expert_idx].T)
            
            # Weighted accumulation
            self.fused_ops.fast_weighted_accum(
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
