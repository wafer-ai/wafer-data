import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# Fused RoPE + QKV reshape kernel - applies rotary embeddings and reshapes efficiently
fused_ops_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused RoPE kernel with vectorized loads/stores
__global__ void fused_rope_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Each thread handles 4 elements (float4)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int elements_per_head = seq_len * head_dim;
    int elements_per_batch = num_heads * elements_per_head;
    int total_threads_needed = batch_size * num_heads * seq_len * (head_dim / 4);
    
    if (tid >= total_threads_needed) return;
    
    // Decode which float4 we're processing
    int d4 = tid % (head_dim / 4);  // Which float4 within head_dim
    int remaining = tid / (head_dim / 4);
    int s = remaining % seq_len;
    remaining = remaining / seq_len;
    int h = remaining % num_heads;
    int b = remaining / num_heads;
    
    int d = d4 * 4;  // Starting dimension index
    
    // Base index in input/output
    int base_idx = b * elements_per_batch + h * elements_per_head + s * head_dim + d;
    int cos_sin_base = s * head_dim + d;
    
    // Process 4 elements at a time
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int curr_d = d + i;
        int curr_idx = base_idx + i;
        int curr_cos_sin = cos_sin_base + i;
        
        float cos_val = cos[curr_cos_sin];
        float sin_val = sin[curr_cos_sin];
        float x = input[curr_idx];
        float x_rotated;
        
        if (curr_d < half_dim) {
            int other_idx = base_idx - d + (curr_d + half_dim);
            x_rotated = -input[other_idx];
        } else {
            int other_idx = base_idx - d + (curr_d - half_dim);
            x_rotated = input[other_idx];
        }
        
        output[curr_idx] = x * cos_val + x_rotated * sin_val;
    }
}

torch::Tensor fused_rope_hip(torch::Tensor input, torch::Tensor cos, torch::Tensor sin) {
    auto batch_size = input.size(0);
    auto num_heads = input.size(1);
    auto seq_len = input.size(2);
    auto head_dim = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int total_threads = batch_size * num_heads * seq_len * (head_dim / 4);
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    
    fused_rope_kernel_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim
    );
    
    return output;
}

// Efficient KV expansion using view/expand (no memory copy)
// Returns expanded key/value tensors
std::vector<torch::Tensor> expand_kv_hip(
    torch::Tensor K,
    torch::Tensor V,
    int num_key_value_groups
) {
    // K, V: [batch, num_kv_heads, seq_len, head_dim]
    auto batch = K.size(0);
    auto num_kv_heads = K.size(1);
    auto seq_len = K.size(2);
    auto head_dim = K.size(3);
    
    // Use expand + reshape for memory-efficient expansion
    // This creates a view without copying data
    auto K_expanded = K.unsqueeze(2)
                       .expand({batch, num_kv_heads, num_key_value_groups, seq_len, head_dim})
                       .reshape({batch, num_kv_heads * num_key_value_groups, seq_len, head_dim});
    
    auto V_expanded = V.unsqueeze(2)
                       .expand({batch, num_kv_heads, num_key_value_groups, seq_len, head_dim})
                       .reshape({batch, num_kv_heads * num_key_value_groups, seq_len, head_dim});
    
    return {K_expanded.contiguous(), V_expanded.contiguous()};
}
"""

fused_ops_cpp = """
torch::Tensor fused_rope_hip(torch::Tensor input, torch::Tensor cos, torch::Tensor sin);
std::vector<torch::Tensor> expand_kv_hip(torch::Tensor K, torch::Tensor V, int num_key_value_groups);
"""

fused_ops = load_inline(
    name="fused_ops_v2",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["fused_rope_hip", "expand_kv_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute cos/sin for common sequence lengths
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        if seq_len != self._cached_seq_len or self._cached_cos is None or self._cached_cos.device != x.device:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cached_cos = emb.cos().unsqueeze(0).unsqueeze(0).contiguous()
            self._cached_sin = emb.sin().unsqueeze(0).unsqueeze(0).contiguous()
            self._cached_seq_len = seq_len
            
        return self._cached_cos, self._cached_sin


class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout
        self.softmax_scale = head_dim ** (-0.5)

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Rotary embeddings with caching
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        self.fused_ops = fused_ops

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V - these are the main compute bottlenecks
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention - use view for zero-copy
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Get cached cos/sin
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        
        # Apply fused RoPE kernel
        query_states = self.fused_ops.fused_rope_hip(query_states, cos, sin)
        key_states = self.fused_ops.fused_rope_hip(key_states, cos, sin)

        # Efficient KV expansion
        key_states, value_states = self.fused_ops.expand_kv_hip(
            key_states, value_states, self.num_key_value_groups
        )
        
        # Use scaled_dot_product_attention - highly optimized for AMD GPUs
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.softmax_scale,
        )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output


def custom_kernel(inputs):
    hidden_states = inputs[0]
    
    model = ModelNew(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
    ).cuda().eval()
    
    with torch.no_grad():
        return model(hidden_states)
