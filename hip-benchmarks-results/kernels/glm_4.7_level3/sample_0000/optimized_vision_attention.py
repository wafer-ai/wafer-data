import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Custom kernel for element-wise add with residual
residual_add_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void residual_add_kernel(
    const float* a,
    const float* b,
    float* out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor residual_add_hip(
    torch::Tensor a,
    torch::Tensor b
) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    residual_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}
"""

# Load the residual add kernel
residual_add_module = load_inline(
    name="residual_add",
    cpp_sources=residual_add_cpp_source,
    functions=["residual_add_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized Vision Attention model with custom HIP kernels
    Uses optimized MultiheadAttention + custom residual addition
    """
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Use built-in optimized MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Standard LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Save reference to optimized kernel
        self.residual_add_kernel = residual_add_module
    
    def forward(self, x):
        """
        Forward pass with optimized residual addition
        """
        B, C, H, W = x.shape
        
        # Reshape to (seq_len, batch_size, embed_dim)
        x_seq = x.view(B, C, H * W).permute(2, 0, 1)
        
        # Use built-in attention (already highly optimized)
        attn_output, _ = self.attn(x_seq, x_seq, x_seq)
        
        # Apply custom residual addition kernel
        residual = self.residual_add_kernel.residual_add_hip(attn_output, x_seq)
        
        # Apply layer norm
        x = self.norm(residual)
        
        # Reshape back to (B, C, H, W)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        
        return x