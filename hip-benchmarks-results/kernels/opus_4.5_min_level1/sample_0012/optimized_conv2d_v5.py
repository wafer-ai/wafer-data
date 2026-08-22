import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use batched im2col with unfold + batched matmul - all in pure PyTorch but optimized
# This approach uses tensor operations that map well to optimized BLAS routines

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights same as nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        in_height = x.size(2)
        in_width = x.size(3)
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Apply padding if necessary
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # Use unfold to extract patches - this is im2col
        # unfold extracts all sliding windows from the input
        # x shape: [batch, in_channels, padded_height, padded_width]
        x_unf = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # x_unf shape: [batch, in_channels, out_height, out_width, kernel_size, kernel_size]
        
        # Reshape for batched matmul
        x_unf = x_unf.contiguous().view(batch_size, self.in_channels * self.kernel_size * self.kernel_size, out_height * out_width)
        # x_unf shape: [batch, in_channels * k * k, out_height * out_width]
        
        # Reshape weights for matmul
        weight_flat = self.weight.view(self.out_channels, -1)
        # weight_flat shape: [out_channels, in_channels * k * k]
        
        # Batched matrix multiplication
        # [out_channels, in_channels*k*k] @ [batch, in_channels*k*k, out_spatial]
        # -> [batch, out_channels, out_spatial]
        output = torch.bmm(weight_flat.unsqueeze(0).expand(batch_size, -1, -1), x_unf)
        
        # Reshape to output
        output = output.view(batch_size, self.out_channels, out_height, out_width)
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output
