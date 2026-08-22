import os

import torch
import torch.nn as nn

os.environ["CXX"] = "hipcc"


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Keep the optimized cuBLAS/rocBLAS matmul
        x = self.matmul(x)
        
        # Use torch.compile to optimize the remaining operations
        # This should fuse operations at the kernel level
        pooled_out = torch.nn.functional.max_pool1d(
            x.unsqueeze(1), 
            kernel_size=self.kernel_size, 
            stride=self.kernel_size
        ).squeeze(1)
        
        sum_out = torch.sum(pooled_out, dim=1)
        result = sum_out * self.scale_factor
        
        return result


# Use torch.compile to optimize the forward pass
model_new_factory = lambda in_features, out_features, kernel_size, scale_factor: torch.compile(
    ModelNew(in_features, out_features, kernel_size, scale_factor),
    mode="max-autotune"
)