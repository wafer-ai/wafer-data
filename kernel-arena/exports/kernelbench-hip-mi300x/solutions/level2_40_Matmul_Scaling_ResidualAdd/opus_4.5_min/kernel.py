import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized model that avoids the clone().detach() operation.
    
    The original computation: x * scaling_factor + x
    Optimized to: x * (1 + scaling_factor)
    
    This avoids:
    1. The clone() allocation
    2. The detach() call
    3. The separate multiplication and addition operations
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        # Pre-compute the combined factor to avoid runtime computation
        self.combined_factor = 1.0 + scaling_factor

    def forward(self, x):
        # Perform linear transformation using PyTorch (uses optimized rocBLAS)
        x = self.matmul(x)
        # Single multiplication replaces: clone, detach, scale, add
        return x * self.combined_factor
