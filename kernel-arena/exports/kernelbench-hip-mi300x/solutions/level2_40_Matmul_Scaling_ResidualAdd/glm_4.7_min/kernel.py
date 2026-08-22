import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that mathematically combines scaling and residual addition.
    
    Original: 
        x = self.matmul(x)
        original_x = x.clone().detach()
        x = x * self.scaling_factor
        x = x + original_x
    
    Optimized:
        x = self.matmul(x)
        x = x * (1 + self.scaling_factor)
    
    Mathematical equivalence: x * scaling_factor + x = x * (scaling_factor + 1)
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        # Combined factor: 1 + scaling_factor
        self.combined_factor = 1.0 + scaling_factor

    def forward(self, x):
        """
        Forward pass of the optimized model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        # Fused scaling + residual addition in single operation
        x = x * self.combined_factor
        return x

# Must keep the same config for compatibility
batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]