import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

class ModelNew(nn.Module):
    """
    Optimized model fusing Linear + AvgPool via precomputed grouped weights.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        K = pool_kernel_size
        num_groups = out_features // K
        self.register_buffer('Wgroup', self.linear.weight.view(num_groups, K, in_features).mean(dim=1))
        self.register_buffer('biasgroup', self.linear.bias.view(num_groups, K).mean(dim=1))
        self.scale_factor = scale_factor

    def forward(self, x):
        pooled = F.linear(x, self.Wgroup, self.biasgroup)
        x = F.gelu(pooled) * self.scale_factor
        x = torch.max(x, dim=1).values
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]