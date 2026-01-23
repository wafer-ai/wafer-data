import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.linear(x)
        num_pools = self.out_features // self.kernel_size
        x = x.view(x.size(0), num_pools, self.kernel_size)
        pooled = torch.amax(x, dim=-1)
        x = pooled.sum(dim=1) * self.scale_factor
        return x

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
