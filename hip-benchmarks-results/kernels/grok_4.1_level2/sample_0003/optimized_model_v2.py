import torch
import torch.nn as nn

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        attn = self.linear(x)
        res = attn.detach()
        attn = attn * self.scaling_factor
        return attn + res
