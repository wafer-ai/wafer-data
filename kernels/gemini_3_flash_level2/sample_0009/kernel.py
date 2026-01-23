
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = float(scaling_factor)
        self.fused_weight = None
        self.fused_bias = None
        self.params_id = None
        
        # Optimize for MIOpen
        torch.backends.cudnn.benchmark = True
        torch.backends.miopen.benchmark = True

    def _get_params_id(self):
        return (
            id(self.conv.weight),
            id(self.conv.bias),
            id(self.bn.weight),
            id(self.bn.bias),
            id(self.bn.running_mean),
            id(self.bn.running_var),
            self.bn.eps,
            self.scaling_factor,
            self.training
        )

    def forward(self, x):
        if self.training:
            # Fallback to standard PyTorch during training
            x = self.conv(x)
            x = self.bn(x)
            x = x * self.scaling_factor
            return x
        
        # Check if we need to recalculate the fused parameters
        current_id = self._get_params_id()
        if self.fused_weight is None or current_id != self.params_id:
            with torch.no_grad():
                w = self.conv.weight
                b = self.conv.bias if self.conv.bias is not None else torch.zeros_like(self.bn.running_mean)
                gamma = self.bn.weight
                beta = self.bn.bias
                mean = self.bn.running_mean
                var = self.bn.running_var
                eps = self.bn.eps
                
                inv_std = torch.rsqrt(var + eps)
                fused_scale = (gamma * inv_std * self.scaling_factor).view(-1, 1, 1, 1)
                self.fused_weight = w * fused_scale
                self.fused_bias = ((b - mean) * inv_std * gamma + beta) * self.scaling_factor
                self.params_id = current_id
        
        return F.conv2d(x, self.fused_weight, self.fused_bias)

def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 128, 128
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    in_channels = 8
    out_channels = 64
    kernel_size = 3
    scaling_factor = 2.0
    return [in_channels, out_channels, kernel_size, scaling_factor]
