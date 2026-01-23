import torch
import torch.nn as nn

# Test what the reference model produces
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

torch.manual_seed(42)
model = Model(in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max)
x = torch.rand(batch_size, in_channels, height, width)

with torch.no_grad():
    y = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Output stats - min: {y.min()}, max: {y.max()}, mean: {y.mean()}")
print(f"Some sample values:")
print(y[0, 0, 0, :5])
print(y[0, 1, 0, :5])