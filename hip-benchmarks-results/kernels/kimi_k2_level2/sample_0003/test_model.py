import torch
import torch.nn as nn
import sys
sys.path.append('/root/Wafer/research/KernelBench/KernelBench/level2')

# Load the reference model
exec(open('/root/Wafer/research/KernelBench/KernelBench/level2/40_Matmul_Scaling_ResidualAdd.py').read())

# Create reference model
in_features = 4096
out_features = 4096
scaling_factor = 0.5
ref_model = Model(in_features, out_features, scaling_factor)
ref_model = ref_model.cuda()

# Get input
batch_size = 16384
x = torch.rand(batch_size, in_features).cuda()

# Run reference
with torch.no_grad():
    output_ref = ref_model(x)

print("Reference output shape:", output_ref.shape)
print("Output range:", output_ref.min().item(), "to", output_ref.max().item())
print("First few values:", output_ref[0, :5])

# Check weight shape
print("Weight shape:", ref_model.matmul.weight.shape)
print("Bias shape:", ref_model.matmul.bias.shape)
print("Weight sample:", ref_model.matmul.weight[0, :5])
print("Bias sample:", ref_model.matmul.bias[:5])