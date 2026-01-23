import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized elementwise scale kernel
# Computes output = input * scale_factor
# Used to replace x * scaling_factor + x with x * (scaling_factor + 1.0)

elementwise_scale_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void elementwise_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Stride loop for better work distribution
    for (int i = idx; i < size; i += stride) {
        output[i] = input[i] * scale;
    }
}

torch::Tensor elementwise_scale_hip(
    torch::Tensor input,
    float scale
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int min_blocks = 256;
    const int num_blocks = max((size + block_size - 1) / block_size, min_blocks);
    
    elementwise_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size, scale
    );
    
    return output;
}
"""

elementwise_scale = load_inline(
    name="elementwise_scale",
    cpp_sources=elementwise_scale_cpp_source,
    functions=["elementwise_scale_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that eliminates the clone/detach operations
    and fuses the scaling into a single operation.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        
        # Keep the original Linear layer
        self.matmul = nn.Linear(in_features, out_features)
        
        # Pre-compute the fused scale factor
        # Original: x = x * scaling_factor + x
        # This is: x = x * (scaling_factor + 1.0)
        self.fused_scale = scaling_factor + 1.0
        
        self.elementwise_scale = elementwise_scale

    def forward(self, x):
        # Perform matmul using PyTorch's optimized implementation
        matmul_result = self.matmul(x)
        
        # Apply fused scaling using custom kernel
        # x = x * scaling_factor + x  becomes  x = x * (scaling_factor + 1.0)
        return self.elementwise_scale.elementwise_scale_hip(matmul_result, self.fused_scale)