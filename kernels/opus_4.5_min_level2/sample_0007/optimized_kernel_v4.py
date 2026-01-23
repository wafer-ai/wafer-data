import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Bias + Swish + Scaling kernel
bias_swish_scale_cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Fast sigmoid using __expf intrinsic
__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Fused: (x + bias) * sigmoid(x + bias) * scale
__global__ void bias_swish_scale_kernel(const float* __restrict__ input,
                                         const float* __restrict__ bias,
                                         float* __restrict__ output, 
                                         const float scaling_factor,
                                         const int batch_size,
                                         const int out_features) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_features;
    
    // Each thread processes 4 elements
    const int base_idx = tid * 4;
    
    if (base_idx + 3 < total_elements) {
        // Calculate bias indices for the 4 elements
        int idx0 = base_idx;
        int idx1 = base_idx + 1;
        int idx2 = base_idx + 2;
        int idx3 = base_idx + 3;
        
        // Get bias for each element (bias is per-column)
        float b0 = bias[idx0 % out_features];
        float b1 = bias[idx1 % out_features];
        float b2 = bias[idx2 % out_features];
        float b3 = bias[idx3 % out_features];
        
        // Load input
        float4 in_val = *reinterpret_cast<const float4*>(input + base_idx);
        
        // Add bias and apply swish + scale
        float x0 = in_val.x + b0;
        float x1 = in_val.y + b1;
        float x2 = in_val.z + b2;
        float x3 = in_val.w + b3;
        
        float4 out_val;
        out_val.x = x0 * fast_sigmoid(x0) * scaling_factor;
        out_val.y = x1 * fast_sigmoid(x1) * scaling_factor;
        out_val.z = x2 * fast_sigmoid(x2) * scaling_factor;
        out_val.w = x3 * fast_sigmoid(x3) * scaling_factor;
        
        *reinterpret_cast<float4*>(output + base_idx) = out_val;
    } else if (base_idx < total_elements) {
        // Handle remaining elements
        for (int i = base_idx; i < total_elements && i < base_idx + 4; i++) {
            float x = input[i] + bias[i % out_features];
            output[i] = x * fast_sigmoid(x) * scaling_factor;
        }
    }
}

torch::Tensor bias_swish_scale_hip(torch::Tensor input, torch::Tensor bias, float scaling_factor) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
    const int batch_size = input.size(0);
    const int out_features = input.size(1);
    const int total_elements = batch_size * out_features;
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_threads_needed = (total_elements + 3) / 4; // Each thread handles 4 elements
    const int num_blocks = (num_threads_needed + block_size - 1) / block_size;
    
    bias_swish_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(),
        output.data_ptr<float>(), 
        scaling_factor,
        batch_size,
        out_features
    );
    
    return output;
}
"""

bias_swish_scale_cpp_decl = """
torch::Tensor bias_swish_scale_hip(torch::Tensor input, torch::Tensor bias, float scaling_factor);
"""

bias_swish_scale_module = load_inline(
    name="bias_swish_scale_v4",
    cpp_sources=bias_swish_scale_cpp_decl,
    cuda_sources=bias_swish_scale_cpp_source,
    functions=["bias_swish_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model: matmul (no bias) + fused bias+swish+scaling
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Use a Linear layer but we'll handle bias separately
        self.matmul = nn.Linear(in_features, out_features, bias=True)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Do matmul without bias
        x = F.linear(x, self.matmul.weight, None)
        # Fused bias + swish + scaling
        x = bias_swish_scale_module.bias_swish_scale_hip(x, self.matmul.bias, self.scaling_factor)
        return x
