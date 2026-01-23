import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused bias + divide + GELU kernel
fused_bias_div_gelu_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float gelu_tanh_approx(float x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Kernel for fused bias + divide + GELU
// Input shape: (batch_size, output_size)
// Bias shape: (output_size,)
__global__ void fused_bias_div_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float inv_divisor,
    const int batch_size,
    const int output_size
) {
    // Each thread processes multiple elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_size;
    
    // Vectorized processing - 4 elements at a time
    int vec_tid = tid * 4;
    
    if (vec_tid + 3 < total_elements) {
        // Calculate column indices for bias access
        int col0 = vec_tid % output_size;
        int col1 = (vec_tid + 1) % output_size;
        int col2 = (vec_tid + 2) % output_size;
        int col3 = (vec_tid + 3) % output_size;
        
        // Load 4 input elements
        float4 in_vec = *reinterpret_cast<const float4*>(&input[vec_tid]);
        
        // Add bias, divide, and apply GELU
        float4 out_vec;
        out_vec.x = gelu_tanh_approx((in_vec.x + bias[col0]) * inv_divisor);
        out_vec.y = gelu_tanh_approx((in_vec.y + bias[col1]) * inv_divisor);
        out_vec.z = gelu_tanh_approx((in_vec.z + bias[col2]) * inv_divisor);
        out_vec.w = gelu_tanh_approx((in_vec.w + bias[col3]) * inv_divisor);
        
        // Store results
        *reinterpret_cast<float4*>(&output[vec_tid]) = out_vec;
    } else if (vec_tid < total_elements) {
        // Handle remainder
        for (int i = vec_tid; i < total_elements && i < vec_tid + 4; i++) {
            int col = i % output_size;
            float val = (input[i] + bias[col]) * inv_divisor;
            output[i] = gelu_tanh_approx(val);
        }
    }
}

// Simple fused divide + GELU kernel (no bias - bias handled in matmul)
__global__ void fused_div_gelu_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float inv_divisor,
    const int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per iteration
    for (int idx = tid * 4; idx < size; idx += stride * 4) {
        if (idx + 3 < size) {
            float4 in_vec = *reinterpret_cast<const float4*>(&input[idx]);
            float4 out_vec;
            
            float v0 = in_vec.x * inv_divisor;
            float v1 = in_vec.y * inv_divisor;
            float v2 = in_vec.z * inv_divisor;
            float v3 = in_vec.w * inv_divisor;
            
            out_vec.x = gelu_tanh_approx(v0);
            out_vec.y = gelu_tanh_approx(v1);
            out_vec.z = gelu_tanh_approx(v2);
            out_vec.w = gelu_tanh_approx(v3);
            
            *reinterpret_cast<float4*>(&output[idx]) = out_vec;
        } else {
            for (int i = idx; i < size && i < idx + 4; i++) {
                float val = input[i] * inv_divisor;
                output[i] = gelu_tanh_approx(val);
            }
        }
    }
}

torch::Tensor fused_bias_div_gelu_hip(
    torch::Tensor input, 
    torch::Tensor bias,
    float divisor
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int output_size = input.size(1);
    int total = batch_size * output_size;
    
    const float inv_divisor = 1.0f / divisor;
    
    const int block_size = 256;
    const int num_blocks = (total + block_size * 4 - 1) / (block_size * 4);
    
    fused_bias_div_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        inv_divisor,
        batch_size,
        output_size
    );
    
    return output;
}

torch::Tensor fused_div_gelu_hip_v2(torch::Tensor input, float divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const float inv_divisor = 1.0f / divisor;
    
    const int block_size = 256;
    const int num_blocks = min((size + block_size * 4 - 1) / (block_size * 4), 65535);
    
    fused_div_gelu_kernel_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        inv_divisor,
        size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_bias_div_gelu_hip(torch::Tensor input, torch::Tensor bias, float divisor);
torch::Tensor fused_div_gelu_hip_v2(torch::Tensor input, float divisor);
"""

fused_module = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=fused_bias_div_gelu_source,
    functions=["fused_bias_div_gelu_hip", "fused_div_gelu_hip_v2"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses bias + divide + GELU operations.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        # Use Linear without bias, we'll fuse bias into our kernel
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.divisor = divisor
        
        # Initialize the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Matrix multiplication without bias
        x = F.linear(x, self.weight, None)
        # Fused bias + divide + GELU
        x = fused_module.fused_bias_div_gelu_hip(x, self.bias, self.divisor)
        return x
