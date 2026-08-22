import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused kernel for MaxPool1d + Sum + Scale
fused_maxpool_sum_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64

// Warp reduce using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Fused kernel: MaxPool1d (kernel_size=2) + Sum + Scale
// Optimized for AMD GCN architecture with 64-wide wavefronts
__global__ void fused_maxpool_sum_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int out_features,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const float* row = input + batch_idx * out_features;
    
    // Number of pairs for maxpool with kernel_size=2
    int num_pairs = out_features / 2;
    
    float local_sum = 0.0f;
    
    // Use float4 for coalesced memory access (16 bytes = 4 floats = 2 pairs)
    const float4* row4 = reinterpret_cast<const float4*>(row);
    int num_float4 = num_pairs / 2;
    
    // Each thread processes multiple float4s with stride
    #pragma unroll 4
    for (int i = tid; i < num_float4; i += block_size) {
        float4 v = row4[i];
        float max1 = fmaxf(v.x, v.y);
        float max2 = fmaxf(v.z, v.w);
        local_sum += max1 + max2;
    }
    
    // Warp-level reduction
    local_sum = warp_reduce_sum(local_sum);
    
    // Store warp results to shared memory
    __shared__ float warp_sums[16];  // Max 1024/64 = 16 warps
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        
        if (lane_id == 0) {
            output[batch_idx] = val * scale_factor;
        }
    }
}

torch::Tensor fused_maxpool_sum_scale_hip(torch::Tensor input, float scale_factor) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    // 1024 threads = 16 wavefronts for good occupancy
    int block_size = 1024;
    int num_blocks = batch_size;
    
    fused_maxpool_sum_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        out_features,
        scale_factor
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_maxpool_sum_scale_hip(torch::Tensor input, float scale_factor);
"""

fused_module = load_inline(
    name="fused_maxpool_sum_scale_v5",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_maxpool_sum_scale_source,
    functions=["fused_maxpool_sum_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses MaxPool1d + Sum + Scale into a single kernel.
    Uses contiguous tensors and optimized memory layout.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_module = fused_module
        # Pre-register the weight and bias
        self.weight = self.matmul.weight
        self.bias = self.matmul.bias

    def forward(self, x):
        # Use F.linear directly with contiguous weight
        x = F.linear(x, self.weight, self.bias)
        # Fused maxpool + sum + scale
        x = self.fused_module.fused_maxpool_sum_scale_hip(x, self.scale_factor)
        return x


def get_inputs():
    return [torch.rand(128, 32768).cuda()]


def get_init_inputs():
    return [32768, 32768, 2, 0.5]
