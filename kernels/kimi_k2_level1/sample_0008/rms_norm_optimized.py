import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rms_norm_cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void rms_norm_kernel(
    const float* x,
    float* out,
    const float eps,
    const int total_positions,
    const int features
) {
    // Each thread block processes one position (b, dim1, dim2)
    // Multiple threads collaborate to process the features dimension
    
    const int pos = blockIdx.x;
    if (pos >= total_positions) return;
    
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    
    // Feature size is 64, so 64 threads per block
    __shared__ float s_sum_squares[64];
    
    // Load and square one feature element per thread
    const int offset = pos * features + tid;
    const float value = x[offset];
    s_sum_squares[tid] = value * value;
    __syncthreads();
    
    // Reduce: compute sum of squares across threads
    // Since features = 64 = 2^6, we can do binary reduction
    for (int stride = 16; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            s_sum_squares[tid] += s_sum_squares[tid + 2 * stride];
        }
        __syncthreads();
    }
    
    // Thread 0 computes RMS and stores to shared memory
    __shared__ float inv_rms;
    if (tid == 0) {
        const float rms = sqrtf(s_sum_squares[0] / features + eps);
        inv_rms = 1.0f / rms;
    }
    __syncthreads();
    
    // All threads normalize their element
    out[offset] = value * inv_rms;
}

torch::Tensor rms_norm_hip(torch::Tensor x, float eps) {
    const int batch_size = x.size(0);
    const int features = x.size(1);
    const int dim1 = x.size(2);
    const int dim2 = x.size(3);
    
    const int total_positions = batch_size * dim1 * dim2;
    
    auto out = torch::empty_like(x);
    
    const int block_size = 64;  // Must match features dimension
    const int num_blocks = total_positions;
    
    rms_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        eps,
        total_positions,
        features
    );
    
    return out;
}
"""

rms_norm_hip = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    functions=["rms_norm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_hip = rms_norm_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm_hip.rms_norm_hip(x, self.eps)

# Input generation functions (same as reference)
batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]
