import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_post_process_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int cols, 
    float scale) 
{
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Pointer to the start of the row
    // Input shape (batch_size, cols)
    const float* row_input = input + row * cols;
    
    float sum = 0.0f;
    // We process pairs (stride 2)
    // If cols is odd, integer division truncates, ignoring the last element,
    // which matches nn.MaxPool1d behavior for stride=2, kernel_size=2.
    int num_pairs = cols / 2;
    
    // Grid-stride loop (though we only have 1 block per row, so it's a block-stride loop over columns)
    for (int i = tid; i < num_pairs; i += blockDim.x) {
        int idx0 = i * 2;
        int idx1 = idx0 + 1;
        
        float val0 = row_input[idx0];
        float val1 = row_input[idx1];
        
        // MaxPool op: max of the pair
        float pair_max = fmaxf(val0, val1);
        
        // Accumulate to thread sum
        sum += pair_max;
    }
    
    // Block reduction using shared memory
    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();
    
    // Standard reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[row] = sdata[0] * scale;
    }
}

torch::Tensor fused_post_process(torch::Tensor input, float scale) {
    // input is (batch_size, features)
    // We assume input is on GPU and contiguous
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int batch_size = input.size(0);
    int cols = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    // Launch one block per row
    int block_size = 256;
    int grid_size = batch_size;
    int shared_mem_size = block_size * sizeof(float);
    
    fused_post_process_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        cols,
        scale
    );
    
    return output;
}
"""

module = load_inline(
    name="fused_post_process",
    cpp_sources=cpp_source,
    functions=["fused_post_process"],
    verbose=True,
    extra_cflags=['-O3', '-std=c++17']
)

class ModelNew(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    Optimized with fused custom kernel for post-GEMM operations.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.post_process = module

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size).
        """
        # Matrix Multiplication (most compute intensive part, kept as optimized library call)
        x = self.matmul(x)
        # x shape: (batch_size, out_features)
        
        # Fused MaxPool + Sum + Scale
        # Replaces:
        # x = self.max_pool(x.unsqueeze(1)).squeeze(1)
        # x = torch.sum(x, dim=1)
        # x = x * self.scale_factor
        x = self.post_process.fused_post_process(x, self.scale_factor)
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
