import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>
#include <cmath>
#include <cfloat>

#define BLOCK_SIZE 512

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* input_row = input + row * dim;
    float* output_row = output + row * dim;
    
    // Each thread calculates a portion of the row
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input_row[i];
        if (val > max_val) max_val = val;
    }
    
    // Shared memory for block reduction
    __shared__ float shared_max[BLOCK_SIZE];
    shared_max[threadIdx.x] = max_val;
    __syncthreads();
    
    // Block reduction for max
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            float other = shared_max[threadIdx.x + offset];
            if (other > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = expf(input_row[i] - row_max);
        output_row[i] = val;
        sum_val += val;
    }
    
    // Reuse shared memory for sum
    shared_max[threadIdx.x] = sum_val;
    __syncthreads();
    
    // Block reduction for sum
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_max[threadIdx.x] += shared_max[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    float row_sum = shared_max[0];
    float inv_sum = 1.0f / row_sum;
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output_row[i] *= inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    
    // Create output tensor
    auto output = torch::empty_like(input, torch::TensorOptions().memory_format(torch::MemoryFormat::Contiguous));
    
    // Ensure input is contiguous
    input = input.contiguous();
    
    const int block_size = BLOCK_SIZE;
    const int num_blocks = batch_size;
    
    softmax_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
    
    return output;
}
"""

softmax_hip = load_inline(
    name="softmax_hip",
    cpp_sources=softmax_hip_source,
    functions=["softmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_hip = softmax_hip
    
    def forward(self, x):
        return self.softmax_hip.softmax_hip(x.cuda())

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
