import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CXX to hipcc for ROCm
os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void cross_entropy_kernel(
    const float* __restrict__ logits,
    const long* __restrict__ targets,
    float* __restrict__ losses,
    int num_classes,
    int batch_size) 
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    
    // Base pointer for this row
    const float* row_ptr = logits + batch_idx * num_classes;
    // We assume num_classes (4096) is divisible by 4 for float4 vectorization.
    const float4* row_vec = reinterpret_cast<const float4*>(row_ptr);
    int num_vec = num_classes / 4;

    float thread_max = -1e30f;

    // 1. Max Pass
    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 v = row_vec[i];
        float local_max = max(v.x, v.y);
        local_max = max(local_max, v.z);
        local_max = max(local_max, v.w);
        thread_max = max(thread_max, local_max);
    }

    // Block Reduce Max
    __shared__ float sdata[256];
    sdata[tid] = thread_max;
    __syncthreads();
    
    if (tid < 128) sdata[tid] = max(sdata[tid], sdata[tid + 128]); __syncthreads();
    if (tid < 64) sdata[tid] = max(sdata[tid], sdata[tid + 64]); __syncthreads();
    if (tid < 32) sdata[tid] = max(sdata[tid], sdata[tid + 32]); __syncthreads();
    if (tid < 16) sdata[tid] = max(sdata[tid], sdata[tid + 16]); __syncthreads();
    if (tid < 8) sdata[tid] = max(sdata[tid], sdata[tid + 8]); __syncthreads();
    if (tid < 4) sdata[tid] = max(sdata[tid], sdata[tid + 4]); __syncthreads();
    if (tid < 2) sdata[tid] = max(sdata[tid], sdata[tid + 2]); __syncthreads();
    if (tid < 1) sdata[tid] = max(sdata[tid], sdata[tid + 1]); __syncthreads();
    
    float row_max = sdata[0];

    // 2. Sum Exp Pass
    float thread_sum = 0.0f;
    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 v = row_vec[i];
        thread_sum += expf(v.x - row_max) + expf(v.y - row_max) + expf(v.z - row_max) + expf(v.w - row_max);
    }

    // Block Reduce Sum
    sdata[tid] = thread_sum;
    __syncthreads();

    if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads();
    if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads();
    if (tid < 32) sdata[tid] += sdata[tid + 32]; __syncthreads();
    if (tid < 16) sdata[tid] += sdata[tid + 16]; __syncthreads();
    if (tid < 8) sdata[tid] += sdata[tid + 8]; __syncthreads();
    if (tid < 4) sdata[tid] += sdata[tid + 4]; __syncthreads();
    if (tid < 2) sdata[tid] += sdata[tid + 2]; __syncthreads();
    if (tid < 1) sdata[tid] += sdata[tid + 1]; __syncthreads();
    
    float row_sum = sdata[0];

    // 3. Final Compute
    if (tid == 0) {
        long target = targets[batch_idx];
        float target_val = row_ptr[target]; 
        losses[batch_idx] = -target_val + logf(row_sum) + row_max;
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor logits, torch::Tensor targets) {
    auto batch_size = logits.size(0);
    auto num_classes = logits.size(1);
    
    // Output buffer
    auto losses = torch::empty({batch_size}, logits.options());
    
    // Ensure inputs are contiguous
    if (!logits.is_contiguous()) logits = logits.contiguous();
    if (!targets.is_contiguous()) targets = targets.contiguous();

    cross_entropy_kernel<<<batch_size, 256>>>(
        logits.data_ptr<float>(),
        targets.data_ptr<long>(),
        losses.data_ptr<float>(),
        num_classes,
        batch_size
    );
    
    return losses.mean();
}
"""

cross_entropy_module = load_inline(
    name='cross_entropy_module',
    cpp_sources=cpp_source,
    functions=['cross_entropy_hip'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.ce_op = cross_entropy_module

    def forward(self, predictions, targets):
        return self.ce_op.cross_entropy_hip(predictions, targets)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda(), torch.randint(0, num_classes, (batch_size,)).cuda()]

def get_init_inputs():
    return []
