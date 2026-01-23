import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Fused HIP kernels for AMD ROCm
hip_code = """
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>

// Fused Linear + Dropout kernel
// Each thread computes one output element
__global__ void linear_dropout_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float dropout_prob,
    unsigned long long seed
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // Compute dot product: sum_k input[row, k] * weight[col, k] + bias[col]
        float sum = 0.0f;
        const float* input_row = &input[row * in_features];
        const float* weight_row = &weight[col * in_features];
        
        #pragma unroll 16
        for (int i = 0; i < in_features; i++) {
            sum += input_row[i] * weight_row[i];
        }
        sum += bias[col];
        
        // Apply dropout using rocrand
        rocrand_state_xorwow state;
        rocrand_init(seed, row * out_features + col, 0, &state);
        
        float rand = rocrand_uniform(&state);
        float mask = (rand > dropout_prob) ? 1.0f / (1.0f - dropout_prob) : 0.0f;
        
        output[row * out_features + col] = sum * mask;
    }
}

// Optimized softmax kernel with parallel reduction
// Each block processes one row (batch element)
__global__ void softmax_kernel(
    float* input_output,
    int batch_size,
    int features
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    extern __shared__ float smem[];
    
    // Set all shared memory to 0
    smem[tid] = 0.0f;
    __syncthreads();
    
    // Step 1: Find maximum value in the row
    float local_max = -INFINITY;
    for (int i = tid; i < features; i += block_size) {
        float val = input_output[row * features + i];
        if (val > local_max) local_max = val;
    }
    
    // Store local max in shared memory
    smem[tid] = local_max;
    __syncthreads();
    
    // Parallel reduction for max
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < block_size) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    
    float row_max = smem[0];
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < features; i += block_size) {
        float exp_val = expf(input_output[row * features + i] - row_max);
        input_output[row * features + i] = exp_val;
        local_sum += exp_val;
    }
    
    // Store local sum in shared memory
    smem[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction for sum
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < block_size) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    float row_sum = smem[0];
    __syncthreads();
    
    // Step 3: Normalize
    for (int i = tid; i < features; i += block_size) {
        input_output[row * features + i] /= row_sum;
    }
}

// Wrapper functions
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor linear_dropout_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_prob,
    unsigned long long seed
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, 
                               torch::dtype(torch::kFloat32).device(input.device()));
    
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (batch_size + 31) / 32);
    
    // Use default stream (0) - this works for both CUDA and HIP
    hipLaunchKernelGGL(
        linear_dropout_kernel,
        grid,
        block,
        0,
        0,  // Use default stream
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        dropout_prob,
        seed
    );
    
    return output;
}

torch::Tensor softmax_hip(torch::Tensor input_output) {
    CHECK_INPUT(input_output);
    
    auto batch_size = input_output.size(0);
    auto features = input_output.size(1);
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = threads * sizeof(float);
    
    // Use default stream (0) - this works for both CUDA and HIP
    hipLaunchKernelGGL(
        softmax_kernel,
        blocks,
        threads,
        shared_mem,
        0,  // Use default stream
        input_output.data_ptr<float>(),
        batch_size,
        features
    );
    
    return input_output;
}
"""

def my_load_inline(name, cpp_sources, functions, **kwargs):
    """Modified version of load_inline to handle ROCm libraries."""
    if "extra_ldflags" not in kwargs or len(kwargs["extra_ldflags"]) == 0:
        kwargs["extra_ldflags"] = ["-lrocrand"]
    return load_inline(name=name, cpp_sources=cpp_sources, functions=functions, **kwargs)

custom_ops = my_load_inline(
    name="custom_ops",
    cpp_sources=hip_code,
    functions=["linear_dropout_hip", "softmax_hip"],
    verbose=True,
    extra_cflags=["-O3", "-D__HIP_PLATFORM_AMD__"],
    extra_ldflags=["-lrocrand"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        # Initialize weight and bias (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.custom_ops = custom_ops
        self.rng_counter = 0
        
    def forward(self, x):
        # Ensure contiguous tensors
        x = x.contiguous()
        
        if not self.weight.is_contiguous():
            self.weight.data = self.weight.data.contiguous()
        if not self.bias.is_contiguous():
            self.bias.data = self.bias.data.contiguous()
        
        # Generate unique seed for dropout
        seed = (torch.randint(0, 1<<30, (1,), device='cuda').item() + self.rng_counter) & 0xFFFFFFFFFFFFFFFF
        self.rng_counter = (self.rng_counter + 1) % (1 << 20)
        
        # Fused linear + dropout
        out = self.custom_ops.linear_dropout_hip(
            x,
            self.weight,
            self.bias,
            self.dropout_p,
            seed
        )
        
        # Optimized softmax
        out = self.custom_ops.softmax_hip(out)
        
        return out

# Test inputs
batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, dropout_p]