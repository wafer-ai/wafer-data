
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>
#include <torch/extension.h>

__global__ void __launch_bounds__(1024) fused_gelu_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    extern __shared__ float shared_mem[];
    float* row_data = shared_mem;
    float* sdata = &row_data[cols];

    int tid = threadIdx.x;
    int row_idx = blockIdx.x;

    if (row_idx >= rows) return;

    int row_offset = row_idx * cols;
    float local_max = -INFINITY;

    const float GELU_C1 = 0.7978845608f;
    const float GELU_C2 = 0.044715f;

    int vec_cols = cols / 4;
    
    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);
    float4* row_data_vec = reinterpret_cast<float4*>(row_data);

    // Pass 1: Load Global -> Shared. Compute GELU. Update Max.
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 in_val = input_vec[i];
        float4 out_val;

        #define GELU_APPROX(x) ({ \
            float _x = (x); \
            float _inner = GELU_C1 * (_x + GELU_C2 * _x * _x * _x); \
            0.5f * _x * (1.0f + tanhf(_inner)); \
        })

        out_val.x = GELU_APPROX(in_val.x);
        local_max = fmaxf(local_max, out_val.x);

        out_val.y = GELU_APPROX(in_val.y);
        local_max = fmaxf(local_max, out_val.y);

        out_val.z = GELU_APPROX(in_val.z);
        local_max = fmaxf(local_max, out_val.z);

        out_val.w = GELU_APPROX(in_val.w);
        local_max = fmaxf(local_max, out_val.w);
        #undef GELU_APPROX
        
        row_data_vec[i] = out_val;
    }
    
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max_val = sdata[0];

    // Pass 2: Shared -> Exp -> Shared. Update Sum.
    float local_sum = 0.0f;
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 val = row_data_vec[i];
        
        val.x = expf(val.x - row_max_val);
        local_sum += val.x;
        val.y = expf(val.y - row_max_val);
        local_sum += val.y;
        val.z = expf(val.z - row_max_val);
        local_sum += val.z;
        val.w = expf(val.w - row_max_val);
        local_sum += val.w;

        row_data_vec[i] = val;
    }

    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum_val = sdata[0];
    float inv_sum = 1.0f / row_sum_val;

    // Pass 3: Shared -> Global
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 val = row_data_vec[i];
        val.x *= inv_sum;
        val.y *= inv_sum;
        val.z *= inv_sum;
        val.w *= inv_sum;
        output_vec[i] = val;
    }
}

void fused_gelu_softmax_hip_(torch::Tensor input) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    float* data = input.data_ptr<float>();

    const int block_size = 1024;
    const int grid_size = rows;
    int shared_mem_size = (cols + block_size) * sizeof(float);

    fused_gelu_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(
        data,
        data,
        rows,
        cols
    );
}
"""

fused_ops = load_inline(
    name="fused_ops_v6",
    cpp_sources=cpp_source,
    functions=["fused_gelu_softmax_hip_"],
    verbose=True,
    extra_cflags=['-O3', '-ffast-math'],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_ops = fused_ops

    def forward(self, x):
        # GEMM + Bias (out-of-place from inputs, new tensor allocated)
        x = torch.addmm(self.linear.bias, x, self.linear.weight.t())
        
        # In-place Fused GELU + Softmax
        self.fused_ops.fused_gelu_softmax_hip_(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]
