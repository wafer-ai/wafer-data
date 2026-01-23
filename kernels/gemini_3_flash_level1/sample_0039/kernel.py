
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cumprod_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WAVEFRONT_SIZE 64

__device__ __forceinline__ float wavefront_scan(float val, int lane) {
    for (int i = 1; i < WAVEFRONT_SIZE; i <<= 1) {
        float tmp = __shfl_up(val, i, WAVEFRONT_SIZE);
        if (lane >= i) val *= tmp;
    }
    return val;
}

__global__ void cumprod_kernel_2d_dim1_vectorized(const float4* __restrict__ input, float4* __restrict__ output, int N, int M_v4) {
    int row = blockIdx.x;
    if (row >= N) return;

    const int bsize = 512;
    const int n_waves = bsize / WAVEFRONT_SIZE;
    
    __shared__ float wave_prods[n_waves];
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane = tid % WAVEFRONT_SIZE;

    int row_offset = row * M_v4;
    float current_prefix = 1.0f;

    for (int chunk_start = 0; chunk_start < M_v4; chunk_start += bsize) {
        int col = chunk_start + tid;
        
        float4 vals;
        if (col < M_v4) {
            vals = input[row_offset + col];
        } else {
            vals = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        }
        
        float p0 = vals.x;
        float p1 = p0 * vals.y;
        float p2 = p1 * vals.z;
        float p3 = p2 * vals.w;
        
        float scanned_p3 = wavefront_scan(p3, lane);
        
        if (lane == WAVEFRONT_SIZE - 1) {
            wave_prods[wave_id] = scanned_p3;
        }
        __syncthreads();
        
        if (wave_id == 0) {
            float wp = (lane < n_waves) ? wave_prods[lane] : 1.0f;
            float swp = wavefront_scan(wp, lane);
            if (lane < n_waves) {
                wave_prods[lane] = swp;
            }
        }
        __syncthreads();
        
        float prev_wave_prod = (wave_id > 0) ? wave_prods[wave_id - 1] : 1.0f;
        float total_prefix = current_prefix * prev_wave_prod;
        
        float thread_prefix = __shfl_up(scanned_p3, 1, WAVEFRONT_SIZE);
        if (lane == 0) thread_prefix = 1.0f;
        
        float final_prefix = total_prefix * thread_prefix;
        
        if (col < M_v4) {
            output[row_offset + col] = make_float4(p0 * final_prefix, p1 * final_prefix, p2 * final_prefix, p3 * final_prefix);
        }
        
        float chunk_total_prod = wave_prods[n_waves - 1];
        current_prefix *= chunk_total_prod;
        __syncthreads();
    }
}

torch::Tensor cumprod_hip(torch::Tensor x, int64_t dim) {
    if (dim < 0) dim += x.dim();
    
    auto original_shape = x.sizes().vec();
    int64_t prefix = 1;
    for (int i = 0; i < dim; ++i) prefix *= original_shape[i];
    int64_t dim_size = original_shape[dim];
    int64_t suffix = 1;
    for (int i = dim + 1; i < x.dim(); ++i) suffix *= original_shape[i];

    auto output = torch::empty_like(x);

    if (suffix == 1 && (dim_size % 4 == 0)) {
        int block_size = 512; 
        int num_blocks = prefix;
        int M_v4 = dim_size / 4;
        cumprod_kernel_2d_dim1_vectorized<<<num_blocks, block_size>>>(
            (const float4*)x.data_ptr<float>(), (float4*)output.data_ptr<float>(), (int)prefix, M_v4);
    } else {
        return torch::cumprod(x, dim);
    }

    return output;
}
"""

cumprod_module = load_inline(
    name="cumprod_module",
    cpp_sources=cumprod_cpp_source,
    functions=["cumprod_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return cumprod_module.cumprod_hip(x, self.dim)

