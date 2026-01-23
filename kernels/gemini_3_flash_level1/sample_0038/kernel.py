
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cumsum_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float block_scan(float val, float* shared_data) {
    int tid = threadIdx.x;
    const int warpSize = 64; // MI300X wavefront size
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    int num_warps = blockDim.x / warpSize;

    // Intra-warp scan
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        float temp = __shfl_up(val, offset, warpSize);
        if (lane >= offset) val += temp;
    }

    // Store warp sum
    if (lane == warpSize - 1) {
        shared_data[wid] = val;
    }
    __syncthreads();

    // Scan warp sums
    if (wid == 0) {
        float wave_sum = (tid < num_warps) ? shared_data[tid] : 0.0f;
        for (int offset = 1; offset < warpSize; offset <<= 1) {
            float temp = __shfl_up(wave_sum, offset, warpSize);
            if (tid >= offset) wave_sum += temp;
        }
        if (tid < num_warps) {
            shared_data[tid] = wave_sum;
        }
    }
    __syncthreads();

    // Add scanned warp sums to original values
    if (wid > 0) {
        val += shared_data[wid - 1];
    }
    __syncthreads();
    
    return val;
}

__global__ void cumsum_kernel_stride1_vec4(const float4* __restrict__ input, float4* __restrict__ output, int outer_size, int scan_size_vec4) {
    int row = blockIdx.x;
    if (row >= outer_size) return;

    extern __shared__ float shared_data_scan[]; // size: blockDim.x / 64 + 1

    float carry = 0.0f;
    const long long row_offset = (long long)row * scan_size_vec4;

    for (int i = 0; i < scan_size_vec4; i += blockDim.x) {
        int tid = threadIdx.x;
        int idx = i + tid;
        
        float4 val4;
        if (idx < scan_size_vec4) {
            val4 = input[row_offset + idx];
        } else {
            val4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // Serial scan within float4
        float s1 = val4.x;
        float s2 = s1 + val4.y;
        float s3 = s2 + val4.z;
        float s4 = s3 + val4.w;

        // Block scan of the total sums (s4)
        float block_scanned_total = block_scan(s4, shared_data_scan);
        // block_scanned_total is the inclusive scan of s4 across the block
        
        float prev_block_total = block_scanned_total - s4;

        if (idx < scan_size_vec4) {
            float base = prev_block_total + carry;
            val4.x = s1 + base;
            val4.y = s2 + base;
            val4.z = s3 + base;
            val4.w = s4 + base;
            output[row_offset + idx] = val4;
        }
        
        // Update carry for next segment. 
        // All threads need the last thread's block_scanned_total.
        if (tid == blockDim.x - 1) {
            shared_data_scan[0] = block_scanned_total;
        }
        __syncthreads();
        carry += shared_data_scan[0];
        __syncthreads();
    }
}

__global__ void cumsum_kernel_stride1(const float* __restrict__ input, float* __restrict__ output, int outer_size, int scan_size) {
    int row = blockIdx.x;
    if (row >= outer_size) return;

    extern __shared__ float shared_data_scan[];

    float carry = 0.0f;
    const long long row_offset = (long long)row * scan_size;

    for (int i = 0; i < scan_size; i += blockDim.x) {
        int tid = threadIdx.x;
        int idx = i + tid;
        float val = (idx < scan_size) ? input[row_offset + idx] : 0.0f;
        
        float scanned_val = block_scan(val, shared_data_scan);
        
        if (tid == blockDim.x - 1) {
            shared_data_scan[0] = scanned_val;
        }
        __syncthreads();
        float segment_total = shared_data_scan[0];
        
        if (idx < scan_size) {
            output[row_offset + idx] = scanned_val + carry;
        }
        
        carry += segment_total;
        __syncthreads();
    }
}

__global__ void cumsum_kernel_general(const float* __restrict__ input, float* __restrict__ output, int outer_size, int scan_size, int inner_size) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)outer_size * inner_size) return;

    int outer_idx = (int)(idx / inner_size);
    int inner_idx = (int)(idx % inner_size);

    const long long base_offset = (long long)outer_idx * scan_size * inner_size + inner_idx;
    const float* in_ptr = input + base_offset;
    float* out_ptr = output + base_offset;

    double sum = 0.0;
    for (int i = 0; i < scan_size; ++i) {
        sum += (double)in_ptr[(long long)i * inner_size];
        out_ptr[(long long)i * inner_size] = (float)sum;
    }
}

torch::Tensor cumsum_hip(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes();
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= sizes[i];
    int scan_size = sizes[dim];
    int inner_size = 1;
    for (int i = dim + 1; i < input.dim(); ++i) inner_size *= sizes[i];

    auto output = torch::empty_like(input);
    
    if (inner_size == 1) {
        if (scan_size % 4 == 0) {
            int block_size = 256; 
            int num_blocks = outer_size;
            int scan_size_vec4 = scan_size / 4;
            size_t shared_mem = (block_size / 64 + 1) * sizeof(float);
            cumsum_kernel_stride1_vec4<<<num_blocks, block_size, shared_mem>>>(
                (const float4*)input.data_ptr<float>(), (float4*)output.data_ptr<float>(), outer_size, scan_size_vec4);
        } else {
            int block_size = 1024;
            int num_blocks = outer_size;
            size_t shared_mem = (block_size / 64 + 1) * sizeof(float);
            cumsum_kernel_stride1<<<num_blocks, block_size, shared_mem>>>(
                input.data_ptr<float>(), output.data_ptr<float>(), outer_size, scan_size);
        }
    } else {
        int block_size = 256;
        long long total_scans = (long long)outer_size * inner_size;
        int grid_size = (int)((total_scans + block_size - 1) / block_size);
        cumsum_kernel_general<<<grid_size, block_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), outer_size, scan_size, inner_size);
    }
    
    return output;
}
"""

cumsum_module = load_inline(
    name="cumsum_module",
    cpp_sources=cumsum_source,
    functions=["cumsum_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return cumsum_module.cumsum_hip(x, self.dim)

