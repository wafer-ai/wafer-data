
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <limits>

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__global__ void __launch_bounds__(1024) softmax_online_kernel_vec(const float* __restrict__ input, float* __restrict__ output, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* input_row = input + row * dim;
    float* output_row = output + row * dim;

    float m = -1e38f;
    float d = 0.0f;

    // First pass: find max and sum (vectorized + unrolled)
    for (int i = threadIdx.x * 8; i < dim; i += blockDim.x * 8) {
        float4 vals1 = load_float4(input_row + i);
        float4 vals2 = load_float4(input_row + i + 4);
        
        float x_vals[8] = {vals1.x, vals1.y, vals1.z, vals1.w, vals2.x, vals2.y, vals2.z, vals2.w};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = x_vals[j];
            if (x > m) {
                d = d * expf(m - x) + 1.0f;
                m = x;
            } else {
                d = d + expf(x - m);
            }
        }
    }

    // Block-wide reduction for m and d
    for (int offset = 32; offset > 0; offset /= 2) {
        float m_remote = __shfl_xor(m, offset, 64);
        float d_remote = __shfl_xor(d, offset, 64);
        
        float m_new = max(m, m_remote);
        if (m > m_remote) {
            d = d + d_remote * expf(m_remote - m);
        } else {
            d = d_remote + d * expf(m - m_remote);
        }
        m = m_new;
    }

    __shared__ float shared_m[16]; 
    __shared__ float shared_d[16];
    
    int lane_id = threadIdx.x % 64;
    int warp_id = threadIdx.x / 64;
    int num_warps = blockDim.x / 64;

    if (lane_id == 0) {
        shared_m[warp_id] = m;
        shared_d[warp_id] = d;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        float m_acc = shared_m[0];
        float d_acc = shared_d[0];
        for (int i = 1; i < num_warps; ++i) {
            float m_val = shared_m[i];
            float d_val = shared_d[i];
            float m_new = max(m_acc, m_val);
            if (m_acc > m_val) {
                d_acc = d_acc + d_val * expf(m_val - m_acc);
            } else {
                d_acc = d_val + d_acc * expf(m_acc - m_val);
            }
            m_acc = m_new;
        }
        shared_m[0] = m_acc;
        shared_d[0] = d_acc;
    }
    __syncthreads();

    float m_final = shared_m[0];
    float d_final = shared_d[0];
    float inv_d = 1.0f / d_final;

    // Second pass: compute output (vectorized + unrolled)
    for (int i = threadIdx.x * 8; i < dim; i += blockDim.x * 8) {
        float4 vals1 = load_float4(input_row + i);
        float4 vals2 = load_float4(input_row + i + 4);
        
        float4 out_vals1, out_vals2;
        out_vals1.x = expf(vals1.x - m_final) * inv_d;
        out_vals1.y = expf(vals1.y - m_final) * inv_d;
        out_vals1.z = expf(vals1.z - m_final) * inv_d;
        out_vals1.w = expf(vals1.w - m_final) * inv_d;
        out_vals2.x = expf(vals2.x - m_final) * inv_d;
        out_vals2.y = expf(vals2.y - m_final) * inv_d;
        out_vals2.z = expf(vals2.z - m_final) * inv_d;
        out_vals2.w = expf(vals2.w - m_final) * inv_d;
        
        store_float4(output_row + i, out_vals1);
        store_float4(output_row + i + 4, out_vals2);
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 1024;
    const int num_blocks = batch_size;

    softmax_online_kernel_vec<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}
"""

softmax_lib = load_inline(
    name="softmax_hip",
    cpp_sources=softmax_hip_source,
    functions=["softmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_lib = softmax_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_lib.softmax_hip(x)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
