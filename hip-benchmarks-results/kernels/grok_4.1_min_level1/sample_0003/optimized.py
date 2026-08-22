import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matvec_cpp = """
#include <hip/hip_runtime.h>

__global__ void matvec_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* a_row = A + (size_t)row * K;
    float sum = 0.0f;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const int vec_len = 4;
    const int tile_size = block_size * vec_len;
    for (int seg = 0; seg < K; seg += tile_size) {
        int idx_base = seg + tid * vec_len;
        if (idx_base + vec_len <= K) {
            float4 va = *(float4*)(a_row + idx_base);
            float4 vb = *(float4*)(B + idx_base);
            sum += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
        } else if (idx_base < K) {
            #pragma unroll
            for (int v = 0; v < vec_len; ++v) {
                int idx = idx_base + v;
                if (idx < K) {
                    sum += a_row[idx] * B[idx];
                }
            }
        }
    }
    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        C[row] = sdata[0];
    }
}

torch::Tensor matvec_hip(torch::Tensor a, torch::Tensor b) {
    int M = a.size(0);
    int K = b.size(0);
    auto out = torch::empty({M, 1}, a.options());
    if (M == 0) return out;
    const int block_size = 1024;
    dim3 block(block_size);
    dim3 grid(M);
    size_t shared_size = (size_t)block_size * sizeof(float);
    matvec_kernel<<<grid, block, shared_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, K);
    return out;
}
"""

matvec = load_inline(
    name="matvec",
    cpp_sources=matvec_cpp,
    functions=["matvec_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matvec = matvec

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec.matvec_hip(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, 1).cuda()
    return [A, B]

def get_init_inputs():
    return []
