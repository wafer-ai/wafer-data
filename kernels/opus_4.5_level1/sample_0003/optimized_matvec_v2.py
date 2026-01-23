import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matvec_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 512

__global__ void matvec_kernel(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, 
                               int M, int K) {
    // Each block handles one row of A
    int row = blockIdx.x;
    if (row >= M) return;
    
    const float* A_row = A + (size_t)row * K;
    
    // Each thread processes multiple elements with vectorized loads
    float sum = 0.0f;
    
    // Use float4 for coalesced memory access (16 bytes at a time)
    int vec_K = K / 4;
    const float4* A_row_vec = reinterpret_cast<const float4*>(A_row);
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    
    // Unroll loop for better instruction-level parallelism
    int i = threadIdx.x;
    int stride = blockDim.x;
    
    #pragma unroll 4
    for (; i < vec_K; i += stride) {
        float4 a_val = __builtin_nontemporal_load(A_row_vec + i);
        float4 b_val = __builtin_nontemporal_load(B_vec + i);
        sum += a_val.x * b_val.x;
        sum += a_val.y * b_val.y;
        sum += a_val.z * b_val.z;
        sum += a_val.w * b_val.w;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
    }
    
    // Write partial sums from each warp to shared memory
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
        }
        
        if (lane == 0) {
            C[row] = sum;
        }
    }
}

torch::Tensor matvec_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    
    auto C = torch::zeros({M, 1}, A.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(M);
    
    matvec_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K
    );
    
    return C;
}
"""

matvec_cpp_source = """
torch::Tensor matvec_hip(torch::Tensor A, torch::Tensor B);
"""

matvec_module = load_inline(
    name="matvec_hip",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_hip_source,
    functions=["matvec_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec = matvec_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec.matvec_hip(A, B)
