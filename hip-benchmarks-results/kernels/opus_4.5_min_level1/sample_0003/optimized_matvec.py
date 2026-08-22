import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matvec_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Each block computes one row of the output
// Uses warp-level reduction for efficiency
__global__ void matvec_kernel(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C,
                               int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const float* A_row = A + row * K;
    
    // Each thread accumulates partial sum
    float sum = 0.0f;
    
    // Vectorized loads using float4 for better memory bandwidth
    int k = threadIdx.x * 4;
    int stride = blockDim.x * 4;
    
    // Process 4 elements at a time
    for (; k + 3 < K; k += stride) {
        float4 a_val = *reinterpret_cast<const float4*>(A_row + k);
        float4 b_val = *reinterpret_cast<const float4*>(B + k);
        sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
    }
    
    // Handle remaining elements
    for (; k < K; k += blockDim.x) {
        if (k < K) {
            sum += A_row[k] * B[k];
        }
    }
    
    // Warp-level reduction using shuffle
    __shared__ float shared_data[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Reduce within warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, WARP_SIZE);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_data[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? shared_data[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset, WARP_SIZE);
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
