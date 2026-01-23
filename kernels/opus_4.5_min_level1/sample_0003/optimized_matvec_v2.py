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
#define ITEMS_PER_THREAD 8

// Optimized kernel: each block handles one row
// Uses vectorized float4 loads and efficient warp reduction
__global__ void matvec_kernel_v2(const float* __restrict__ A, 
                                  const float* __restrict__ B, 
                                  float* __restrict__ C,
                                  int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const float* A_row = A + row * K;
    
    float sum = 0.0f;
    
    // Each thread processes multiple elements with float4 vectorization
    // Total elements per iteration = BLOCK_SIZE * 4
    int tid = threadIdx.x;
    int num_float4 = K / 4;
    
    const float4* A_row_f4 = reinterpret_cast<const float4*>(A_row);
    const float4* B_f4 = reinterpret_cast<const float4*>(B);
    
    for (int i = tid; i < num_float4; i += blockDim.x) {
        float4 a_val = A_row_f4[i];
        float4 b_val = B_f4[i];
        sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
    }
    
    // Handle remaining elements (K % 4)
    int remaining_start = num_float4 * 4;
    for (int i = remaining_start + tid; i < K; i += blockDim.x) {
        sum += A_row[i] * B[i];
    }
    
    // Warp-level reduction
    __shared__ float shared_data[BLOCK_SIZE / WARP_SIZE];
    
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Reduce within warp using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, WARP_SIZE);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_data[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared_data[lane] : 0.0f;
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
    
    matvec_kernel_v2<<<grid, block>>>(
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
