import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matvec_hip_source = """
#include <hip/hip_runtime.h>

#define THREADS_PER_ROW 16

__global__ void matvec_mul_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    int sub_idx = threadIdx.x;  // 0 to THREADS_PER_ROW-1
    
    // Distribute work across THREADS_PER_ROW threads per row
    int chunk = (K + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    int start = sub_idx * chunk;
    int end = min(start + chunk, K);
    
    float sum = 0.0f;
    for (int j = start; j < end; j++) {
        sum += A[row * K + j] * B[j];
    }
    
    // Write partial sum to shared memory
    __shared__ float partial_sums[THREADS_PER_ROW];
    partial_sums[sub_idx] = sum;
    __syncthreads();
    
    // Parallel reduction
    for (int s = THREADS_PER_ROW / 2; s > 0; s /= 2) {
        if (sub_idx < s) {
            partial_sums[sub_idx] += partial_sums[sub_idx + s];
        }
        __syncthreads();
    }
    
    if (sub_idx == 0) {
        C[row] = partial_sums[0];
    }
}

torch::Tensor matvec_mul_hip(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto C = torch::zeros({M, 1}, A.options());
    
    dim3 block(THREADS_PER_ROW);
    dim3 grid(M);
    
    matvec_mul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);
    
    return C;
}
"""

matvec_mul = load_inline(
    name="matvec_mul",
    cpp_sources=matvec_hip_source,
    functions=["matvec_mul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B) using custom HIP kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec_mul = matvec_mul
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized HIP kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return self.matvec_mul.matvec_mul_hip(A, B)

# Keeping the same configuration for compatibility
M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed