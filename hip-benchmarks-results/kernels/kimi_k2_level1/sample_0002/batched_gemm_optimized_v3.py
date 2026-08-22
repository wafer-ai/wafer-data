import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

kernel_code = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n) {
    
    // Get batch index
    int batch_idx = blockIdx.z;
    
    // Calculate position in the matrix
    int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    
    if (batch_idx >= batch_size || row >= m || col >= n) {
        return;
    }
    
    // Compute batch offsets
    const float* A_batch = A + batch_idx * m * k;
    const float* B_batch = B + batch_idx * k * n;
    float* C_batch = C + batch_idx * m * n;
    
    // Compute dot product
    float sum = 0.0f;
    for (int tile_k = 0; tile_k < k; tile_k++) {
        sum += A_batch[row * k + tile_k] * B_batch[tile_k * n + col];
    }
    
    // Write result
    C_batch[row * n + col] = sum;
}

torch::Tensor batched_gemm_hip(torch::Tensor A, torch::Tensor B) {
    // Check inputs
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Expected 3D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match for matrix multiplication");
    
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    // Allocate output tensor
    auto C = torch::zeros({batch_size, m, n}, A.options());
    
    // Calculate grid and block dimensions
    dim3 threads_per_block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 num_blocks(
        (n + threads_per_block.x - 1) / threads_per_block.x,
        (m + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );
    
    // Launch kernel
    batched_gemm_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        k,
        n
    );
    
    return C;
}
"""

# Compile the kernel
batched_gemm = load_inline(
    name="batched_gemm",
    cpp_sources=kernel_code,
    functions=["batched_gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized batched matrix multiplication using custom HIP kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_gemm = batched_gemm
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous and on CUDA
        A = A.cuda().contiguous()
        B = B.cuda().contiguous()
        
        return self.batched_gemm.batched_gemm_hip(A, B)

def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    
    A = torch.randn(batch_size, m, k, dtype=torch.float32, device='cuda')
    B = torch.randn(batch_size, k, n, dtype=torch.float32, device='cuda')
    return [A, B]

def get_init_inputs():
    return []
