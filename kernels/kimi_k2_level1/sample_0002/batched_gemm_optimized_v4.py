import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Using rocBLAS wrapper for optimal performance
kernel_code = """
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <torch/extension.h>

// Custom batched gemm kernel using rocBLAS
void batched_gemm_rocblas(
    const float* A, const float* B, float* C,
    int batch_size, int m, int k, int n,
    hipblasHandle_t handle) {
    
    // Set C to zero
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Leading dimensions
    int64_t lda = k;
    int64_t ldb = n;
    int64_t ldc = n;
    
    // rocBLAS expects row-major format, which is what PyTorch uses for 3D tensors
    hipblasStatus_t status = hipblasSgemmStridedBatched(
        handle,
        HIPBLAS_OP_N,  // No transpose
        HIPBLAS_OP_N,  // No transpose
        n, m, k,       // Dimensions (note: swapped m and n because CUBLAS uses column-major internally)
        &alpha,
        B, lda, k * n, // B matrix (note: PyTorch's bmm expects A[batch, m, k] * B[batch, k, n])
        A, lda, m * k, // A matrix
        &beta,
        C, ldc, m * n, // C matrix
        batch_size
    );
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
    
    // Get hipBLAS handle
    static hipblasHandle_t handle = nullptr;
    if (handle == nullptr) {
        hipblasCreate(&handle);
    }
    
    // Call rocBLAS batched gemm
    batched_gemm_rocblas(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, m, k, n,
        handle
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
    extra_ldflags=['-lhipblas'],
)

class ModelNew(nn.Module):
    """
    Optimized batched matrix multiplication using rocBLAS HIP backend.
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
