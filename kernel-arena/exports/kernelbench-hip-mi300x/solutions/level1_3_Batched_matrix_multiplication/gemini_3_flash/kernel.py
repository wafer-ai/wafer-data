
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Optimized Batched Matrix Multiplication using hipBLAS directly from C++
bmm_cpp_source = """
#include <torch/extension.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

// Handle function to manage hipBLAS handle
hipblasHandle_t get_hipblas_handle() {
    static hipblasHandle_t handle = nullptr;
    if (handle == nullptr) {
        hipblasCreate(&handle);
    }
    return handle;
}

torch::Tensor bmm_hip(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::empty({batch_size, M, N}, A.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    // Use hipblasSgemmStridedBatched for batched matrix multiplication.
    // Tensors are row-major in PyTorch (batch, rows, cols).
    // hipBLAS is column-major.
    // A(batch, M, K) row-major is A^T(batch, K, M) column-major.
    // B(batch, K, N) row-major is B^T(batch, N, K) column-major.
    // C = A * B in row-major is C^T = B^T * A^T in column-major.
    // So we compute B^T * A^T = C^T
    // B^T: opN, N rows, K cols
    // A^T: opN, K rows, M cols
    // Result C^T: N rows, M cols
    
    hipblasSgemmStridedBatched(
        get_hipblas_handle(),
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        N, M, K,
        &alpha,
        B.data_ptr<float>(), N, K * N,
        A.data_ptr<float>(), K, M * K,
        &beta,
        C.data_ptr<float>(), N, M * N,
        batch_size
    );

    return C;
}
"""

bmm_module = load_inline(
    name="bmm_module",
    cpp_sources=bmm_cpp_source,
    functions=["bmm_hip"],
    extra_ldflags=["-lhipblas"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_module = bmm_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Simple wrapper to handle the calling
        return self.bmm_module.bmm_hip(A, B)
