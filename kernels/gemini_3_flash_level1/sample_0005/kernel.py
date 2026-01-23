
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

tall_skinny_matmul_source = """
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <torch/extension.h>

// Handle for hipBLAS
hipblasHandle_t handle = nullptr;

torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B) {
    if (handle == nullptr) {
        hipblasCreate(&handle);
    }

    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    // hipBLAS uses column-major order.
    // Our matrices are row-major.
    // A (M x K) row-major is A^T (K x M) column-major.
    // B (K x N) row-major is B^T (N x K) column-major.
    // C (M x N) row-major is C^T (N x M) column-major.
    // We want C = A * B.
    // In column-major: C^T = (A * B)^T = B^T * A^T.
    // So we call hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N).

    hipblasSgemm(handle, 
                 HIPBLAS_OP_N, HIPBLAS_OP_N, 
                 N, M, K, 
                 &alpha, 
                 B.data_ptr<float>(), N, 
                 A.data_ptr<float>(), K, 
                 &beta, 
                 C.data_ptr<float>(), N);

    return C;
}
"""

tall_skinny_matmul = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=tall_skinny_matmul_source,
    functions=["tall_skinny_matmul_hip"],
    libraries=["hipblas"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tall_skinny_matmul = tall_skinny_matmul

    def forward(self, A, B):
        return self.tall_skinny_matmul.tall_skinny_matmul_hip(A, B)

