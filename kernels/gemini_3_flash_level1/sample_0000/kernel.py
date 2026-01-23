
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure the CXX environment variable is set to hipcc for ROCm
os.environ["CXX"] = "hipcc"

gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblas/hipblas.h>
#include <ATen/hip/HIPContext.h>

// Reuse a global hipBLAS handle to avoid creation overhead in each forward call.
static hipblasHandle_t global_handle = nullptr;

torch::Tensor gemm_hipblas(torch::Tensor A, torch::Tensor B) {
    if (global_handle == nullptr) {
        hipblasCreate(&global_handle);
    }

    // A is M x K, B is K x N.
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    auto C = torch::empty({m, n}, A.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    // Set the stream for the hipBLAS handle to the current PyTorch HIP stream.
    hipblasSetStream(global_handle, at::hip::getCurrentHIPStream());

    // Call hipblasSgemm for matrix multiplication C = A * B.
    // hipBLAS assumes column-major ordering. 
    // We treat row-major matrices A and B as column-major matrices B^T and A^T.
    // Thus, C^T = B^T * A^T where B^T is N x K and A^T is K x M.
    hipblasSgemm(global_handle, 
                 HIPBLAS_OP_N, 
                 HIPBLAS_OP_N, 
                 n, m, k, 
                 &alpha, 
                 B.data_ptr<float>(), n, 
                 A.data_ptr<float>(), k, 
                 &beta, 
                 C.data_ptr<float>(), n);

    return C;
}
"""

# Load the custom HIP extension
gemm_module = load_inline(
    name="gemm_module",
    cpp_sources=gemm_cpp_source,
    functions=["gemm_hipblas"],
    extra_ldflags=["-lhipblas"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a square matrix multiplication using hipBLAS.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm_hip = gemm_module.gemm_hipblas

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.
        """
        # Ensure inputs are on GPU
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        return self.gemm_hip(A, B)

# Constants
N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
