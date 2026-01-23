import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>

// Global handle - created once
static rocblas_handle g_handle = nullptr;
static bool g_handle_initialized = false;

void ensure_handle() {
    if (!g_handle_initialized) {
        rocblas_create_handle(&g_handle);
        g_handle_initialized = true;
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
    
    ensure_handle();
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // rocBLAS uses column-major, but PyTorch uses row-major
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    rocblas_sgemm(g_handle,
                  rocblas_operation_none,  // B treated as B^T
                  rocblas_operation_none,  // A treated as A^T
                  N,                        // Rows of C^T (= cols of C)
                  M,                        // Cols of C^T (= rows of C)
                  K,                        // Shared dimension
                  &alpha,
                  B.data_ptr<float>(),     // B data
                  N,                        // Leading dimension of B
                  A.data_ptr<float>(),     // A data
                  K,                        // Leading dimension of A
                  &beta,
                  C.data_ptr<float>(),     // C data
                  N);                       // Leading dimension of C
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B);
"""

matmul_module = load_inline(
    name="matmul_hip",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_hip_source,
    functions=["matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lrocblas"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_hip(A, B)


def get_inputs():
    N = 2048 * 2
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
