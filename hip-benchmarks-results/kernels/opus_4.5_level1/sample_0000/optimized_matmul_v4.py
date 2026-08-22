import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use rocBLAS with better handle management and stream synchronization
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>
#include <ATen/hip/HIPContext.h>

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have same size");
    
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());
    
    // Get rocBLAS handle from PyTorch's HIP context
    rocblas_handle handle = at::cuda::getCurrentCUDABlasHandle();
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // For row-major: C = A * B
    // In col-major this is: C^T = B^T * A^T
    // rocBLAS computes C = alpha * op(A) * op(B) + beta * C in column-major
    // We want C_row = A_row * B_row
    // Which in col-major terms: we pass B first, then A
    rocblas_sgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_none,
                  N, N, N,
                  &alpha,
                  B.data_ptr<float>(), N,
                  A.data_ptr<float>(), N,
                  &beta,
                  C.data_ptr<float>(), N);
    
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
    extra_ldflags=["-lrocblas"],
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
