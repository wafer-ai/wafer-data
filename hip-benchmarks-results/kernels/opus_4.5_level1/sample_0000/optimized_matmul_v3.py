import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use rocBLAS for optimal performance
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>

static rocblas_handle handle = nullptr;

void init_rocblas() {
    if (handle == nullptr) {
        rocblas_create_handle(&handle);
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    init_rocblas();
    
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // rocBLAS expects column-major, PyTorch is row-major
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    // So we compute C = B^T * A^T with swapped arguments
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
