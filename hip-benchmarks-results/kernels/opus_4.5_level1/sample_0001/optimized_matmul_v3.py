import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

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
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    
    // Ensure contiguous
    A = A.contiguous();
    B = B.contiguous();
    
    init_rocblas();
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // rocBLAS uses column-major, so we compute C^T = B^T * A^T
    // Which gives us C = A * B in row-major
    rocblas_sgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_none,
                  N,   // number of rows of C^T
                  M,   // number of cols of C^T  
                  K,   // inner dimension
                  &alpha,
                  B.data_ptr<float>(),  // B^T in column major
                  N,   // leading dimension of B
                  A.data_ptr<float>(),  // A^T in column major
                  K,   // leading dimension of A
                  &beta,
                  C.data_ptr<float>(),  // C^T in column major
                  N);  // leading dimension of C
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B);
"""

matmul_module = load_inline(
    name="matmul_hip_rocblas",
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
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
