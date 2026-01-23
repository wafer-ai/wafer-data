import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblas/hipblas.h>

// Use hipBLAS for optimized matrix multiplication
torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // hipBLAS uses column-major, but PyTorch uses row-major
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    // So we compute C^T = B^T * A^T which gives us C in row-major layout
    hipblasSgemm(handle,
                 HIPBLAS_OP_N,  // No transpose for B (becomes B^T in col-major view)
                 HIPBLAS_OP_N,  // No transpose for A (becomes A^T in col-major view)
                 N,             // Number of rows of C^T (= cols of C)
                 M,             // Number of cols of C^T (= rows of C)
                 K,             // Shared dimension
                 &alpha,
                 B.data_ptr<float>(),  // B^T in col-major
                 N,                     // Leading dimension of B
                 A.data_ptr<float>(),  // A^T in col-major
                 K,                     // Leading dimension of A
                 &beta,
                 C.data_ptr<float>(),  // C^T in col-major = C in row-major
                 N);                    // Leading dimension of C
    
    hipblasDestroy(handle);
    
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
    extra_ldflags=["-lhipblas"]
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
