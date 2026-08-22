import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Using rocBLAS for matrix multiplication
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPBlas.h>
#include <rocblas/rocblas.h>

torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B) {
    // A: (M, K), B: (K, N), C: (M, N)
    // In this case: A: (M, K), B: (K, M), C: (M, M)
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    // Get the rocBLAS handle from PyTorch
    rocblas_handle handle = at::cuda::getCurrentCUDABlasHandle();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // rocBLAS uses column-major format, so we compute C^T = B^T * A^T
    // which is equivalent to computing C = A * B in row-major format
    rocblas_sgemm(
        handle,
        rocblas_operation_none,  // B^T (no transpose since B is row-major)
        rocblas_operation_none,  // A^T (no transpose since A is row-major)
        N, M, K,                 // dimensions
        &alpha,
        B.data_ptr<float>(), N,  // B leading dim
        A.data_ptr<float>(), K,  // A leading dim
        &beta,
        C.data_ptr<float>(), N   // C leading dim
    );
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

matmul_module = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_hip_source,
    functions=["tall_skinny_matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
    extra_ldflags=["-lrocblas"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module
    
    def forward(self, A, B):
        return self.matmul.tall_skinny_matmul_hip(A, B)


def get_inputs():
    M = 16384 * 2
    N = 16 * 2
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]


def get_init_inputs():
    return []
