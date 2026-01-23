import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

bmm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblas/hipblas.h>
#include <ATen/hip/HIPContext.h>

torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match");
    
    // Get hipBLAS handle from PyTorch's context - reuses existing handle
    hipblasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);
    
    auto C = torch::empty({batch_size, M, N}, A.options());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // For row-major matrices A (M x K), B (K x N), C (M x N):
    // C = A * B becomes C^T = B^T * A^T in column-major
    //
    // In hipBLAS (column-major):
    // - A row-major (M x K) looks like A^T column-major (K x M)
    // - B row-major (K x N) looks like B^T column-major (N x K) 
    // - C row-major (M x N) looks like C^T column-major (N x M)
    //
    // We want: C^T (N x M) = B^T (N x K) * A^T (K x M)
    
    hipblasSgemmStridedBatched(
        handle,
        HIPBLAS_OP_N,   // B is B^T in col-major (no transpose needed)
        HIPBLAS_OP_N,   // A is A^T in col-major (no transpose needed)
        N,              // rows of op(B) and C
        M,              // cols of op(A) and C
        K,              // cols of op(B) = rows of op(A)
        &alpha,
        B.data_ptr<float>(),  // B^T col-major = B row-major
        N,                    // leading dimension of B^T = N
        static_cast<long long int>(K * N),  // stride between batches in B
        A.data_ptr<float>(),  // A^T col-major = A row-major
        K,                    // leading dimension of A^T = K
        static_cast<long long int>(M * K),  // stride between batches in A
        &beta,
        C.data_ptr<float>(),
        N,                    // leading dimension of C^T = N
        static_cast<long long int>(M * N),  // stride between batches in C
        batch_size
    );
    
    return C;
}
"""

bmm_cpp_source = """
torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

bmm_module = load_inline(
    name="bmm_hip_v5",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_hip_source,
    functions=["batched_matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
    extra_ldflags=["-lhipblas"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_op = bmm_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_op.batched_matmul_hip(A.contiguous(), B.contiguous())


def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]


def get_init_inputs():
    return []
