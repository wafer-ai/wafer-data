import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use rocBLAS with optimized settings and GemmEx for algorithm selection
bmm_hip_source = """
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match");
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    auto C = torch::empty({batch_size, M, N}, A.options());
    
    // Get PyTorch's hipBLAS handle
    hipblasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    long long strideA = (long long)M * K;
    long long strideB = (long long)K * N;
    long long strideC = (long long)M * N;
    
    // Use GemmEx with specific algorithm for potentially better performance
    hipblasStatus_t status = hipblasGemmStridedBatchedEx(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B.data_ptr<float>(), HIPBLAS_R_32F, N, strideB,
        A.data_ptr<float>(), HIPBLAS_R_32F, K, strideA,
        &beta,
        C.data_ptr<float>(), HIPBLAS_R_32F, N, strideC,
        batch_size,
        HIPBLAS_R_32F,
        HIPBLAS_GEMM_DEFAULT
    );
    
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, "hipBLAS GemmEx failed: ", (int)status);
    
    return C;
}
"""

bmm_cpp_source = """
torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

bmm_module = load_inline(
    name="batched_matmul_hipblas_v5",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_hip_source,
    functions=["batched_matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lhipblas"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_op = bmm_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_op.batched_matmul_hip(A, B)


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
