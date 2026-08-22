import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use rocBLAS with stream and better configuration
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>
#include <ATen/hip/HIPContext.h>

static rocblas_handle handle = nullptr;
static bool handle_initialized = false;

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    if (!handle_initialized) {
        rocblas_create_handle(&handle);
        // Set stream to the current CUDA stream
        handle_initialized = true;
    }
    
    // Get current HIP stream from PyTorch
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();
    rocblas_set_stream(handle, stream);
    
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // rocBLAS uses column-major ordering
    // For row-major C = A * B, we compute in column-major: C^T = B^T * A^T
    // which is equivalent to computing with swapped order and transposed view
    rocblas_sgemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        N,
        M,
        K,
        &alpha,
        B.data_ptr<float>(), N,
        A.data_ptr<float>(), K,
        &beta,
        C.data_ptr<float>(), N
    );
    
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
    M = 8205
    K = 2949
    N = 5921
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
