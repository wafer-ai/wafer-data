import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>
#include <c10/hip/HIPStream.h>

// Persistent handle for performance
static rocblas_handle handle = nullptr;
static bool handle_initialized = false;

rocblas_handle get_handle() {
    if (!handle_initialized) {
        rocblas_create_handle(&handle);
        handle_initialized = true;
    }
    return handle;
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    // Ensure contiguous tensors
    A = A.contiguous();
    B = B.contiguous();
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    TORCH_CHECK(B.size(0) == K, "Matrix dimensions mismatch");
    
    auto C = torch::empty({M, N}, A.options());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    rocblas_handle h = get_handle();
    
    // Get current HIP stream
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    rocblas_set_stream(h, stream);
    
    // rocBLAS uses column-major, so we compute C^T = B^T * A^T
    // Which gives us C = A * B in row-major format
    rocblas_sgemm(h,
                  rocblas_operation_none,
                  rocblas_operation_none,
                  N, M, K,
                  &alpha,
                  B.data_ptr<float>(), N,
                  A.data_ptr<float>(), K,
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
    extra_cuda_cflags=["-O3", "-ffast-math"],
    extra_ldflags=["-lrocblas"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module
        # Warm up the handle
        dummy_a = torch.zeros(1, 1, device='cuda')
        dummy_b = torch.zeros(1, 1, device='cuda')
        self.matmul.matmul_hip(dummy_a, dummy_b)
    
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
