
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

hipblas_source = """
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <torch/extension.h>
#include <mutex>

static hipblasHandle_t global_handle = nullptr;
static std::mutex handle_mutex;

void init_handle() {
    std::lock_guard<std::mutex> lock(handle_mutex);
    if (global_handle == nullptr) {
        hipblasCreate(&global_handle);
    }
}

torch::Tensor batched_gemm_hipblas(torch::Tensor A, torch::Tensor B) {
    if (global_handle == nullptr) {
        init_handle();
    }
    
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    auto C = torch::empty({batch_size, m, n}, A.options());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Column-major SGEMM
    // We treat row-major A(m, k) as column-major A'(k, m)
    // We treat row-major B(k, n) as column-major B'(n, k)
    // We treat row-major C(m, n) as column-major C'(n, m)
    // Formula: C' = B' * A'
    
    hipblasSgemmStridedBatched(
        global_handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        n, m, k,
        &alpha,
        B.data_ptr<float>(), n, k * n,
        A.data_ptr<float>(), k, m * k,
        &beta,
        C.data_ptr<float>(), n, m * n,
        batch_size
    );
    
    return C;
}
"""

batched_gemm_lib = load_inline(
    name="batched_gemm_hipblas_v3",
    cpp_sources="torch::Tensor batched_gemm_hipblas(torch::Tensor A, torch::Tensor B);",
    cuda_sources=hipblas_source,
    functions=["batched_gemm_hipblas"],
    extra_ldflags=["-lhipblas"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_gemm = batched_gemm_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_gemm.batched_gemm_hipblas(A, B)
