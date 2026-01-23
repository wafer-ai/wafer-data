
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblas/hipblas.h>

static hipblasHandle_t handle = nullptr;

torch::Tensor gemm_hip(torch::Tensor A, torch::Tensor B) {
    if (handle == nullptr) {
        hipblasCreate(&handle);
    }

    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    hipblasSgemm(handle,
                 HIPBLAS_OP_N, HIPBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B.data_ptr<float>(), N,
                 A.data_ptr<float>(), K,
                 &beta,
                 C.data_ptr<float>(), N);

    return C;
}
"""

gemm_module = load_inline(
    name="gemm_module",
    cpp_sources=gemm_cpp_source,
    functions=["gemm_hip"],
    extra_ldflags=["-lhipblas"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm_module = gemm_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        return self.gemm_module.gemm_hip(A, B)
