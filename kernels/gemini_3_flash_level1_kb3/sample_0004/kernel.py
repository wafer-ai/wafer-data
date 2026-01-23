
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblas/hipblas.h>

static hipblasHandle_t handle = nullptr;

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    if (handle == nullptr) {
        hipblasCreate(&handle);
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    // Use hipblasSgemm with column-major representation.
    // Result of A (MxK) * B (KxN) is C (MxN).
    // In column-major:
    // A_cm is KxM, B_cm is NxK, C_cm is NxM.
    // C_cm = B_cm * A_cm.
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

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    functions=["matmul_hip"],
    extra_ldflags=["-lhipblas"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.device.type == 'cuda':
            return self.matmul_cuda.matmul_hip(A, B)
        else:
            return torch.matmul(A, B)

M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
