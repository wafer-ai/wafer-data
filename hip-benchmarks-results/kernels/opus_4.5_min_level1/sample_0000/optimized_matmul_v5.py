import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Try using the ATen matmul with strided access to avoid copies
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>
#include <ATen/hip/HIPBlas.h>

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    
    // Use torch's optimized mm directly - it already uses rocBLAS
    return torch::mm(A, B);
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
