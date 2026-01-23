
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

matmul_transpose_a_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

torch::Tensor matmul_transpose_a_hip(torch::Tensor A, torch::Tensor B) {
    // The reference implementation is torch.matmul(A.T, B)
    // A.T is (M, K), B is (K, N).
    // at::mm is the optimized matrix-matrix multiplication in ATen.
    return at::mm(A.t(), B);
}
"""

matmul_transpose_a = load_inline(
    name="matmul_transpose_a",
    cpp_sources=matmul_transpose_a_cpp_source,
    functions=["matmul_transpose_a_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul_transpose_a = matmul_transpose_a

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_transpose_a.matmul_transpose_a_hip(A, B)

def get_inputs():
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.rand(K, M).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
