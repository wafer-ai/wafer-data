
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void __launch_bounds__(256)
gemm_float4_kernel(const float4* __restrict__ A, const float4* __restrict__ B_transposed, float* __restrict__ C,
                   int M, int N, int K) {
    // This is a simplified version. A real high-performance GEMM
    // would be much more complex.
    // Assuming K and N are multiples of 4.
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        int K4 = K / 4;
        for (int k = 0; k < K4; ++k) {
            float4 a4 = A[row * K4 + k];
            float4 b4 = B_transposed[col * K4 + k];
            sum += a4.x * b4.x + a4.y * b4.y + a4.z * b4.z + a4.w * b4.w;
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor gemm_hip(torch::Tensor A, torch::Tensor B) {
    const int batch_size = A.size(0);
    const int M_dim = A.size(1);
    const int K = A.size(2);
    const int N = B.size(1);
    const int M = batch_size * M_dim;

    auto C = torch::empty({batch_size, M_dim, N}, A.options());
    auto B_transposed = B.t().contiguous();

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    gemm_float4_kernel<<<grid, block>>>(
        (const float4*)A.data_ptr<float>(),
        (const float4*)B_transposed.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K);

    return C;
}
"""

gemm_module = load_inline(
    name="gemm_module_float4",
    cpp_sources=gemm_cpp_source,
    functions=["gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm_module = gemm_module

    def forward(self, A, B):
        return self.gemm_module.gemm_hip(A, B)
