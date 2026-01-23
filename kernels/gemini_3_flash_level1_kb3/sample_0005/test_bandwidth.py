
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

write_only_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void write_only_kernel(float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = 1.0f;
    }
}

torch::Tensor write_only_hip(int M, int N) {
    auto C = torch::empty({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    int size = M * N;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    write_only_kernel<<<num_blocks, block_size>>>(C.data_ptr<float>(), size);
    return C;
}
"""

write_only = load_inline(
    name="write_only",
    cpp_sources=write_only_cpp_source,
    functions=["write_only_hip"],
    verbose=True,
)

import time
M = 32768
N = 32768
# Warmup
for _ in range(10):
    C = write_only.write_only_hip(M, N)
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    C = write_only.write_only_hip(M, N)
torch.cuda.synchronize()
end = time.time()
print(f"Write-only time: {(end - start) / 100 * 1000:.3f} ms")
print(f"Bandwidth: {(M * N * 4) / ((end - start) / 100) / 1e12:.3f} TB/s")
