# Trajectory: sample_0003

## Input
**level:** level1
**problem_id:** 3
**problem_path:** /root/Wafer/research/KernelBench/KernelBench/level1/4_Matrix_vector_multiplication_.py
**ref_arch_src:** import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(Model, self).

... (truncated, 772 chars total)
**name:** 4_Matrix_vector_multiplication_
**user_prompt:** Optimize the HIP kernel for 4_Matrix_vector_multiplication_
**_sample_id:** sample_0003

## Score
- **judge_score:** 0.700
- **judge_score_raw:** 7.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
Agent correctly ran wafer evaluate kernelbench with --defensive flag and achieved correctness. The optimization uses hipBLAS SGEMM with N=1 instead of GEMV, which is a reasonable approach for ROCm. However, the speedup is modest at 1.05x (2.280ms vs 2.349ms reference). The code is well-structured and uses proper row-major to column-major reinterpretation. While functional and correct, the performance gain is minimal compared to more aggressive custom kernel optimizations that could achieve higher speedups.
```

## Conversation

### System Prompt
```
You are a GPU AMD HIP kernel optimization expert. Your task is to optimize GPU kernels for performance.

CRITICAL: You do NOT have a local GPU. You MUST use `wafer evaluate kernelbench` to test kernels on remote GPU hardware. Do not try to run kernels locally with Python - it will fail.

## Kernel Format

Kernels use PyTorch's load_inline to compile HIP code. The format is:
- `custom_kernel(inputs)` - Your kernel function that takes a tuple of tensors
- The HIP code is embedded as a string and compiled with load_inline

## Tools Available
- bash(command): Run shell commands including wafer CLI
- write(file_path, content): Write optimized kernel

## MANDATORY Workflow - Follow These Steps EXACTLY

1. Understand the naive kernel (reference implementation provided in prompt)
2. Identify bottlenecks (memory access patterns, lack of tiling, etc.)
3. Use the write tool to create your optimized kernel file with `custom_kernel(inputs)` function
4. IMMEDIATELY run wafer evaluate with a 300-seco

... (truncated, 2563 chars total)
```

### User
You write custom HIP/ROCm kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom HIP/ROCm kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Here's an example to show you the syntax of inline embedding custom HIP/ROCm kernels in PyTorch:

Example:

Input architecture:

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []


Optimized with HIP/ROCm kernels:

import os

import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

elementwise_add_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_hip(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(

... (truncated, 4656 chars total)

### Assistant (Turn 1)
**Tool Call: `write`**
```json
{
  "path": "optimized_matvec.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\n# Ensure we build with hipcc on ROCm\nos.environ.setdefault(\"CXX\", \"hipcc\")\n\n# hipBLAS-based GEMV. This often beats GEMM for (M x K) @ (K x 1)\n# and avoids launching a full GEMM kernel.\nsource = r'''\n#include <torch/extension.h>\n#include <ATen/ATen.h>\n#include <ATen/hip/HIPContext.h>\n\n#include <hip/hip_runtime.h>\n#include <hipblas/hipblas.h>\n\nstatic inline hipblasHandle_t get_handle() {\n    // One handle per process; hipBLAS handles are thread-safe for different streams\n    static hipblasHandle_t handle = nullptr;\n    static std::once_flag flag;\n    std::call_once(flag, [](){\n        hipblasCreate(&handle);\n        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);\n    });\n    return handle;\n}\n\ntorch::Tensor matvec_hipblas(torch::Tensor A, torch::Tensor B) {\n    TORCH_CHECK(A.is_cuda(), \"A must be CUDA/ROCm tensor\");\n    TORCH_CHECK(B.is_cuda(), \"B must be CUDA/ROCm tensor\");\n    TORCH_CHECK(A.scalar_type() == torch::kFloat32, \"A must be float32\");\n    TORCH_CHECK(B.scalar_type() == torch::kFloat32, \"B must be float32\");\n    TORCH_CHECK(A.dim() == 2, \"A must be 2D\");\n    TORCH_CHECK(B.dim() == 2, \"B must be 2D (K,1)\");\n\n    // Force contiguous to guarantee expected layout.\n    if (!A.is_contiguous()) A = A.contiguous();\n    if (!B.is_contiguous()) B = B.contiguous();\n\n    const int64_t M = A.size(0);\n    const int64_t K = A.size(1);\n    TORCH_CHECK(B.size(0) == K && B.size(1) == 1, \"B must have shape (K,1)\");\n\n    auto out = torch::empty({M, 1}, A.options());\n\n    // We interpret row-major A (M x K) as column-major A' (K x M) where A' = A^T.\n    // Then compute out = (A')^T * x  => out = A * x.\n    const int m = (int)K;   // rows of A' in col-major\n    const int n = (int)M;   // cols of A'\n    const int lda = (int)K; // leading dimension for col-major A'\n\n    const float alpha = 1.0f;\n    const float beta  = 0.0f;\n\n    hipblasHandle_t handle = get_handle();\n    auto stream = at::hip::getDefaultHIPStream();\n    hipblasSetStream(handle, stream);\n\n    const float* A_ptr = (const float*)A.data_ptr<float>();\n    const float* x_ptr = (const float*)B.data_ptr<float>();\n    float* y_ptr = (float*)out.data_ptr<float>();\n\n    // x is length K, incx=1. y is length M, incy=1.\n    hipblasStatus_t st = hipblasSgemv(handle,\n                                     HIPBLAS_OP_T,\n                                     m, n,\n                                     &alpha,\n                                     A_ptr, lda,\n                                     x_ptr, 1,\n                                     &beta,\n                                     y_ptr, 1);\n    TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, \"hipblasSgemv failed\");\n\n    return out;\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def(\"matvec_hipblas\", &matvec_hipblas, \"matvec via hipBLAS SGEMV (FP32)\");\n}\n'''\n\next = load_inline(\n    name='matvec_hipblas_ext',\n    cpp_sources='',\n    cuda_sources=source,\n    functions=None,\n    extra_cuda_cflags=['-O3'],\n    extra_cflags=['-O3'],\n    with_cuda=True,\n    verbose=False,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:\n        # kernelbench inputs are CPU by default; move to GPU if needed\n        if not A.is_cuda:\n            A = A.cuda()\n        if not B.is_cuda:\n            B = B.cuda()\n        return ext.matvec_hipblas(A, B)\n\n\n# Keep the same shapes as the reference\nM = 256 * 8\nK = 131072 * 8\n\ndef get_inputs():\n    A = torch.rand(M, K)\n    B = torch.rand(K, 1)\n    return [A, B]\n\n\ndef get_init_inputs():\n    return []\n"
}
```

### Tool Result
```
Successfully wrote 3674 bytes to optimized_matvec.py
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_matvec.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/4_Matrix_vector_multiplication_.py --benchmark --defensive",
  "timeout": 600
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_122205
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
Successfully preprocessed all matching files.
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 3.796ms, Ref: 2.417ms, Speedup: 0.64x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_122205/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.64x
============================================================

```

### Assistant (Turn 3)
**Tool Call: `write`**
```json
{
  "path": "optimized_matvec.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ.setdefault(\"CXX\", \"hipcc\")\n\n# Use hipBLAS SGEMV. Key for performance: assume inputs are already on GPU and contiguous\n# (KernelBench harness provides CUDA tensors for GPU targets).\nsource = r'''\n#include <torch/extension.h>\n#include <ATen/hip/HIPContext.h>\n\n#include <hipblas/hipblas.h>\n#include <mutex>\n\nstatic inline hipblasHandle_t get_handle() {\n    static hipblasHandle_t handle = nullptr;\n    static std::once_flag flag;\n    std::call_once(flag, [](){\n        hipblasCreate(&handle);\n        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);\n    });\n    return handle;\n}\n\ntorch::Tensor matvec_hipblas(torch::Tensor A, torch::Tensor B) {\n    TORCH_CHECK(A.is_cuda() && B.is_cuda(), \"A and B must be CUDA/ROCm tensors\");\n    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, \"FP32 only\");\n    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), \"A and B must be contiguous\");\n    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, \"A 2D, B 2D\");\n\n    const int64_t M = A.size(0);\n    const int64_t K = A.size(1);\n    TORCH_CHECK(B.size(0) == K && B.size(1) == 1, \"B must be (K,1)\");\n\n    auto out = torch::empty({M, 1}, A.options());\n\n    // Interpret row-major A (M x K) as column-major A' (K x M) with A' = A^T.\n    // Compute out = (A')^T * x => A * x.\n    const int m = (int)K;\n    const int n = (int)M;\n    const int lda = (int)K;\n\n    const float alpha = 1.0f;\n    const float beta  = 0.0f;\n\n    hipblasHandle_t handle = get_handle();\n    hipblasSetStream(handle, at::hip::getDefaultHIPStream());\n\n    const float* A_ptr = (const float*)A.data_ptr<float>();\n    const float* x_ptr = (const float*)B.data_ptr<float>();\n    float* y_ptr = (float*)out.data_ptr<float>();\n\n    hipblasStatus_t st = hipblasSgemv(handle,\n                                     HIPBLAS_OP_T,\n                                     m, n,\n                                     &alpha,\n                                     A_ptr, lda,\n                                     x_ptr, 1,\n                                     &beta,\n                                     y_ptr, 1);\n    TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, \"hipblasSgemv failed\");\n\n    return out;\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def(\"matvec_hipblas\", &matvec_hipblas, \"matvec via hipBLAS SGEMV (FP32)\");\n}\n'''\n\next = load_inline(\n    name='matvec_hipblas_ext',\n    cpp_sources='',\n    cuda_sources=source,\n    functions=None,\n    extra_cuda_cflags=['-O3'],\n    extra_cflags=['-O3'],\n    with_cuda=True,\n    verbose=False,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:\n        return ext.matvec_hipblas(A, B)\n\n\nM = 256 * 8\nK = 131072 * 8\n\ndef get_inputs():\n    A = torch.rand(M, K)\n    B = torch.rand(K, 1)\n    return [A, B]\n\n\ndef get_init_inputs():\n    return []\n"
}
```

### Tool Result
```
Successfully wrote 2977 bytes to optimized_matvec.py
```

### Assistant (Turn 4)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_matvec.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/4_Matrix_vector_multiplication_.py --benchmark --defensive",
  "timeout": 600
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_122508
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
Successfully preprocessed all matching files.
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 2.402ms, Ref: 2.356ms, Speedup: 0.98x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_122508/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.98x
============================================================

```

### Assistant (Turn 5)
**Tool Call: `write`**
```json
{
  "path": "optimized_matvec.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ.setdefault(\"CXX\", \"hipcc\")\n\n# Use hipBLAS SGEMM with N=1. On some ROCm versions this can outperform SGEMV.\nsource = r'''\n#include <torch/extension.h>\n#include <ATen/hip/HIPContext.h>\n\n#include <hipblas/hipblas.h>\n#include <mutex>\n\nstatic inline hipblasHandle_t get_handle() {\n    static hipblasHandle_t handle = nullptr;\n    static std::once_flag flag;\n    std::call_once(flag, [](){\n        hipblasCreate(&handle);\n        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);\n    });\n    return handle;\n}\n\ntorch::Tensor matvec_sgemm(torch::Tensor A, torch::Tensor B) {\n    TORCH_CHECK(A.is_cuda() && B.is_cuda(), \"A and B must be CUDA/ROCm tensors\");\n    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, \"FP32 only\");\n    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), \"A and B must be contiguous\");\n    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, \"A 2D, B 2D\");\n\n    const int64_t M = A.size(0);\n    const int64_t K = A.size(1);\n    TORCH_CHECK(B.size(0) == K && B.size(1) == 1, \"B must be (K,1)\");\n\n    auto C = torch::empty({M, 1}, A.options());\n\n    // Column-major interpretation trick:\n    // - Treat row-major A (M x K) as column-major A' (K x M) where A' = A^T.\n    // - B is (K x 1); for N=1, row/col-major are identical.\n    // Compute: C = op(A') * B with op(A') = transpose, giving (M x K) * (K x 1).\n\n    const int m = (int)M;\n    const int n = 1;\n    const int k = (int)K;\n\n    const int lda = (int)K; // A' is (k x m) col-major\n    const int ldb = (int)K; // B is (k x 1)\n    const int ldc = (int)M; // C is (m x 1)\n\n    const float alpha = 1.0f;\n    const float beta  = 0.0f;\n\n    hipblasHandle_t handle = get_handle();\n    hipblasSetStream(handle, at::hip::getDefaultHIPStream());\n\n    const float* A_ptr = (const float*)A.data_ptr<float>();\n    const float* B_ptr = (const float*)B.data_ptr<float>();\n    float* C_ptr = (float*)C.data_ptr<float>();\n\n    hipblasStatus_t st = hipblasSgemm(handle,\n                                     HIPBLAS_OP_T, HIPBLAS_OP_N,\n                                     m, n, k,\n                                     &alpha,\n                                     A_ptr, lda,\n                                     B_ptr, ldb,\n                                     &beta,\n                                     C_ptr, ldc);\n    TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, \"hipblasSgemm failed\");\n\n    return C;\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def(\"matvec_sgemm\", &matvec_sgemm, \"matvec via hipBLAS SGEMM (FP32, N=1)\");\n}\n'''\n\next = load_inline(\n    name='matvec_sgemm_ext',\n    cpp_sources='',\n    cuda_sources=source,\n    functions=None,\n    extra_cuda_cflags=['-O3'],\n    extra_cflags=['-O3'],\n    with_cuda=True,\n    verbose=False,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:\n        return ext.matvec_sgemm(A, B)\n\n\nM = 256 * 8\nK = 131072 * 8\n\ndef get_inputs():\n    A = torch.rand(M, K)\n    B = torch.rand(K, 1)\n    return [A, B]\n\n\ndef get_init_inputs():\n    return []\n"
}
```

### Tool Result
```
Successfully wrote 3193 bytes to optimized_matvec.py
```

### Assistant (Turn 6)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_matvec.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/4_Matrix_vector_multiplication_.py --benchmark --defensive",
  "timeout": 600
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_122711
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
Successfully preprocessed all matching files.
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 2.280ms, Ref: 2.349ms, Speedup: 1.03x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_122711/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 1.05x
============================================================

```

### Assistant (Turn 7)
Implemented a custom ROCm extension that replaces `torch.matmul(A, B)` (with `B` shaped `(K,1)`) by calling **hipBLAS SGEMM with N=1** (often faster than GEMV on ROCm for tall/skinny cases). The kernel uses a **row-major → column-major reinterpretation trick** so we can invoke hipBLAS without explicitly transposing `A`.

### Optimized implementation (`ModelNew`) written to: `optimized_matvec.py`
- Replaces matmul with `hipblasSgemm` (FP32).
- Assumes inputs are already CUDA/ROCm tensors and contiguous (as KernelBench provides on GPU targets).

### Testing (MANDATORY) command executed
```bash
wafer evaluate kernelbench --target local-mi300x --implementation optimized_matvec.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/4_Matrix_vector_multiplication_.py --benchmark --defensive
```

### Result
- Correctness: **PASS**
- Performance: **New: 2.280ms, Ref: 2.349ms**
- Speedup: **~1.05x**

DONE

### Assistant (Turn 8)
Implemented a custom ROCm extension that replaces `torch.matmul(A, B)` (with `B` shaped `(K,1)`) by calling **hipBLAS SGEMM with N=1** (often faster than GEMV on ROCm for tall/skinny cases). The kernel uses a **row-major → column-major reinterpretation trick** so we can invoke hipBLAS without explicitly transposing `A`.

### Optimized implementation (`ModelNew`) written to: `optimized_matvec.py`
- Replaces matmul with `hipblasSgemm` (FP32).
- Assumes inputs are already CUDA/ROCm tensors and contiguous (as KernelBench provides on GPU targets).

### Testing (MANDATORY) command executed
```bash
wafer evaluate kernelbench --target local-mi300x --implementation optimized_matvec.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/4_Matrix_vector_multiplication_.py --benchmark --defensive
```

### Result
- Correctness: **PASS**
- Performance: **New: 2.280ms, Ref: 2.349ms**
- Speedup: **~1.05x**

DONE

### Assistant (Turn 9)
Implemented a custom ROCm extension that replaces `torch.matmul(A, B)` (with `B` shaped `(K,1)`) by calling **hipBLAS SGEMM with N=1** (often faster than GEMV on ROCm for tall/skinny cases). The kernel uses a **row-major → column-major reinterpretation trick** so we can invoke hipBLAS without explicitly transposing `A`.

### Optimized implementation (`ModelNew`) written to: `optimized_matvec.py`
- Replaces matmul with `hipblasSgemm` (FP32).
- Assumes inputs are already CUDA/ROCm tensors and contiguous (as KernelBench provides on GPU targets).

### Testing (MANDATORY) command executed
```bash
wafer evaluate kernelbench --target local-mi300x --implementation optimized_matvec.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/4_Matrix_vector_multiplication_.py --benchmark --defensive
```

### Result
- Correctness: **PASS**
- Performance: **New: 2.280ms, Ref: 2.349ms**
- Speedup: **~1.05x**

DONE

### Assistant (Turn 10)
Implemented a custom ROCm extension that replaces `torch.matmul(A, B)` (with `B` shaped `(K,1)`) by calling **hipBLAS SGEMM with N=1** (often faster than GEMV on ROCm for tall/skinny cases). The kernel uses a **row-major → column-major reinterpretation trick** so we can invoke hipBLAS without explicitly transposing `A`.

### Optimized implementation (`ModelNew`) written to: `optimized_matvec.py`
- Replaces matmul with `hipblasSgemm` (FP32).
- Assumes inputs are already CUDA/ROCm tensors and contiguous (as KernelBench provides on GPU targets).

### Testing (MANDATORY) command executed
```bash
wafer evaluate kernelbench --target local-mi300x --implementation optimized_matvec.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/4_Matrix_vector_multiplication_.py --benchmark --defensive
```

### Result
- Correctness: **PASS**
- Performance: **New: 2.280ms, Ref: 2.349ms**
- Speedup: **~1.05x**

DONE

## Expected Answer
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
