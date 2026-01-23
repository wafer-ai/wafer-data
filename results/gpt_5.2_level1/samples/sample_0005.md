# Trajectory: sample_0005

## Input
**level:** level1
**problem_id:** 8
**problem_path:** /root/Wafer/research/KernelBench/KernelBench/level1/9_Tall_skinny_matrix_multiplication_.py
**ref_arch_src:** import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N 

... (truncated, 890 chars total)
**name:** 9_Tall_skinny_matrix_multiplication_
**user_prompt:** Optimize the HIP kernel for 9_Tall_skinny_matrix_multiplication_
**_sample_id:** sample_0005

## Score
- **judge_score:** 0.300
- **judge_score_raw:** 3.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
The agent ran wafer evaluate kernelbench as required and achieved correctness (100%), but the kernel actually slowed down performance (0.58x speedup, meaning 1.7x slower than reference). The optimization approach was technically sound with tiling and shared memory usage, but was poorly tuned for the specific problem size (32768x32 @ 32x32768). The kernel used large 64x64 tiles which likely caused poor occupancy and excessive shared memory usage for this tall-skinny matrix multiplication pattern. While the code structure was reasonable, the performance regression significantly impacts the score.
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

... (truncated, 4784 chars total)

### Assistant (Turn 1)
**Tool Call: `bash`**
```json
{
  "command": "sed -n '1,200p' /root/Wafer/research/KernelBench/KernelBench/level1/9_Tall_skinny_matrix_multiplication_.py"
}
```

### Tool Result
```
Command(s) require approval: "sed -n '1,200p' /root/Wafer/research/KernelBench/KernelBench/level1/9_Tall_skinny_matrix_multiplication_.py"
This environment only permits commands starting with: 'wafer evaluate', 'python', 'python3', 'ls', 'cat', 'head', 'tail', 'wc', 'find', 'cd', 'pwd', 'which'
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "head -n 200 /root/Wafer/research/KernelBench/KernelBench/level1/9_Tall_skinny_matrix_multiplication_.py"
}
```

### Tool Result
```
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Assistant (Turn 3)
**Tool Call: `write`**
```json
{
  "path": "optimized_9_tall_skinny_matmul.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\n# Compile with hipcc for ROCm\nos.environ.setdefault(\"CXX\", \"hipcc\")\n\n# K=32 specialized GEMM: (M x 32) @ (32 x N) -> (M x N)\n# Tuned for large M,N and small K (32).\n\nhip_source = r'''\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <hip/hip_runtime.h>\n\n#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x \" must be a CUDA/HIP tensor\")\n#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x \" must be contiguous\")\n#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x \" must be float32\")\n\n// 64x64 tile, 16x16 threads, each thread computes a 4x4 micro-tile.\n// Shared memory uses padding to mitigate bank conflicts.\n__global__ void gemm_k32_tiled_64x64_kernel(\n    const float* __restrict__ A, // [M, 32]\n    const float* __restrict__ B, // [32, N]\n    float* __restrict__ C,       // [M, N]\n    int M, int N)\n{\n    // block tile origin\n    const int block_row = (int)blockIdx.y * 64;\n    const int block_col = (int)blockIdx.x * 64;\n\n    const int tx = (int)threadIdx.x; // 0..15\n    const int ty = (int)threadIdx.y; // 0..15\n    const int tid = ty * 16 + tx;\n\n    __shared__ float As[64][33];   // [BM][K+1]\n    __shared__ float Bs[32][65];   // [K][BN+1]\n\n    // Load A tile: 64x32 = 2048 floats = 512 float4 loads.\n    // Each thread loads 2 float4.\n    #pragma unroll\n    for (int i = 0; i < 2; ++i) {\n        int vec_id = tid * 2 + i;          // 0..511\n        int elem = vec_id * 4;            // 0..2044 step 4\n        int r = elem >> 5;                // /32, 0..63\n        int k = elem & 31;                // %32, multiple of 4\n        int gr = block_row + r;\n        if (gr < M) {\n            const float4 v = *reinterpret_cast<const float4*>(A + gr * 32 + k);\n            As[r][k + 0] = v.x;\n            As[r][k + 1] = v.y;\n            As[r][k + 2] = v.z;\n            As[r][k + 3] = v.w;\n        } else {\n            // out of bounds rows\n            As[r][k + 0] = 0.0f;\n            As[r][k + 1] = 0.0f;\n            As[r][k + 2] = 0.0f;\n            As[r][k + 3] = 0.0f;\n        }\n    }\n\n    // Load B tile: 32x64 = 2048 floats = 512 float4 loads.\n    #pragma unroll\n    for (int i = 0; i < 2; ++i) {\n        int vec_id = tid * 2 + i;          // 0..511\n        int elem = vec_id * 4;\n        int k = elem >> 6;                 // /64, 0..31\n        int c = elem & 63;                 // %64, multiple of 4\n        int gc = block_col + c;\n        if (gc + 3 < N) {\n            const float4 v = *reinterpret_cast<const float4*>(B + k * N + gc);\n            Bs[k][c + 0] = v.x;\n            Bs[k][c + 1] = v.y;\n            Bs[k][c + 2] = v.z;\n            Bs[k][c + 3] = v.w;\n        } else {\n            // tail (rare for benchmark sizes)\n            float tmp0 = 0.f, tmp1 = 0.f, tmp2 = 0.f, tmp3 = 0.f;\n            if (gc + 0 < N) tmp0 = B[k * N + (gc + 0)];\n            if (gc + 1 < N) tmp1 = B[k * N + (gc + 1)];\n            if (gc + 2 < N) tmp2 = B[k * N + (gc + 2)];\n            if (gc + 3 < N) tmp3 = B[k * N + (gc + 3)];\n            Bs[k][c + 0] = tmp0;\n            Bs[k][c + 1] = tmp1;\n            Bs[k][c + 2] = tmp2;\n            Bs[k][c + 3] = tmp3;\n        }\n    }\n\n    __syncthreads();\n\n    // Each thread computes C for rows (ty*4..ty*4+3) and cols (tx*4..tx*4+3)\n    const int row0 = ty * 4;\n    const int col0 = tx * 4;\n\n    float acc00 = 0.f, acc01 = 0.f, acc02 = 0.f, acc03 = 0.f;\n    float acc10 = 0.f, acc11 = 0.f, acc12 = 0.f, acc13 = 0.f;\n    float acc20 = 0.f, acc21 = 0.f, acc22 = 0.f, acc23 = 0.f;\n    float acc30 = 0.f, acc31 = 0.f, acc32 = 0.f, acc33 = 0.f;\n\n    #pragma unroll\n    for (int k = 0; k < 32; ++k) {\n        float a0 = As[row0 + 0][k];\n        float a1 = As[row0 + 1][k];\n        float a2 = As[row0 + 2][k];\n        float a3 = As[row0 + 3][k];\n\n        float b0 = Bs[k][col0 + 0];\n        float b1 = Bs[k][col0 + 1];\n        float b2 = Bs[k][col0 + 2];\n        float b3 = Bs[k][col0 + 3];\n\n        acc00 = fmaf(a0, b0, acc00);\n        acc01 = fmaf(a0, b1, acc01);\n        acc02 = fmaf(a0, b2, acc02);\n        acc03 = fmaf(a0, b3, acc03);\n\n        acc10 = fmaf(a1, b0, acc10);\n        acc11 = fmaf(a1, b1, acc11);\n        acc12 = fmaf(a1, b2, acc12);\n        acc13 = fmaf(a1, b3, acc13);\n\n        acc20 = fmaf(a2, b0, acc20);\n        acc21 = fmaf(a2, b1, acc21);\n        acc22 = fmaf(a2, b2, acc22);\n        acc23 = fmaf(a2, b3, acc23);\n\n        acc30 = fmaf(a3, b0, acc30);\n        acc31 = fmaf(a3, b1, acc31);\n        acc32 = fmaf(a3, b2, acc32);\n        acc33 = fmaf(a3, b3, acc33);\n    }\n\n    // Store results\n    const int gr = block_row + row0;\n    const int gc = block_col + col0;\n\n    if (gr + 3 < M && gc + 3 < N) {\n        // vectorized stores\n        *reinterpret_cast<float4*>(C + (gr + 0) * N + gc) = make_float4(acc00, acc01, acc02, acc03);\n        *reinterpret_cast<float4*>(C + (gr + 1) * N + gc) = make_float4(acc10, acc11, acc12, acc13);\n        *reinterpret_cast<float4*>(C + (gr + 2) * N + gc) = make_float4(acc20, acc21, acc22, acc23);\n        *reinterpret_cast<float4*>(C + (gr + 3) * N + gc) = make_float4(acc30, acc31, acc32, acc33);\n    } else {\n        // tails\n        if (gr + 0 < M) {\n            if (gc + 0 < N) C[(gr + 0) * N + (gc + 0)] = acc00;\n            if (gc + 1 < N) C[(gr + 0) * N + (gc + 1)] = acc01;\n            if (gc + 2 < N) C[(gr + 0) * N + (gc + 2)] = acc02;\n            if (gc + 3 < N) C[(gr + 0) * N + (gc + 3)] = acc03;\n        }\n        if (gr + 1 < M) {\n            if (gc + 0 < N) C[(gr + 1) * N + (gc + 0)] = acc10;\n            if (gc + 1 < N) C[(gr + 1) * N + (gc + 1)] = acc11;\n            if (gc + 2 < N) C[(gr + 1) * N + (gc + 2)] = acc12;\n            if (gc + 3 < N) C[(gr + 1) * N + (gc + 3)] = acc13;\n        }\n        if (gr + 2 < M) {\n            if (gc + 0 < N) C[(gr + 2) * N + (gc + 0)] = acc20;\n            if (gc + 1 < N) C[(gr + 2) * N + (gc + 1)] = acc21;\n            if (gc + 2 < N) C[(gr + 2) * N + (gc + 2)] = acc22;\n            if (gc + 3 < N) C[(gr + 2) * N + (gc + 3)] = acc23;\n        }\n        if (gr + 3 < M) {\n            if (gc + 0 < N) C[(gr + 3) * N + (gc + 0)] = acc30;\n            if (gc + 1 < N) C[(gr + 3) * N + (gc + 1)] = acc31;\n            if (gc + 2 < N) C[(gr + 3) * N + (gc + 2)] = acc32;\n            if (gc + 3 < N) C[(gr + 3) * N + (gc + 3)] = acc33;\n        }\n    }\n}\n\ntorch::Tensor matmul_k32_hip(torch::Tensor A, torch::Tensor B) {\n    CHECK_CUDA(A);\n    CHECK_CUDA(B);\n    CHECK_FLOAT(A);\n    CHECK_FLOAT(B);\n\n    // Expect 2D\n    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, \"A and B must be 2D\");\n    int64_t M = A.size(0);\n    int64_t K = A.size(1);\n    int64_t K2 = B.size(0);\n    int64_t N = B.size(1);\n    TORCH_CHECK(K == K2, \"inner dimensions mismatch\");\n\n    // Specialize for K=32 and contiguous\n    if (K != 32 || !A.is_contiguous() || !B.is_contiguous()) {\n        return at::matmul(A, B);\n    }\n\n    auto C = torch::empty({M, N}, A.options());\n\n    const dim3 block(16, 16, 1);\n    const dim3 grid((unsigned int)((N + 63) / 64), (unsigned int)((M + 63) / 64), 1);\n\n    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();\n    hipLaunchKernelGGL(\n        gemm_k32_tiled_64x64_kernel,\n        grid,\n        block,\n        0,\n        stream,\n        (const float*)A.data_ptr<float>(),\n        (const float*)B.data_ptr<float>(),\n        (float*)C.data_ptr<float>(),\n        (int)M,\n        (int)N\n    );\n    return C;\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def(\"matmul_k32_hip\", &matmul_k32_hip, \"K=32 matmul (HIP)\");\n}\n'''\n\n# Build extension (cached by name)\nmatmul_k32_ext = load_inline(\n    name=\"matmul_k32_ext\",\n    cpp_sources=hip_source,\n    functions=None,\n    extra_cuda_cflags=[\"-O3\"],\n    extra_cflags=[\"-O3\"],\n    with_cuda=True,\n    verbose=False,\n)\n\n\nclass ModelNew(nn.Module):\n    \"\"\"Optimized Model using a custom HIP kernel specialized for K=32.\"\"\"\n\n    def __init__(self):\n        super().__init__()\n        self.ext = matmul_k32_ext\n\n    def forward(self, A, B):\n        # Fallback for CPU tensors\n        if not A.is_cuda or not B.is_cuda:\n            return torch.matmul(A, B)\n        return self.ext.matmul_k32_hip(A, B)\n"
}
```

### Tool Result
```
Successfully wrote 8165 bytes to optimized_9_tall_skinny_matmul.py
```

### Assistant (Turn 4)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_9_tall_skinny_matmul.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/9_Tall_skinny_matrix_multiplication_.py --benchmark --defensive",
  "timeout": 600
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_112131
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
Successfully preprocessed all matching files.
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 2.745ms, Ref: 1.582ms, Speedup: 0.58x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_112131/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.58x
============================================================

```

### Assistant (Turn 5)
**Tool Call: `bash`**
```json
{
  "command": "python3 - <<'PY'\nimport os, re, glob\npaths=[]\nfor root,dirs,files in os.walk('/opt/rocm/include'):\n    for f in files:\n        if f.endswith(('.h','.hpp')):\n            p=os.path.join(root,f)\n            try:\n                with open(p,'r',errors='ignore') as fh:\n                    txt=fh.read()\n                if 'mfma_f32' in txt and '__builtin_amdgcn_mfma' in txt:\n                    paths.append(p)\n            except Exception:\n                pass\nprint('found',len(paths))\nfor p in paths[:20]:\n    print(p)\nPY"
}
```

### Tool Result
```
found 3
/opt/rocm/include/rocwmma/internal/mfma_impl.hpp
/opt/rocm/include/ck/utility/amd_xdlops.hpp
/opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp

```

### Assistant (Turn 6)
**Tool Call: `bash`**
```json
{
  "command": "python3 - <<'PY'\nimport re\np='/opt/rocm/include/rocwmma/internal/mfma_impl.hpp'\nwith open(p,'r',errors='ignore') as f:\n    lines=f.readlines()\nfor i,l in enumerate(lines):\n    if 'mfma_f32' in l and '__builtin_amdgcn_mfma' in l:\n        for j in range(max(0,i-2), min(len(lines), i+3)):\n            print(f'{j+1}: {lines[j].rstrip()}')\nPY"
}
```

### Tool Result
```
252:                 DRegsT result;
253:                 to_native_vector(result)
254:                     = {__builtin_amdgcn_mfma_f32_16x16x16f16(to_native_vector(regsA),
255:                                                              to_native_vector(regsB),
256:                                                              to_native_vector(regsC),
287:                 DRegsT result;
288:                 to_native_vector(result)
289:                     = {__builtin_amdgcn_mfma_f32_16x16x32_f16(to_native_vector(regsA),
290:                                                               to_native_vector(regsB),
291:                                                               to_native_vector(regsC),
353:                 DRegsT result;
354:                 to_native_vector(result)
355:                     = {__builtin_amdgcn_mfma_f32_32x32x8f16(to_native_vector(regsA),
356:                                                             to_native_vector(regsB),
357:                                                             to_native_vector(regsC),
388:                 DRegsT result;
389:                 to_native_vector(result)
390:                     = {__builtin_amdgcn_mfma_f32_32x32x16_f16(to_native_vector(regsA),
391:                                                               to_native_vector(regsB),
392:                                                               to_native_vector(regsC),
473: 
474:                 DRegsT result;
475:                 to_native_vect

... (truncated, 9902 chars total)
```

### Assistant (Turn 7)
**Tool Call: `bash`**
```json
{
  "command": "python3 - <<'PY'\nimport re\np='/opt/rocm/include/rocblas/rocblas-types.h'\nwith open(p,'r',errors='ignore') as f:\n    txt=f.read()\n# find math_mode enum\nm=re.search(r'typedef enum rocblas_math_mode\\s*\\{([^}]*)\\}', txt, re.S)\nprint('found enum', bool(m))\nif m:\n    body=m.group(1)\n    for line in body.splitlines():\n        if 'rocblas_' in line and 'math' in line:\n            print(line.strip())\nPY"
}
```

### Tool Result
```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/opt/rocm/include/rocblas/rocblas-types.h'


Error: Command exited with code 1
```

### Assistant (Turn 8)
**Tool Call: `bash`**
```json
{
  "command": "python3 - <<'PY'\nimport os\nfor root,dirs,files in os.walk('/opt/rocm/include'):\n    for f in files:\n        if f.startswith('rocblas') and f.endswith('.h'):\n            if 'types' in f:\n                print(os.path.join(root,f))\n                raise SystemExit\nprint('done')\nPY"
}
```

### Tool Result
```
/opt/rocm/include/rocblas/internal/rocblas-complex-types.h

```

### Assistant (Turn 9)
**Tool Call: `bash`**
```json
{
  "command": "python3 - <<'PY'\nimport re\np='/opt/rocm/include/rocblas/rocblas.h'\nwith open(p,'r',errors='ignore') as f:\n    txt=f.read()\nprint('size',len(txt))\n# search math_mode\nfor pat in ['rocblas_math_mode','xf32','xdl']:\n    print(pat, txt.find(pat))\n# print snippet around xf32\nidx=txt.find('xf32')\nprint(txt[idx-200:idx+300])\nPY"
}
```

### Tool Result
```
size 1748
rocblas_math_mode -1
xf32 -1
xdl -1


```

### Assistant (Turn 10)
**Tool Call: `bash`**
```json
{
  "command": "cat /opt/rocm/include/rocblas/rocblas.h"
}
```

### Tool Result
```
/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

/*!\file
 * \brief rocblas.h includes other *.h and exposes a common interface
 */

#ifndef ROCBLAS_H
#define ROCBLAS_H

/* library headers */
#include "internal/rocblas-auxiliary.h"
#in

... (truncated, 1748 chars total)
```

## Expected Answer
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
