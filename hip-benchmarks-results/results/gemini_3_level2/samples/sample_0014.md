# Trajectory: sample_0014

## Input
**level:** level2
**problem_id:** 98
**problem_path:** /root/Wafer/research/KernelBench/KernelBench/level2/99_Matmul_GELU_Softmax.py
**ref_arch_src:** import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_fe

... (truncated, 669 chars total)
**name:** 99_Matmul_GELU_Softmax
**user_prompt:** Optimize the HIP kernel for 99_Matmul_GELU_Softmax
**_sample_id:** sample_0014

## Score
- **judge_score:** 0.600
- **judge_score_raw:** 6.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
The agent successfully ran wafer evaluate kernelbench multiple times to test their optimizations, meeting the critical requirement. The final implementation achieved correctness (100% pass rate) and implemented reasonable optimization techniques including operator fusion (GELU+Softmax), vectorization with float4, shared memory usage, and in-place operations. However, the measured speedup was consistently below 1.0x (0.97x in the final test), indicating the custom kernel performed slightly slower than the reference PyTorch implementation. The code quality was good with proper kernel structure and optimization attempts, but the lack of positive speedup significantly impacts the score.
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

... (truncated, 4535 chars total)

### Assistant (Turn 1)
**Tool Call: `write`**
```json
{
  "content": "\nimport os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <cmath>\n#include <torch/extension.h>\n\n__global__ void fused_bias_gelu_softmax_kernel(\n    const float* __restrict__ input,\n    const float* __restrict__ bias,\n    float* __restrict__ output,\n    int rows,\n    int cols\n) {\n    // Shared memory layout:\n    // [0 ... cols-1]: row data (values)\n    // [cols ... cols+blockDim.x-1]: reduction buffer\n    extern __shared__ float shared_mem[];\n    float* row_data = shared_mem;\n    float* sdata = &row_data[cols];\n\n    int tid = threadIdx.x;\n    int row_idx = blockIdx.x;\n\n    if (row_idx >= rows) return;\n\n    int row_offset = row_idx * cols;\n\n    // 1. Load, Add Bias, GELU, Find Max\n    float local_max = -INFINITY;\n    const float SQRT_2_INV = 0.70710678f;\n\n    for (int i = tid; i < cols; i += blockDim.x) {\n        float val = input[row_offset + i] + bias[i];\n\n        // GELU Exact: 0.5 * x * (1 + erf(x / sqrt(2)))\n        val = 0.5f * val * (1.0f + erff(val * SQRT_2_INV));\n\n        row_data[i] = val;\n        if (val > local_max) local_max = val;\n    }\n\n    // Block Reduce Max\n    sdata[tid] = local_max;\n    __syncthreads();\n\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];\n        }\n        __syncthreads();\n    }\n    float row_max_val = sdata[0];\n\n    // 2. Compute Exp and Sum\n    float local_sum = 0.0f;\n    for (int i = tid; i < cols; i += blockDim.x) {\n        float val = row_data[i];\n        val = expf(val - row_max_val);\n        row_data[i] = val;\n        local_sum += val;\n    }\n\n    // Block Reduce Sum\n    sdata[tid] = local_sum;\n    __syncthreads();\n\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] += sdata[tid + s];\n        }\n        __syncthreads();\n    }\n    float row_sum_val = sdata[0];\n\n    // 3. Normalize and Write\n    float inv_sum = 1.0f / row_sum_val;\n\n    for (int i = tid; i < cols; i += blockDim.x) {\n        output[row_offset + i] = row_data[i] * inv_sum;\n    }\n}\n\ntorch::Tensor fused_bias_gelu_softmax_hip(torch::Tensor input, torch::Tensor bias) {\n    auto rows = input.size(0);\n    auto cols = input.size(1);\n    auto output = torch::empty_like(input);\n\n    const int block_size = 1024;\n    const int grid_size = rows;\n    \n    // Shared memory: cols floats + block_size floats\n    int shared_mem_size = (cols + block_size) * sizeof(float);\n\n    fused_bias_gelu_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        bias.data_ptr<float>(),\n        output.data_ptr<float>(),\n        rows,\n        cols\n    );\n\n    return output;\n}\n\"\"\"\n\nfused_ops = load_inline(\n    name=\"fused_ops\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_bias_gelu_softmax_hip\"],\n    verbose=True,\n    extra_cflags=['-O3'],\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features)\n        self.fused_ops = fused_ops\n\n    def forward(self, x):\n        x_mm = torch.mm(x, self.linear.weight.t())\n        return self.fused_ops.fused_bias_gelu_softmax_hip(x_mm, self.linear.bias)\n\nbatch_size = 1024\nin_features = 8192\nout_features = 8192\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features).cuda()]\n\ndef get_init_inputs():\n    return [in_features, out_features]\n",
  "path": "custom_kernel.py"
}
```

### Tool Result
```
Successfully wrote 3557 bytes to custom_kernel.py
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/99_Matmul_GELU_Softmax.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_021232
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_ops -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -c /root/.cache/torch_extensions/py312_cpu/fused_ops/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_ops.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 1.285ms, Ref: 1.251ms, Speedup: 0.97x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_021232/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.97x
============================================================

```

### Assistant (Turn 3)
**Tool Call: `write`**
```json
{
  "content": "\nimport os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <cmath>\n#include <torch/extension.h>\n\n__global__ void fused_gelu_softmax_kernel(\n    const float* __restrict__ input,\n    float* __restrict__ output,\n    int rows,\n    int cols\n) {\n    // Shared memory layout:\n    // [0 ... cols-1]: row data (values)\n    // [cols ... cols+blockDim.x-1]: reduction buffer\n    extern __shared__ float shared_mem[];\n    float* row_data = shared_mem;\n    float* sdata = &row_data[cols];\n\n    int tid = threadIdx.x;\n    int row_idx = blockIdx.x;\n\n    if (row_idx >= rows) return;\n\n    int row_offset = row_idx * cols;\n    const float SQRT_2_INV = 0.70710678f;\n    float local_max = -INFINITY;\n\n    // Vectorized Handling using float4\n    // cols must be divisible by 4\n    int vec_cols = cols / 4;\n    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);\n    float4* row_data_vec = reinterpret_cast<float4*>(row_data);\n\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 in_val = input_vec[i];\n        float4 out_val;\n\n        // Process 4 elements\n        // x\n        float v = in_val.x;\n        v = 0.5f * v * (1.0f + erff(v * SQRT_2_INV));\n        out_val.x = v;\n        local_max = fmaxf(local_max, v);\n\n        // y\n        v = in_val.y;\n        v = 0.5f * v * (1.0f + erff(v * SQRT_2_INV));\n        out_val.y = v;\n        local_max = fmaxf(local_max, v);\n\n        // z\n        v = in_val.z;\n        v = 0.5f * v * (1.0f + erff(v * SQRT_2_INV));\n        out_val.z = v;\n        local_max = fmaxf(local_max, v);\n\n        // w\n        v = in_val.w;\n        v = 0.5f * v * (1.0f + erff(v * SQRT_2_INV));\n        out_val.w = v;\n        local_max = fmaxf(local_max, v);\n\n        row_data_vec[i] = out_val;\n    }\n\n    // Block Reduce Max\n    sdata[tid] = local_max;\n    __syncthreads();\n\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);\n        }\n        __syncthreads();\n    }\n    float row_max_val = sdata[0];\n\n    // Compute Exp and Sum\n    float local_sum = 0.0f;\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = row_data_vec[i];\n        \n        val.x = expf(val.x - row_max_val);\n        local_sum += val.x;\n\n        val.y = expf(val.y - row_max_val);\n        local_sum += val.y;\n\n        val.z = expf(val.z - row_max_val);\n        local_sum += val.z;\n\n        val.w = expf(val.w - row_max_val);\n        local_sum += val.w;\n\n        row_data_vec[i] = val;\n    }\n\n    // Block Reduce Sum\n    sdata[tid] = local_sum;\n    __syncthreads();\n\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] += sdata[tid + s];\n        }\n        __syncthreads();\n    }\n    float row_sum_val = sdata[0];\n\n    // Normalize and Write\n    float inv_sum = 1.0f / row_sum_val;\n    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);\n\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = row_data_vec[i];\n        val.x *= inv_sum;\n        val.y *= inv_sum;\n        val.z *= inv_sum;\n        val.w *= inv_sum;\n        output_vec[i] = val;\n    }\n}\n\ntorch::Tensor fused_gelu_softmax_hip(torch::Tensor input) {\n    auto rows = input.size(0);\n    auto cols = input.size(1);\n    auto output = torch::empty_like(input);\n\n    const int block_size = 256;\n    const int grid_size = rows;\n    \n    // Shared memory: cols floats + block_size floats\n    int shared_mem_size = (cols + block_size) * sizeof(float);\n\n    fused_gelu_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        rows,\n        cols\n    );\n\n    return output;\n}\n\"\"\"\n\nfused_ops = load_inline(\n    name=\"fused_ops_v2\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_gelu_softmax_hip\"],\n    verbose=True,\n    extra_cflags=['-O3'],\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features)\n        self.fused_ops = fused_ops\n\n    def forward(self, x):\n        # Use torch.addmm for optimized GEMM + Bias\n        # self.linear.bias is 1D, addmm broadcasts it\n        x = torch.addmm(self.linear.bias, x, self.linear.weight.t())\n        \n        # Then Fused GELU + Softmax\n        return self.fused_ops.fused_gelu_softmax_hip(x)\n\nbatch_size = 1024\nin_features = 8192\nout_features = 8192\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features).cuda()]\n\ndef get_init_inputs():\n    return [in_features, out_features]\n",
  "path": "custom_kernel.py"
}
```

### Tool Result
```
Successfully wrote 4731 bytes to custom_kernel.py
```

### Assistant (Turn 4)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/99_Matmul_GELU_Softmax.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_023054
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_ops_v2 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -c /root/.cache/torch_extensions/py312_cpu/fused_ops_v2/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_ops_v2.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 1.274ms, Ref: 1.248ms, Speedup: 0.98x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_023054/results.json

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
  "content": "\nimport os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <cmath>\n#include <torch/extension.h>\n\n__global__ void fused_bias_gelu_softmax_kernel(\n    const float* __restrict__ input,\n    const float* __restrict__ bias,\n    float* __restrict__ output,\n    int rows,\n    int cols\n) {\n    extern __shared__ float shared_mem[];\n    float* row_data = shared_mem;\n    float* sdata = &row_data[cols];\n\n    int tid = threadIdx.x;\n    int row_idx = blockIdx.x;\n\n    if (row_idx >= rows) return;\n\n    int row_offset = row_idx * cols;\n    float local_max = -INFINITY;\n\n    const float GELU_C1 = 0.7978845608f;\n    const float GELU_C2 = 0.044715f;\n\n    int vec_cols = cols / 4;\n    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);\n    const float4* bias_vec = reinterpret_cast<const float4*>(bias);\n    float4* row_data_vec = reinterpret_cast<float4*>(row_data);\n\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 in_val = input_vec[i];\n        float4 b_val = bias_vec[i];\n        float4 out_val;\n\n        #define GELU_APPROX(x) ({ \\\n            float _x = (x); \\\n            float _inner = GELU_C1 * (_x + GELU_C2 * _x * _x * _x); \\\n            0.5f * _x * (1.0f + tanhf(_inner)); \\\n        })\n\n        out_val.x = GELU_APPROX(in_val.x + b_val.x);\n        local_max = fmaxf(local_max, out_val.x);\n\n        out_val.y = GELU_APPROX(in_val.y + b_val.y);\n        local_max = fmaxf(local_max, out_val.y);\n\n        out_val.z = GELU_APPROX(in_val.z + b_val.z);\n        local_max = fmaxf(local_max, out_val.z);\n\n        out_val.w = GELU_APPROX(in_val.w + b_val.w);\n        local_max = fmaxf(local_max, out_val.w);\n        \n        #undef GELU_APPROX\n\n        row_data_vec[i] = out_val;\n    }\n\n    sdata[tid] = local_max;\n    __syncthreads();\n\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);\n        }\n        __syncthreads();\n    }\n    float row_max_val = sdata[0];\n\n    float local_sum = 0.0f;\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = row_data_vec[i];\n        \n        val.x = expf(val.x - row_max_val);\n        local_sum += val.x;\n\n        val.y = expf(val.y - row_max_val);\n        local_sum += val.y;\n\n        val.z = expf(val.z - row_max_val);\n        local_sum += val.z;\n\n        val.w = expf(val.w - row_max_val);\n        local_sum += val.w;\n\n        row_data_vec[i] = val;\n    }\n\n    sdata[tid] = local_sum;\n    __syncthreads();\n\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] += sdata[tid + s];\n        }\n        __syncthreads();\n    }\n    float row_sum_val = sdata[0];\n\n    float inv_sum = 1.0f / row_sum_val;\n    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);\n\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = row_data_vec[i];\n        val.x *= inv_sum;\n        val.y *= inv_sum;\n        val.z *= inv_sum;\n        val.w *= inv_sum;\n        output_vec[i] = val;\n    }\n}\n\ntorch::Tensor fused_bias_gelu_softmax_hip(torch::Tensor input, torch::Tensor bias) {\n    auto rows = input.size(0);\n    auto cols = input.size(1);\n    auto output = torch::empty_like(input);\n\n    const int block_size = 256;\n    const int grid_size = rows;\n    int shared_mem_size = (cols + block_size) * sizeof(float);\n\n    fused_bias_gelu_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        bias.data_ptr<float>(),\n        output.data_ptr<float>(),\n        rows,\n        cols\n    );\n\n    return output;\n}\n\"\"\"\n\nfused_ops = load_inline(\n    name=\"fused_ops_v3\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_bias_gelu_softmax_hip\"],\n    verbose=True,\n    extra_cflags=['-O3'],\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features)\n        self.fused_ops = fused_ops\n\n    def forward(self, x):\n        x = torch.mm(x, self.linear.weight.t())\n        return self.fused_ops.fused_bias_gelu_softmax_hip(x, self.linear.bias)\n\nbatch_size = 1024\nin_features = 8192\nout_features = 8192\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features).cuda()]\n\ndef get_init_inputs():\n    return [in_features, out_features]\n",
  "path": "custom_kernel.py"
}
```

### Tool Result
```
Successfully wrote 4465 bytes to custom_kernel.py
```

### Assistant (Turn 6)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/99_Matmul_GELU_Softmax.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_023939
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_ops_v3 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -c /root/.cache/torch_extensions/py312_cpu/fused_ops_v3/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_ops_v3.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 1.294ms, Ref: 1.257ms, Speedup: 0.97x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_023939/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.97x
============================================================

```

### Assistant (Turn 7)
**Tool Call: `write`**
```json
{
  "path": "custom_kernel.py",
  "content": "\nimport os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <cmath>\n#include <torch/extension.h>\n\n__global__ void __launch_bounds__(1024) fused_gelu_softmax_kernel(\n    const float* __restrict__ input,\n    float* __restrict__ output,\n    int rows,\n    int cols\n) {\n    extern __shared__ float shared_mem[];\n    float* row_data = shared_mem;\n    float* sdata = &row_data[cols];\n\n    int tid = threadIdx.x;\n    int row_idx = blockIdx.x;\n\n    if (row_idx >= rows) return;\n\n    int row_offset = row_idx * cols;\n    float local_max = -INFINITY;\n\n    const float GELU_C1 = 0.7978845608f;\n    const float GELU_C2 = 0.044715f;\n\n    int vec_cols = cols / 4;\n    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);\n    float4* row_data_vec = reinterpret_cast<float4*>(row_data);\n\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 in_val = input_vec[i];\n        float4 out_val;\n\n        #define GELU_APPROX(x) ({ \\\n            float _x = (x); \\\n            float _inner = GELU_C1 * (_x + GELU_C2 * _x * _x * _x); \\\n            0.5f * _x * (1.0f + tanhf(_inner)); \\\n        })\n\n        out_val.x = GELU_APPROX(in_val.x);\n        local_max = fmaxf(local_max, out_val.x);\n\n        out_val.y = GELU_APPROX(in_val.y);\n        local_max = fmaxf(local_max, out_val.y);\n\n        out_val.z = GELU_APPROX(in_val.z);\n        local_max = fmaxf(local_max, out_val.z);\n\n        out_val.w = GELU_APPROX(in_val.w);\n        local_max = fmaxf(local_max, out_val.w);\n        \n        #undef GELU_APPROX\n\n        row_data_vec[i] = out_val;\n    }\n\n    sdata[tid] = local_max;\n    __syncthreads();\n\n    // Reduction Max\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);\n        }\n        __syncthreads();\n    }\n    float row_max_val = sdata[0];\n\n    float local_sum = 0.0f;\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = row_data_vec[i];\n        \n        val.x = expf(val.x - row_max_val);\n        local_sum += val.x;\n\n        val.y = expf(val.y - row_max_val);\n        local_sum += val.y;\n\n        val.z = expf(val.z - row_max_val);\n        local_sum += val.z;\n\n        val.w = expf(val.w - row_max_val);\n        local_sum += val.w;\n\n        row_data_vec[i] = val;\n    }\n\n    sdata[tid] = local_sum;\n    __syncthreads();\n\n    // Reduction Sum\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] += sdata[tid + s];\n        }\n        __syncthreads();\n    }\n    float row_sum_val = sdata[0];\n\n    float inv_sum = 1.0f / row_sum_val;\n    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);\n\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = row_data_vec[i];\n        val.x *= inv_sum;\n        val.y *= inv_sum;\n        val.z *= inv_sum;\n        val.w *= inv_sum;\n        output_vec[i] = val;\n    }\n}\n\ntorch::Tensor fused_gelu_softmax_hip(torch::Tensor input) {\n    auto rows = input.size(0);\n    auto cols = input.size(1);\n    auto output = torch::empty_like(input);\n\n    const int block_size = 1024;\n    const int grid_size = rows;\n    int shared_mem_size = (cols + block_size) * sizeof(float);\n\n    fused_gelu_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        rows,\n        cols\n    );\n\n    return output;\n}\n\"\"\"\n\nfused_ops = load_inline(\n    name=\"fused_ops_v4\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_gelu_softmax_hip\"],\n    verbose=True,\n    extra_cflags=['-O3', '-ffast-math'],\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features)\n        self.fused_ops = fused_ops\n\n    def forward(self, x):\n        # Use addmm for optimal GEMM + Bias\n        x = torch.addmm(self.linear.bias, x, self.linear.weight.t())\n        return self.fused_ops.fused_gelu_softmax_hip(x)\n\nbatch_size = 1024\nin_features = 8192\nout_features = 8192\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features).cuda()]\n\ndef get_init_inputs():\n    return [in_features, out_features]\n"
}
```

### Tool Result
```
Successfully wrote 4337 bytes to custom_kernel.py
```

### Assistant (Turn 8)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/99_Matmul_GELU_Softmax.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_025238
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_ops_v4 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -ffast-math -c /root/.cache/torch_extensions/py312_cpu/fused_ops_v4/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
/root/.cache/torch_extensions/py312_cpu/fused_ops_v4/main.cpp:23:24: warning: use of infinity via a macro is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
   23 |     float local_max = -INFINITY;
      |                        ^
/root/.cache/torch_extensions/py312_cpu/fused_ops_v4/main.cpp:23:24: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
   23 |     float local_max = -INFINITY;
      |                        ^~~~~~~~
/usr/include/math.h:91:21: note: expanded from macro 'INFINITY'
   91 | #  define INFINIT

... (truncated, 2990 chars total)
```

### Assistant (Turn 9)
**Tool Call: `write`**
```json
{
  "path": "custom_kernel.py",
  "content": "\nimport os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <cmath>\n#include <cfloat>\n#include <torch/extension.h>\n\n__global__ void __launch_bounds__(1024) fused_gelu_softmax_kernel(\n    const float* __restrict__ input,\n    float* __restrict__ output,\n    int rows,\n    int cols\n) {\n    __shared__ float sdata[1024];\n\n    int tid = threadIdx.x;\n    int row_idx = blockIdx.x;\n\n    if (row_idx >= rows) return;\n\n    int row_offset = row_idx * cols;\n    float local_max = -FLT_MAX;\n\n    const float GELU_C1 = 0.7978845608f;\n    const float GELU_C2 = 0.044715f;\n\n    int vec_cols = cols / 4;\n    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);\n    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);\n\n    // Pass 1: GELU + Max\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 in_val = input_vec[i];\n        float4 out_val;\n\n        #define GELU_APPROX(x) ({ \\\n            float _x = (x); \\\n            float _inner = GELU_C1 * (_x + GELU_C2 * _x * _x * _x); \\\n            0.5f * _x * (1.0f + tanhf(_inner)); \\\n        })\n\n        out_val.x = GELU_APPROX(in_val.x);\n        local_max = fmaxf(local_max, out_val.x);\n\n        out_val.y = GELU_APPROX(in_val.y);\n        local_max = fmaxf(local_max, out_val.y);\n\n        out_val.z = GELU_APPROX(in_val.z);\n        local_max = fmaxf(local_max, out_val.z);\n\n        out_val.w = GELU_APPROX(in_val.w);\n        local_max = fmaxf(local_max, out_val.w);\n        #undef GELU_APPROX\n        \n        output_vec[i] = out_val;\n    }\n    \n    sdata[tid] = local_max;\n    __syncthreads();\n\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);\n        __syncthreads();\n    }\n    float row_max = sdata[0];\n    __syncthreads(); \n\n    // Pass 2: Exp + Sum\n    float local_sum = 0.0f;\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = output_vec[i];\n        \n        val.x = expf(val.x - row_max);\n        local_sum += val.x;\n\n        val.y = expf(val.y - row_max);\n        local_sum += val.y;\n\n        val.z = expf(val.z - row_max);\n        local_sum += val.z;\n\n        val.w = expf(val.w - row_max);\n        local_sum += val.w;\n\n        output_vec[i] = val;\n    }\n\n    sdata[tid] = local_sum;\n    __syncthreads();\n\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) sdata[tid] += sdata[tid + s];\n        __syncthreads();\n    }\n    float row_sum = sdata[0];\n    float inv_sum = 1.0f / row_sum;\n    __syncthreads();\n\n    // Pass 3: Normalize\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = output_vec[i];\n        val.x *= inv_sum;\n        val.y *= inv_sum;\n        val.z *= inv_sum;\n        val.w *= inv_sum;\n        output_vec[i] = val;\n    }\n}\n\ntorch::Tensor fused_gelu_softmax_hip(torch::Tensor input) {\n    auto rows = input.size(0);\n    auto cols = input.size(1);\n    auto output = torch::empty_like(input);\n\n    const int block_size = 1024;\n    const int grid_size = rows;\n    \n    fused_gelu_softmax_kernel<<<grid_size, block_size>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        rows,\n        cols\n    );\n\n    return output;\n}\n\"\"\"\n\nfused_ops = load_inline(\n    name=\"fused_ops_v5\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_gelu_softmax_hip\"],\n    verbose=True,\n    extra_cflags=['-O3', '-ffast-math'],\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features)\n        self.fused_ops = fused_ops\n\n    def forward(self, x):\n        x = torch.addmm(self.linear.bias, x, self.linear.weight.t())\n        return self.fused_ops.fused_gelu_softmax_hip(x)\n\nbatch_size = 1024\nin_features = 8192\nout_features = 8192\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features).cuda()]\n\ndef get_init_inputs():\n    return [in_features, out_features]\n"
}
```

### Tool Result
```
Successfully wrote 4087 bytes to custom_kernel.py
```

### Assistant (Turn 10)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/99_Matmul_GELU_Softmax.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_030620
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_ops_v5 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -ffast-math -c /root/.cache/torch_extensions/py312_cpu/fused_ops_v5/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_ops_v5.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 1.325ms, Ref: 1.253ms, Speedup: 0.95x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_030620/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.95x
============================================================

```

### Assistant (Turn 11)
**Tool Call: `write`**
```json
{
  "content": "\nimport os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <cmath>\n#include <torch/extension.h>\n\n__global__ void __launch_bounds__(1024) fused_gelu_softmax_kernel(\n    const float* __restrict__ input,\n    float* __restrict__ output,\n    int rows,\n    int cols\n) {\n    extern __shared__ float shared_mem[];\n    float* row_data = shared_mem;\n    float* sdata = &row_data[cols];\n\n    int tid = threadIdx.x;\n    int row_idx = blockIdx.x;\n\n    if (row_idx >= rows) return;\n\n    int row_offset = row_idx * cols;\n    float local_max = -INFINITY;\n\n    const float GELU_C1 = 0.7978845608f;\n    const float GELU_C2 = 0.044715f;\n\n    int vec_cols = cols / 4;\n    \n    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);\n    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);\n    float4* row_data_vec = reinterpret_cast<float4*>(row_data);\n\n    // Pass 1: Load Global -> Shared. Compute GELU. Update Max.\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 in_val = input_vec[i];\n        float4 out_val;\n\n        #define GELU_APPROX(x) ({ \\\n            float _x = (x); \\\n            float _inner = GELU_C1 * (_x + GELU_C2 * _x * _x * _x); \\\n            0.5f * _x * (1.0f + tanhf(_inner)); \\\n        })\n\n        out_val.x = GELU_APPROX(in_val.x);\n        local_max = fmaxf(local_max, out_val.x);\n\n        out_val.y = GELU_APPROX(in_val.y);\n        local_max = fmaxf(local_max, out_val.y);\n\n        out_val.z = GELU_APPROX(in_val.z);\n        local_max = fmaxf(local_max, out_val.z);\n\n        out_val.w = GELU_APPROX(in_val.w);\n        local_max = fmaxf(local_max, out_val.w);\n        #undef GELU_APPROX\n        \n        row_data_vec[i] = out_val;\n    }\n    \n    sdata[tid] = local_max;\n    __syncthreads();\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);\n        __syncthreads();\n    }\n    float row_max_val = sdata[0];\n\n    // Pass 2: Shared -> Exp -> Shared. Update Sum.\n    float local_sum = 0.0f;\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = row_data_vec[i];\n        \n        val.x = expf(val.x - row_max_val);\n        local_sum += val.x;\n        val.y = expf(val.y - row_max_val);\n        local_sum += val.y;\n        val.z = expf(val.z - row_max_val);\n        local_sum += val.z;\n        val.w = expf(val.w - row_max_val);\n        local_sum += val.w;\n\n        row_data_vec[i] = val;\n    }\n\n    sdata[tid] = local_sum;\n    __syncthreads();\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) sdata[tid] += sdata[tid + s];\n        __syncthreads();\n    }\n    float row_sum_val = sdata[0];\n    float inv_sum = 1.0f / row_sum_val;\n\n    // Pass 3: Shared -> Global\n    for (int i = tid; i < vec_cols; i += blockDim.x) {\n        float4 val = row_data_vec[i];\n        val.x *= inv_sum;\n        val.y *= inv_sum;\n        val.z *= inv_sum;\n        val.w *= inv_sum;\n        output_vec[i] = val;\n    }\n}\n\nvoid fused_gelu_softmax_hip_(torch::Tensor input) {\n    auto rows = input.size(0);\n    auto cols = input.size(1);\n    float* data = input.data_ptr<float>();\n\n    const int block_size = 1024;\n    const int grid_size = rows;\n    int shared_mem_size = (cols + block_size) * sizeof(float);\n\n    fused_gelu_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(\n        data,\n        data,\n        rows,\n        cols\n    );\n}\n\"\"\"\n\nfused_ops = load_inline(\n    name=\"fused_ops_v6\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_gelu_softmax_hip_\"],\n    verbose=True,\n    extra_cflags=['-O3', '-ffast-math'],\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features)\n        self.fused_ops = fused_ops\n\n    def forward(self, x):\n        # GEMM + Bias (out-of-place from inputs, new tensor allocated)\n        x = torch.addmm(self.linear.bias, x, self.linear.weight.t())\n        \n        # In-place Fused GELU + Softmax\n        self.fused_ops.fused_gelu_softmax_hip_(x)\n        return x\n\nbatch_size = 1024\nin_features = 8192\nout_features = 8192\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features).cuda()]\n\ndef get_init_inputs():\n    return [in_features, out_features]\n",
  "path": "custom_kernel.py"
}
```

### Tool Result
```
Successfully wrote 4417 bytes to custom_kernel.py
```

### Assistant (Turn 12)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/99_Matmul_GELU_Softmax.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_031716
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_ops_v6 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -ffast-math -c /root/.cache/torch_extensions/py312_cpu/fused_ops_v6/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
/root/.cache/torch_extensions/py312_cpu/fused_ops_v6/main.cpp:23:24: warning: use of infinity via a macro is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
   23 |     float local_max = -INFINITY;
      |                        ^
/root/.cache/torch_extensions/py312_cpu/fused_ops_v6/main.cpp:23:24: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
   23 |     float local_max = -INFINITY;
      |                        ^~~~~~~~
/usr/include/math.h:91:21: note: expanded from macro 'INFINITY'
   91 | #  define INFINIT

... (truncated, 2990 chars total)
```

## Expected Answer
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
