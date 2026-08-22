# Trajectory: sample_0001

## Input
**level:** level2
**problem_id:** 16
**problem_path:** /root/Wafer/research/KernelBench/KernelBench/level2/17_Conv2d_InstanceNorm_Divide.py
**ref_arch_src:** import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """
    def __init__(se

... (truncated, 877 chars total)
**name:** 17_Conv2d_InstanceNorm_Divide
**user_prompt:** Optimize the HIP kernel for 17_Conv2d_InstanceNorm_Divide
**_sample_id:** sample_0001

## Score
- **judge_score:** 0.800
- **judge_score_raw:** 8.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
The agent successfully ran wafer evaluate kernelbench multiple times and achieved a verified 1.12x speedup with 100% correctness. The optimization implements a well-designed fused kernel that combines Instance Normalization and division using the Welford algorithm for numerically stable mean/variance computation. The kernel uses shared memory reduction and processes one instance per block, which is appropriate for the problem size. The agent iteratively improved the implementation and properly tested each version. Minor deductions for not achieving higher speedup, but the approach is solid and the results are verified.
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

... (truncated, 4757 chars total)

### Assistant (Turn 1)
**Tool Call: `write`**
```json
{
  "path": "custom_kernel.py",
  "content": "import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport os\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <torch/extension.h>\n\n__global__ void fused_instance_norm_divide_kernel(\n    float* __restrict__ data,\n    int HW,\n    float divide_by,\n    float epsilon) \n{\n    extern __shared__ float sdata[];\n    \n    int tid = threadIdx.x;\n    int nc = blockIdx.x; \n    \n    // Offset for this instance\n    size_t offset = (size_t)nc * HW;\n    float* img = data + offset;\n\n    // --- Pass 1: Sum ---\n    float local_sum = 0.0f;\n    for (int i = tid; i < HW; i += blockDim.x) {\n        local_sum += img[i];\n    }\n    \n    sdata[tid] = local_sum;\n    __syncthreads();\n    \n    // Reduction\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] += sdata[tid + s];\n        }\n        __syncthreads();\n    }\n    \n    float mean = sdata[0] / HW;\n    \n    __shared__ float sh_mean;\n    if (tid == 0) sh_mean = mean;\n    __syncthreads();\n    mean = sh_mean;\n\n    // --- Pass 2: Sum of Squared Differences ---\n    float local_sq_diff = 0.0f;\n    for (int i = tid; i < HW; i += blockDim.x) {\n        float diff = img[i] - mean;\n        local_sq_diff += diff * diff;\n    }\n    \n    sdata[tid] = local_sq_diff;\n    __syncthreads();\n    \n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] += sdata[tid + s];\n        }\n        __syncthreads();\n    }\n    \n    // --- Pass 3: Normalize & Divide ---\n    __shared__ float sh_scale;\n    if (tid == 0) {\n        float var = sdata[0] / HW;\n        sh_scale = rsqrtf(var + epsilon) / divide_by;\n    }\n    __syncthreads();\n    float scale = sh_scale;\n    \n    for (int i = tid; i < HW; i += blockDim.x) {\n        img[i] = (img[i] - mean) * scale;\n    }\n}\n\ntorch::Tensor fused_instance_norm_divide_hip(torch::Tensor input, float divide_by) {\n    if (!input.is_contiguous()) {\n        input = input.contiguous();\n    }\n    \n    int N = input.size(0);\n    int C = input.size(1);\n    int H = input.size(2);\n    int W = input.size(3);\n    int HW = H * W;\n    \n    int num_instances = N * C;\n    int block_size = 256;\n    size_t shared_mem_size = block_size * sizeof(float);\n    \n    fused_instance_norm_divide_kernel<<<num_instances, block_size, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        HW,\n        divide_by,\n        1e-5f\n    );\n    \n    return input;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_instance_norm_divide\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_instance_norm_divide_hip\"],\n    verbose=True,\n    extra_cflags=['-O3']\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_channels, out_channels, kernel_size, divide_by):\n        super(ModelNew, self).__init__()\n        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)\n        self.divide_by = float(divide_by)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = fused_op.fused_instance_norm_divide_hip(x, self.divide_by)\n        return x\n\nbatch_size = 128\nin_channels  = 64  \nout_channels = 128  \nheight = width = 128  \nkernel_size = 3\ndivide_by = 2.0\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_channels, height, width).cuda()]\n\ndef get_init_inputs():\n    return [in_channels, out_channels, kernel_size, divide_by]\n"
}
```

### Tool Result
```
Successfully wrote 3330 bytes to custom_kernel.py
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/17_Conv2d_InstanceNorm_Divide.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_021645
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_instance_norm_divide -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -c /root/.cache/torch_extensions/py312_cpu/fused_instance_norm_divide/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_instance_norm_divide.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 5.722ms, Ref: 6.036ms, Speedup: 1.05x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_021645/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 1.05x
===========================================================

... (truncated, 1502 chars total)
```

### Assistant (Turn 3)
**Tool Call: `write`**
```json
{
  "content": "import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport os\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <torch/extension.h>\n\nstruct WelfordData {\n    float mean;\n    float m2;\n    int n;\n};\n\n__device__ inline WelfordData welford_combine(WelfordData a, WelfordData b) {\n    if (a.n == 0) return b;\n    if (b.n == 0) return a;\n    \n    WelfordData res;\n    res.n = a.n + b.n;\n    float delta = b.mean - a.mean;\n    \n    float fn = (float)res.n;\n    float fnb = (float)b.n;\n    float fna = (float)a.n;\n    \n    // Standard parallel Welford merge\n    res.mean = a.mean + delta * fnb / fn;\n    res.m2 = a.m2 + b.m2 + delta * delta * fna * fnb / fn;\n    return res;\n}\n\n__global__ void fused_instance_norm_divide_kernel_welford(\n    float* __restrict__ data,\n    int HW,\n    float divide_by,\n    float epsilon) \n{\n    extern __shared__ char smem[];\n    WelfordData* sdata = (WelfordData*)smem;\n    \n    int tid = threadIdx.x;\n    int nc = blockIdx.x; \n    \n    // Offset for this instance\n    size_t offset = (size_t)nc * HW;\n    float* img = data + offset;\n\n    // --- Pass 1: Local Welford ---\n    WelfordData local_data;\n    local_data.mean = 0.0f;\n    local_data.m2 = 0.0f;\n    local_data.n = 0;\n    \n    // Grid stride loop over pixels\n    for (int i = tid; i < HW; i += blockDim.x) {\n        float val = img[i];\n        \n        local_data.n++;\n        float delta = val - local_data.mean;\n        local_data.mean += delta / local_data.n;\n        float delta2 = val - local_data.mean;\n        local_data.m2 += delta * delta2;\n    }\n    \n    sdata[tid] = local_data;\n    __syncthreads();\n    \n    // --- Reduction ---\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] = welford_combine(sdata[tid], sdata[tid + s]);\n        }\n        __syncthreads();\n    }\n    \n    // --- Pass 2: Normalize & Divide ---\n    __shared__ float sh_mean;\n    __shared__ float sh_scale;\n    \n    if (tid == 0) {\n        float mean = sdata[0].mean;\n        float var = sdata[0].m2 / HW; \n        sh_mean = mean;\n        sh_scale = rsqrtf(var + epsilon) / divide_by;\n    }\n    __syncthreads();\n    \n    float mean = sh_mean;\n    float scale = sh_scale;\n    \n    for (int i = tid; i < HW; i += blockDim.x) {\n        img[i] = (img[i] - mean) * scale;\n    }\n}\n\ntorch::Tensor fused_instance_norm_divide_hip(torch::Tensor input, float divide_by) {\n    if (!input.is_contiguous()) {\n        input = input.contiguous();\n    }\n    \n    int N = input.size(0);\n    int C = input.size(1);\n    int H = input.size(2);\n    int W = input.size(3);\n    int HW = H * W;\n    \n    int num_instances = N * C;\n    int block_size = 256;\n    size_t shared_mem_size = block_size * sizeof(WelfordData);\n    \n    fused_instance_norm_divide_kernel_welford<<<num_instances, block_size, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        HW,\n        divide_by,\n        1e-5f\n    );\n    \n    return input;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_instance_norm_divide_v2\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_instance_norm_divide_hip\"],\n    verbose=True,\n    extra_cflags=['-O3']\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_channels, out_channels, kernel_size, divide_by):\n        super(ModelNew, self).__init__()\n        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)\n        self.divide_by = float(divide_by)\n        torch.backends.cudnn.benchmark = True\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = fused_op.fused_instance_norm_divide_hip(x, self.divide_by)\n        return x\n\nbatch_size = 128\nin_channels  = 64  \nout_channels = 128  \nheight = width = 128  \nkernel_size = 3\ndivide_by = 2.0\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_channels, height, width).cuda()]\n\ndef get_init_inputs():\n    return [in_channels, out_channels, kernel_size, divide_by]\n",
  "path": "custom_kernel.py"
}
```

### Tool Result
```
Successfully wrote 3910 bytes to custom_kernel.py
```

### Assistant (Turn 4)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/17_Conv2d_InstanceNorm_Divide.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_023252
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_instance_norm_divide_v2 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -c /root/.cache/torch_extensions/py312_cpu/fused_instance_norm_divide_v2/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_instance_norm_divide_v2.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 5.367ms, Ref: 6.007ms, Speedup: 1.12x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_023252/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 1.12x
==================================================

... (truncated, 1511 chars total)
```

### Assistant (Turn 5)
**Tool Call: `write`**
```json
{
  "content": "import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport os\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <torch/extension.h>\n\nstruct WelfordData {\n    float mean;\n    float m2;\n    int n;\n};\n\n__device__ inline WelfordData welford_combine(WelfordData a, WelfordData b) {\n    if (a.n == 0) return b;\n    if (b.n == 0) return a;\n    \n    WelfordData res;\n    res.n = a.n + b.n;\n    float delta = b.mean - a.mean;\n    \n    float fn = (float)res.n;\n    float fnb = (float)b.n;\n    float fna = (float)a.n;\n    \n    res.mean = a.mean + delta * fnb / fn;\n    res.m2 = a.m2 + b.m2 + delta * delta * fna * fnb / fn;\n    return res;\n}\n\n__global__ void fused_instance_norm_divide_kernel_welford(\n    float* __restrict__ data,\n    int HW,\n    float divide_by,\n    float epsilon) \n{\n    extern __shared__ char smem[];\n    WelfordData* sdata = (WelfordData*)smem;\n    \n    int tid = threadIdx.x;\n    int nc = blockIdx.x; \n    \n    size_t offset = (size_t)nc * HW;\n    float* img = data + offset;\n\n    // --- Pass 1: Local Welford ---\n    WelfordData local_data;\n    local_data.mean = 0.0f;\n    local_data.m2 = 0.0f;\n    local_data.n = 0;\n    \n    for (int i = tid; i < HW; i += blockDim.x) {\n        float val = img[i];\n        local_data.n++;\n        float delta = val - local_data.mean;\n        local_data.mean += delta / local_data.n;\n        float delta2 = val - local_data.mean;\n        local_data.m2 += delta * delta2;\n    }\n    \n    sdata[tid] = local_data;\n    __syncthreads();\n    \n    // --- Reduction ---\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] = welford_combine(sdata[tid], sdata[tid + s]);\n        }\n        __syncthreads();\n    }\n    \n    // --- Pass 2: Normalize & Divide ---\n    __shared__ float sh_mean;\n    __shared__ float sh_scale;\n    \n    if (tid == 0) {\n        float mean = sdata[0].mean;\n        float var = sdata[0].m2 / HW; \n        sh_mean = mean;\n        sh_scale = rsqrtf(var + epsilon) / divide_by;\n    }\n    __syncthreads();\n    \n    float mean = sh_mean;\n    float scale = sh_scale;\n    \n    for (int i = tid; i < HW; i += blockDim.x) {\n        img[i] = (img[i] - mean) * scale;\n    }\n}\n\ntorch::Tensor fused_instance_norm_divide_hip(torch::Tensor input, float divide_by) {\n    if (!input.is_contiguous()) {\n        input = input.contiguous();\n    }\n    \n    int N = input.size(0);\n    int C = input.size(1);\n    int H = input.size(2);\n    int W = input.size(3);\n    int HW = H * W;\n    \n    int num_instances = N * C;\n    int block_size = 256;\n    size_t shared_mem_size = block_size * sizeof(WelfordData);\n    \n    fused_instance_norm_divide_kernel_welford<<<num_instances, block_size, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        HW,\n        divide_by,\n        1e-5f\n    );\n    \n    return input;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_instance_norm_divide_v3\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_instance_norm_divide_hip\"],\n    verbose=True,\n    extra_cflags=['-O3']\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_channels, out_channels, kernel_size, divide_by):\n        super(ModelNew, self).__init__()\n        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)\n        self.divide_by = float(divide_by)\n        torch.backends.cudnn.benchmark = True\n\n    def forward(self, x):\n        # Convert to channels_last for optimization\n        x = x.to(memory_format=torch.channels_last)\n        x = self.conv(x)\n        x = x.contiguous()\n        x = fused_op.fused_instance_norm_divide_hip(x, self.divide_by)\n        return x\n\nbatch_size = 128\nin_channels  = 64  \nout_channels = 128  \nheight = width = 128  \nkernel_size = 3\ndivide_by = 2.0\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_channels, height, width).cuda()]\n\ndef get_init_inputs():\n    return [in_channels, out_channels, kernel_size, divide_by]\n",
  "path": "custom_kernel.py"
}
```

### Tool Result
```
Successfully wrote 3925 bytes to custom_kernel.py
```

### Assistant (Turn 6)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/17_Conv2d_InstanceNorm_Divide.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_024849
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_instance_norm_divide_v3 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -c /root/.cache/torch_extensions/py312_cpu/fused_instance_norm_divide_v3/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_instance_norm_divide_v3.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 8.986ms, Ref: 6.049ms, Speedup: 0.67x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_024849/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.67x
==================================================

... (truncated, 1511 chars total)
```

### Assistant (Turn 7)
**Tool Call: `write`**
```json
{
  "path": "custom_kernel.py",
  "content": "import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport os\n\nos.environ[\"CXX\"] = \"hipcc\"\n\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <torch/extension.h>\n\nstruct WelfordData {\n    float mean;\n    float m2;\n    int n;\n};\n\n__device__ inline WelfordData welford_combine(WelfordData a, WelfordData b) {\n    if (a.n == 0) return b;\n    if (b.n == 0) return a;\n    \n    WelfordData res;\n    res.n = a.n + b.n;\n    float delta = b.mean - a.mean;\n    \n    float fn = (float)res.n;\n    float fnb = (float)b.n;\n    float fna = (float)a.n;\n    \n    res.mean = a.mean + delta * fnb / fn;\n    res.m2 = a.m2 + b.m2 + delta * delta * fna * fnb / fn;\n    return res;\n}\n\n__global__ void fused_instance_norm_divide_kernel_welford(\n    float* __restrict__ data,\n    int HW,\n    float divide_by,\n    float epsilon) \n{\n    extern __shared__ char smem[];\n    WelfordData* sdata = (WelfordData*)smem;\n    \n    int tid = threadIdx.x;\n    int nc = blockIdx.x; \n    \n    size_t offset = (size_t)nc * HW;\n    float* img = data + offset;\n\n    // --- Pass 1: Local Welford ---\n    WelfordData local_data;\n    local_data.mean = 0.0f;\n    local_data.m2 = 0.0f;\n    local_data.n = 0;\n    \n    for (int i = tid; i < HW; i += blockDim.x) {\n        float val = img[i];\n        local_data.n++;\n        float delta = val - local_data.mean;\n        local_data.mean += delta / local_data.n;\n        float delta2 = val - local_data.mean;\n        local_data.m2 += delta * delta2;\n    }\n    \n    sdata[tid] = local_data;\n    __syncthreads();\n    \n    // --- Reduction ---\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] = welford_combine(sdata[tid], sdata[tid + s]);\n        }\n        __syncthreads();\n    }\n    \n    // --- Pass 2: Normalize & Divide ---\n    __shared__ float sh_mean;\n    __shared__ float sh_scale;\n    \n    if (tid == 0) {\n        float mean = sdata[0].mean;\n        float var = sdata[0].m2 / HW; \n        sh_mean = mean;\n        sh_scale = rsqrtf(var + epsilon) / divide_by;\n    }\n    __syncthreads();\n    \n    float mean = sh_mean;\n    float scale = sh_scale;\n    \n    for (int i = tid; i < HW; i += blockDim.x) {\n        img[i] = (img[i] - mean) * scale;\n    }\n}\n\ntorch::Tensor fused_instance_norm_divide_hip(torch::Tensor input, float divide_by) {\n    if (!input.is_contiguous()) {\n        input = input.contiguous();\n    }\n    \n    int N = input.size(0);\n    int C = input.size(1);\n    int H = input.size(2);\n    int W = input.size(3);\n    int HW = H * W;\n    \n    int num_instances = N * C;\n    int block_size = 256;\n    size_t shared_mem_size = block_size * sizeof(WelfordData);\n    \n    fused_instance_norm_divide_kernel_welford<<<num_instances, block_size, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        HW,\n        divide_by,\n        1e-5f\n    );\n    \n    return input;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_instance_norm_divide_v4\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_instance_norm_divide_hip\"],\n    verbose=True,\n    extra_cflags=['-O3']\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_channels, out_channels, kernel_size, divide_by):\n        super(ModelNew, self).__init__()\n        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)\n        self.divide_by = float(divide_by)\n        torch.backends.cudnn.benchmark = True\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = fused_op.fused_instance_norm_divide_hip(x, self.divide_by)\n        return x\n\nbatch_size = 128\nin_channels  = 64  \nout_channels = 128  \nheight = width = 128  \nkernel_size = 3\ndivide_by = 2.0\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_channels, height, width).cuda()]\n\ndef get_init_inputs():\n    return [in_channels, out_channels, kernel_size, divide_by]\n"
}
```

### Tool Result
```
Successfully wrote 3794 bytes to custom_kernel.py
```

### Assistant (Turn 8)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation custom_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/17_Conv2d_InstanceNorm_Divide.py --benchmark --defensive"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_030233
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_instance_norm_divide_v4 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -O3 -c /root/.cache/torch_extensions/py312_cpu/fused_instance_norm_divide_v4/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_instance_norm_divide_v4.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 5.349ms, Ref: 6.008ms, Speedup: 1.12x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_030233/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 1.12x
==================================================

... (truncated, 1511 chars total)
```

### Assistant (Turn 9)
DONE

### Assistant (Turn 10)


### Assistant (Turn 11)


### Assistant (Turn 12)


## Expected Answer
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = x / self.divide_by
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
