# Trajectory: sample_0013

## Input
**level:** level2
**problem_id:** 97
**problem_path:** /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py
**ref_arch_src:** import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool

... (truncated, 1144 chars total)
**name:** 98_Matmul_AvgPool_GELU_Scale_Max
**user_prompt:** Optimize the HIP kernel for 98_Matmul_AvgPool_GELU_Scale_Max
**_sample_id:** sample_0013

## Score
- **judge_score:** 0.900
- **judge_score_raw:** 9.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
The agent successfully ran wafer evaluate kernelbench multiple times and achieved excellent results. The final implementation passes correctness tests (100%) and delivers a significant 9.16x speedup. The optimization strategy is sophisticated - it cleverly reduces computation by pooling weights during initialization rather than computing average pooling at runtime, then fuses GELU, scaling, and max reduction into a single efficient HIP kernel using shared memory reduction. The code is well-structured with proper error handling and device compatibility. Minor deduction for the iterative development process, but the final result demonstrates strong GPU optimization skills with verified performance gains.
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

... (truncated, 5030 chars total)

### Assistant (Turn 1)
**Tool Call: `write`**
```json
{
  "content": "\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport os\n\nos.environ[\"CXX\"] = \"hipcc\"\n\nfused_ops_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <torch/extension.h>\n#include <cmath>\n#include <algorithm>\n\n__device__ inline float gelu(float x) {\n    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));\n}\n\n__global__ void fused_ops_kernel(\n    const float* __restrict__ input,\n    float* __restrict__ output,\n    int batch_size,\n    int out_features,\n    int pool_kernel_size,\n    float scale_factor,\n    int num_pooled) {\n\n    int row = blockIdx.x;\n    if (row >= batch_size) return;\n\n    extern __shared__ float shared_data[];\n\n    float max_val = -1e20f;\n    \n    // Each thread calculates one or more pooled values\n    for (int i = threadIdx.x; i < num_pooled; i += blockDim.x) {\n        float sum = 0.0f;\n        int start_idx = i * pool_kernel_size;\n        for (int k = 0; k < pool_kernel_size; ++k) {\n            sum += input[row * out_features + start_idx + k];\n        }\n        float avg = sum / (float)pool_kernel_size;\n        float val = gelu(avg) * scale_factor;\n        if (val > max_val) max_val = val;\n    }\n\n    // Now reduce max_val within the block\n    shared_data[threadIdx.x] = max_val;\n    __syncthreads();\n\n    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n        if (threadIdx.x < stride) {\n            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);\n        }\n        __syncthreads();\n    }\n\n    if (threadIdx.x == 0) {\n        output[row] = shared_data[0];\n    }\n}\n\ntorch::Tensor fused_ops_hip(torch::Tensor input, int pool_kernel_size, float scale_factor) {\n    int batch_size = input.size(0);\n    int out_features = input.size(1);\n    int num_pooled = out_features / pool_kernel_size;\n    \n    auto output = torch::empty({batch_size}, input.options());\n\n    int block_size = 256;\n    if (num_pooled < block_size) {\n        // Adjust block size if num_pooled is small, but 256 is usually fine.\n    }\n    \n    dim3 grid(batch_size);\n    dim3 block(block_size);\n    int shared_mem_size = block_size * sizeof(float);\n\n    fused_ops_kernel<<<grid, block, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        batch_size,\n        out_features,\n        pool_kernel_size,\n        scale_factor,\n        num_pooled\n    );\n\n    return output;\n}\n\"\"\"\n\nfused_ops_lib = load_inline(\n    name=\"fused_ops_lib\",\n    cpp_sources=fused_ops_cpp_source,\n    functions=[\"fused_ops_hip\"],\n    verbose=True,\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.matmul = nn.Linear(in_features, out_features)\n        self.pool_kernel_size = pool_kernel_size\n        self.scale_factor = scale_factor\n        self.fused_ops_lib = fused_ops_lib\n\n    def forward(self, x):\n        x = self.matmul(x)\n        return self.fused_ops_lib.fused_ops_hip(x, self.pool_kernel_size, self.scale_factor)\n",
  "path": "kernel.py"
}
```

### Tool Result
```
Successfully wrote 3050 bytes to kernel.py
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive\n"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260121_214514
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_ops_lib -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_ops_lib/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_ops_lib.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 1.259ms, Ref: 1.235ms, Speedup: 0.98x
[KernelBench] Results written to /workspace/kernelbench_eval_20260121_214514/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.98x
============================================================

```

### Assistant (Turn 3)
**Tool Call: `write`**
```json
{
  "content": "\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport os\n\nos.environ[\"CXX\"] = \"hipcc\"\n\nfused_ops_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <torch/extension.h>\n#include <cmath>\n#include <algorithm>\n\n__device__ inline float gelu(float x) {\n    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));\n}\n\n__global__ void post_matmul_fused_kernel(\n    const float* __restrict__ input, // shape: (batch_size, num_pooled)\n    float* __restrict__ output,       // shape: (batch_size,)\n    int batch_size,\n    int num_pooled,\n    float scale_factor) {\n\n    int row = blockIdx.x;\n    if (row >= batch_size) return;\n\n    extern __shared__ float shared_data[];\n\n    float max_val = -1e20f;\n    for (int i = threadIdx.x; i < num_pooled; i += blockDim.x) {\n        float val = gelu(input[row * num_pooled + i]) * scale_factor;\n        if (val > max_val) max_val = val;\n    }\n\n    shared_data[threadIdx.x] = max_val;\n    __syncthreads();\n\n    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n        if (threadIdx.x < stride) {\n            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);\n        }\n        __syncthreads();\n    }\n\n    if (threadIdx.x == 0) {\n        output[row] = shared_data[0];\n    }\n}\n\ntorch::Tensor post_matmul_fused_hip(torch::Tensor input, float scale_factor) {\n    int batch_size = input.size(0);\n    int num_pooled = input.size(1);\n    \n    auto output = torch::empty({batch_size}, input.options());\n\n    int block_size = 256;\n    dim3 grid(batch_size);\n    dim3 block(block_size);\n    int shared_mem_size = block_size * sizeof(float);\n\n    post_matmul_fused_kernel<<<grid, block, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        batch_size,\n        num_pooled,\n        scale_factor\n    );\n\n    return output;\n}\n\"\"\"\n\npost_matmul_fused_lib = load_inline(\n    name=\"post_matmul_fused_lib\",\n    cpp_sources=fused_ops_cpp_source,\n    functions=[\"post_matmul_fused_hip\"],\n    verbose=True,\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.in_features = in_features\n        self.out_features = out_features\n        self.pool_kernel_size = pool_kernel_size\n        self.scale_factor = scale_factor\n        \n        # We still need the original Linear layer to match the original Model's parameters\n        self.matmul = nn.Linear(in_features, out_features)\n        self.post_matmul_fused_lib = post_matmul_fused_lib\n        \n        self.initialized = False\n        self.weight_pooled = None\n        self.bias_pooled = None\n\n    def _initialize_pooled_weights(self):\n        # pool the weights and bias\n        # weight shape is (out_features, in_features)\n        # bias shape is (out_features)\n        with torch.no_grad():\n            w = self.matmul.weight\n            b = self.matmul.bias\n            num_pooled = self.out_features // self.pool_kernel_size\n            \n            # W_pooled: (num_pooled, in_features)\n            self.weight_pooled = w[:num_pooled * self.pool_kernel_size, :].view(\n                num_pooled, self.pool_kernel_size, self.in_features\n            ).mean(dim=1)\n            \n            if b is not None:\n                self.bias_pooled = b[:num_pooled * self.pool_kernel_size].view(\n                    num_pooled, self.pool_kernel_size\n                ).mean(dim=1)\n            else:\n                self.bias_pooled = None\n        self.initialized = True\n\n    def forward(self, x):\n        if not self.initialized:\n            self._initialize_pooled_weights()\n        \n        # New matmul: (batch_size, in_features) @ (in_features, num_pooled)\n        # weight_pooled is (num_pooled, in_features), so we transpose it.\n        x = torch.matmul(x, self.weight_pooled.t())\n        if self.bias_pooled is not None:\n            x = x + self.bias_pooled\n            \n        return self.post_matmul_fused_lib.post_matmul_fused_hip(x, self.scale_factor)\n",
  "path": "kernel.py"
}
```

### Tool Result
```
Successfully wrote 4061 bytes to kernel.py
```

### Assistant (Turn 4)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive\n"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260121_214625
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=post_matmul_fused_lib -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/post_matmul_fused_lib/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o post_matmul_fused_lib.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 0.138ms, Ref: 1.282ms, Speedup: 9.30x
[KernelBench] Results written to /workspace/kernelbench_eval_20260121_214625/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 9.30x
============================================================

```

### Assistant (Turn 5)
**Tool Call: `write`**
```json
{
  "content": "\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport os\n\nos.environ[\"CXX\"] = \"hipcc\"\n\nfused_ops_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <torch/extension.h>\n#include <cmath>\n#include <algorithm>\n\n__device__ inline float gelu(float x) {\n    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));\n}\n\n__global__ void post_matmul_fused_kernel(\n    const float* __restrict__ input, // shape: (batch_size, num_pooled)\n    float* __restrict__ output,       // shape: (batch_size,)\n    int batch_size,\n    int num_pooled,\n    float scale_factor) {\n\n    int row = blockIdx.x;\n    if (row >= batch_size) return;\n\n    extern __shared__ float shared_data[];\n\n    float max_val = -1e20f;\n    for (int i = threadIdx.x; i < num_pooled; i += blockDim.x) {\n        float val = gelu(input[row * num_pooled + i]) * scale_factor;\n        if (val > max_val) max_val = val;\n    }\n\n    shared_data[threadIdx.x] = max_val;\n    __syncthreads();\n\n    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n        if (threadIdx.x < stride) {\n            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);\n        }\n        __syncthreads();\n    }\n\n    if (threadIdx.x == 0) {\n        output[row] = shared_data[0];\n    }\n}\n\ntorch::Tensor post_matmul_fused_hip(torch::Tensor input, float scale_factor) {\n    int batch_size = input.size(0);\n    int num_pooled = input.size(1);\n    \n    auto output = torch::empty({batch_size}, input.options());\n\n    int block_size = 256;\n    dim3 grid(batch_size);\n    dim3 block(block_size);\n    int shared_mem_size = block_size * sizeof(float);\n\n    post_matmul_fused_kernel<<<grid, block, shared_mem_size>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        batch_size,\n        num_pooled,\n        scale_factor\n    );\n\n    return output;\n}\n\"\"\"\n\npost_matmul_fused_lib = load_inline(\n    name=\"post_matmul_fused_lib\",\n    cpp_sources=fused_ops_cpp_source,\n    functions=[\"post_matmul_fused_hip\"],\n    verbose=True,\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.in_features = in_features\n        self.out_features = out_features\n        self.pool_kernel_size = pool_kernel_size\n        self.scale_factor = float(scale_factor)\n        \n        self.matmul = nn.Linear(in_features, out_features)\n        self.post_matmul_fused_lib = post_matmul_fused_lib\n        \n        # We'll use these to cache the pooled weight/bias\n        self.register_buffer('weight_pooled', None)\n        self.register_buffer('bias_pooled', None)\n\n    def _initialize_pooled_weights(self):\n        # pool the weights and bias\n        with torch.no_grad():\n            w = self.matmul.weight\n            b = self.matmul.bias\n            num_pooled = self.out_features // self.pool_kernel_size\n            \n            # W_pooled: (num_pooled, in_features)\n            wp = w[:num_pooled * self.pool_kernel_size, :].view(\n                num_pooled, self.pool_kernel_size, self.in_features\n            ).mean(dim=1)\n            self.weight_pooled = wp\n            \n            if b is not None:\n                bp = b[:num_pooled * self.pool_kernel_size].view(\n                    num_pooled, self.pool_kernel_size\n                ).mean(dim=1)\n                self.bias_pooled = bp\n            else:\n                self.bias_pooled = None\n\n    def forward(self, x):\n        if self.weight_pooled is None:\n            self._initialize_pooled_weights()\n        \n        # New matmul: (batch_size, in_features) @ (in_features, num_pooled)\n        # weight_pooled is (num_pooled, in_features), so we transpose it.\n        # Ensure x and weights are on the same device and use same dtype\n        x = torch.matmul(x, self.weight_pooled.t())\n        if self.bias_pooled is not None:\n            x = x + self.bias_pooled\n            \n        return self.post_matmul_fused_lib.post_matmul_fused_hip(x, self.scale_factor)\n",
  "path": "kernel.py"
}
```

### Tool Result
```
Successfully wrote 4036 bytes to kernel.py
```

### Assistant (Turn 6)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive\n"
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260121_214806
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=post_matmul_fused_lib -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/post_matmul_fused_lib/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o post_matmul_fused_lib.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 0.138ms, Ref: 1.265ms, Speedup: 9.16x
[KernelBench] Results written to /workspace/kernelbench_eval_20260121_214806/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 9.16x
============================================================

```

### Assistant (Turn 7)
DONE

### Assistant (Turn 8)

DONE

### Assistant (Turn 9)


### Assistant (Turn 10)


### Assistant (Turn 11)


### Assistant (Turn 12)

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_ops_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
}

__global__ void post_matmul_fused_kernel(
    const float* __restrict__ input, // shape: (batch_size, num_pooled)
    float* __restrict__ output,       // shape: (batch_size,)
    int batch_size,
    int num_pooled,
    float scale_factor) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared_data[];

    float max_val = -1e20f;
    for (int i = threadIdx.x; i < num_pooled; i += blockDim.x) {
        float val = gelu(input[row * num_pooled + i]) * scale_factor;
        if (val > max_val) max_val = val;
    }

    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = shared_data[0];
    }
}

torch::Tensor post_matmul_fused_hip(torch::Tensor input, float scale_factor) {
    int batch_size = input.size(0);
    int num_pooled = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());

    int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    int shared_mem_size = block_size * sizeof(float);

    post_matmul_fused_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_pooled,
        scale_factor
    );

    return output;
}
"""

post_matmul_fused_lib = load_inline(
    name="post_matmul_fused_lib",
    cpp_sources=fused_ops_cpp_source,
    functions=["post_matmul_fused_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        
        self.matmul = nn.Linear(in_features, out_features)
        self.post_matmul_fused_lib = post_matmul_fused_lib
        
        self.register_buffer('weight_pooled', None)
        self.register_buffer('bias_pooled', None)

    def _initialize_pooled_weights(self):
        with torch.no_grad():
            w = self.matmul.weight
            b = self.matmul.bias
            num_pooled = self.out_features // self.pool_kernel_size
            
            wp = w[:num_pooled * self.pool_kernel_size, :].view(
                num_pooled, self.pool_kernel_size, self.in_features
            ).mean(dim=1)
            self.weight_pooled = wp
            
            if b is not None:
                bp = b[:num_pooled * self.pool_kernel_size].view(
                    num_pooled, self.pool_kernel_size
                ).mean(dim=1)
                self.bias_pooled = bp
            else:
                self.bias_pooled = None

    def forward(self, x):
        if self.weight_pooled is None:
            self._initialize_pooled_weights()
        
        x = torch.matmul(x, self.weight_pooled.t())
        if self.bias_pooled is not None:
            x = x + self.bias_pooled
            
        return self.post_matmul_fused_lib.post_matmul_fused_hip(x, self.scale_factor)

DONE

## Expected Answer
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.nn.functional.gelu(x)
        x = x * self.scale_factor
        x = torch.max(x, dim=1).values
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
