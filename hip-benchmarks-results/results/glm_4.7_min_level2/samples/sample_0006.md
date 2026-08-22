# Trajectory: sample_0006

## Input
**level:** level2
**problem_id:** 54
**problem_path:** /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py
**ref_arch_src:** import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_featur

... (truncated, 1073 chars total)
**name:** 55_Matmul_MaxPool_Sum_Scale
**user_prompt:** Optimize the HIP kernel for 55_Matmul_MaxPool_Sum_Scale
**_sample_id:** sample_0006

## Score
- **judge_score:** 0.200
- **judge_score_raw:** 2.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
Agent ran wafer evaluate kernelbench multiple times as required, showing proper testing methodology. However, the final HIP kernel implementation failed correctness tests with 100% wrong elements and massive numerical errors (max diff 1545, mean 98% relative error). Earlier attempts showed correct implementation with 0.99x speedup but no performance gain. The optimization approach was sound (fusion of maxpool+sum+scale operations) but the custom HIP kernel had a critical bug causing completely wrong results. Credit given only for proper testing procedure and initial correct implementation, but major deduction for final incorrect kernel.
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

... (truncated, 4949 chars total)

### Assistant (Turn 1)
Looking atLooking at this architecture, I can identify several optimization opportunities:

1. **Fuse MaxPool + Sum + Scale** into a single kernel - this eliminates intermediate memory allocations and kernel launches
2. The kernel can perform a global max-pooling (kernel_size=2) followed by sum reduction and scaling in a fused operation
3. Keep the.Linear (matmul) as-is since it's already highly optimized in cuBLAS/rocBLAS

Let me create the optimized imple

**Tool Call: `write`**
```json
{
  "path": "optimized_model.py",
  "content": "import os\n\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\nmax_pool_sum_scale_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n__global__ void max_pool_sum_scale_kernel(const float* x, float* out, int batch_size, int features, float scale) {\n    // Each thread block processes one batch element\n    // Each thread within the block starts with one pooled value (from max pool of 2)\n    // Then we reduce within the block to sum all pooled values\n    \n    int batch_idx = blockIdx.x;\n    if (batch_idx >= batch_size) return;\n    \n    int pooled_features = features / 2;  // After max pooling with kernel size 2\n    \n    __shared__ float sdata[512];\n    \n    int pooled_idx = threadIdx.x;\n    \n    // Each thread handles one pooled feature\n    // The pooled value is max of two consecutive features\n    float local_val;\n    if (pooled_idx < pooled_features) {\n        int idx0 = batch_idx * features + pooled_idx * 2;\n        int idx1 = batch_idx * features + pooled_idx * 2 + 1;\n        local_val = max(x[idx0], x[idx1]);\n    } else {\n        local_val = 0.0f;\n    }\n    \n    sdata[threadIdx.x] = local_val;\n    __syncthreads();\n    \n    // Reduction: sum all pooled values for this batch element\n    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (threadIdx.x < s) {\n            sdata[threadIdx.x] += sdata[threadIdx.x + s];\n        }\n        __syncthreads();\n    }\n    \n    // Thread 0 writes the scaled sum\n    if (threadIdx.x == 0) {\n        out[batch_idx] = sdata[0] * scale;\n    }\n}\n\ntorch::Tensor max_pool_sum_scale_hip(torch::Tensor x, float scale) {\n    int batch_size = x.size(0);\n    int features = x.size(1);\n    int pooled_features = features / 2;\n    \n    auto out = torch::zeros({batch_size}, x.options());\n    \n    const int block_size = 512;\n    int num_blocks = batch_size;\n    \n    max_pool_sum_scale_kernel<<<num_blocks, block_size>>>(\n        x.data_ptr<float>(), \n        out.data_ptr<float>(), \n        batch_size, \n        features, \n        scale\n    );\n    \n    return out;\n}\n\"\"\"\n\nmax_pool_sum_scale = load_inline(\n    name=\"max_pool_sum_scale\",\n    cpp_sources=max_pool_sum_scale_cpp_source,\n    functions=[\"max_pool_sum_scale_hip\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.matmul = nn.Linear(in_features, out_features)\n        self.max_pool_sum_scale = max_pool_sum_scale\n        self.scale_factor = scale_factor\n\n    def forward(self, x):\n        \"\"\"\n        Args:\n            x (torch.Tensor): Input tensor of shape (batch_size, in_features).\n\n        Returns:\n            torch.Tensor: Output tensor of shape (batch_size,).\n        \"\"\"\n        # Keep the optimized cuBLAS/rocBLAS matmul\n        x = self.matmul(x)\n        \n        # Fuse MaxPool + Sum + Scale into a single kernel\n        x = self.max_pool_sum_scale.max_pool_sum_scale_hip(x, self.scale_factor)\n        \n        return x"
}
```

### Tool Result
```
Successfully wrote 3042 bytes to optimized_model.py
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_064001
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=max_pool_sum_scale -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/max_pool_sum_scale/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o max_pool_sum_scale.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
LLVM ERROR: IO failure on output stream: No space left on device

Error: Evaluation failed. Log tail:
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=max_pool_sum_scale -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/tor

... (truncated, 2038 chars total)
```

### Assistant (Turn 3)
Let me tryLet me try again - the compilation succeeded but there was a disk space issue. Let

**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_064350
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=max_pool_sum_scale -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/max_pool_sum_scale/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o max_pool_sum_scale.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
LLVM ERROR: IO failure on output stream: No space left on device

Error: Evaluation failed. Log tail:
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=max_pool_sum_scale -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/tor

... (truncated, 2038 chars total)
```

### Assistant (Turn 4)
There's aThere's a disk space issue on the remote system. Let me wait a moment and 

**Tool Call: `bash`**
```json
{
  "command": "sleep 5 && wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Command(s) require approval: 'sleep 5'
This environment only permits commands starting with: 'wafer evaluate', 'python', 'python3', 'ls', 'cat', 'head', 'tail', 'wc', 'find', 'cd', 'pwd', 'which'
```

### Assistant (Turn 5)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_065220
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=max_pool_sum_scale -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/max_pool_sum_scale/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o max_pool_sum_scale.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
LLVM ERROR: IO failure on output stream: No space left on device

Error: Evaluation failed. Log tail:
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=max_pool_sum_scale -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/tor

... (truncated, 2038 chars total)
```

### Assistant (Turn 6)
Let me tryLet me try an alternative approach using torch.compile which doesn't require C++ co

**Tool Call: `write`**
```json
{
  "path": "optimized_model_v2.py",
  "content": "import os\n\nimport torch\nimport torch.nn as nn\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.matmul = nn.Linear(in_features, out_features)\n        self.scale_factor = scale_factor\n        self.kernel_size = kernel_size\n\n    def forward(self, x):\n        \"\"\"\n        Args:\n            x (torch.Tensor): Input tensor of shape (batch_size, in_features).\n\n        Returns:\n            torch.Tensor: Output tensor of shape (batch_size,).\n        \"\"\"\n        # Keep the optimized cuBLAS/rocBLAS matmul\n        x = self.matmul(x)\n        \n        # Use torch.compile to optimize the remaining operations\n        # This should fuse operations at the kernel level\n        pooled_out = torch.nn.functional.max_pool1d(\n            x.unsqueeze(1), \n            kernel_size=self.kernel_size, \n            stride=self.kernel_size\n        ).squeeze(1)\n        \n        sum_out = torch.sum(pooled_out, dim=1)\n        result = sum_out * self.scale_factor\n        \n        return result\n\n\n# Use torch.compile to optimize the forward pass\nmodel_new_factory = lambda in_features, out_features, kernel_size, scale_factor: torch.compile(\n    ModelNew(in_features, out_features, kernel_size, scale_factor),\n    mode=\"max-autotune\"\n)"
}
```

### Tool Result
```
Successfully wrote 1352 bytes to optimized_model_v2.py
```

### Assistant (Turn 7)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model_v2.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_065909
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
Successfully preprocessed all matching files.
/root/.cache/torch_extensions/py312_cpu/fused_lgs_tiled/main.cpp -> /root/.cache/torch_extensions/py312_cpu/fused_lgs_tiled/main.cpp [skipped, no changes]
Total number of unsupported CUDA function calls: 0


Total number of replaced kernel launches: 0
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_lgs_tiled -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/THH -isystem /opt/rocm-7.1.1/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_lgs_tiled/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -lc10_hip -ltorch_cpu -ltorch_hip -ltorch -ltorch_python -L/opt/rocm-7.1.1/lib -lamdhip64 -o fused_lgs_tiled.so
[KernelBench] Modules loaded successfully
[KernelBench] Model

... (truncated, 3483 chars total)
```

### Assistant (Turn 8)
The errorThe error shows the old compiled extension is interfering. Let me create a cleaner implementation with the same approach 

**Tool Call: `write`**
```json
{
  "path": "optimized_model_v3.py",
  "content": "import os\n\nimport torch\nimport torch.nn as nn\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Use plain PyTorch with simple layout optimization\n# The key insight is that we can avoid unnecessary transpose/reshape operations\n# and rely on PyTorch's native optimized kernels\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.matmul = nn.Linear(in_features, out_features)\n        self.scale_factor = scale_factor\n        self.kernel_size = kernel_size\n\n    def forward(self, x):\n        \"\"\"\n        Args:\n            x (torch.Tensor): Input tensor of shape (batch_size, in_features).\n\n        Returns:\n            torch.Tensor: Output tensor of shape (batch_size,).\n        \"\"\"\n        # Linear layer (already optimized with rocBLAS/cuBLAS)\n        x = self.matmul(x)\n        \n        # Manual max pooling with kernel_size=2: stride=kernel_size\n        # Since kernel_size is small (2), manual implementation is efficient\n        batch_size, features = x.shape\n        # Reshape to view pairs [batch, features//2, 2] for max operation\n        x_reshaped = x.view(batch_size, features // 2, 2)\n        # Compute max across the last dimension\n        x_pooled = x_reshaped.max(dim=2).values\n        \n        # Sum and scale\n        result = x_pooled.sum(dim=1) * self.scale_factor\n        \n        return result"
}
```

### Tool Result
```
Successfully wrote 1401 bytes to optimized_model_v3.py
```

### Assistant (Turn 9)
**Tool Call: `write`**
```json
{
  "path": "optimized_model_v3.py",
  "content": "import os\n\nimport torch\nimport torch.nn as nn\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Use plain PyTorch with simple layout optimization\n# The key insight is that we can avoid unnecessary transpose/reshape operations\n# and rely on PyTorch's native optimized kernels\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.matmul = nn.Linear(in_features, out_features)\n        self.scale_factor = scale_factor\n        self.kernel_size = kernel_size\n\n    def forward(self, x):\n        \"\"\"\n        Args:\n            x (torch.Tensor): Input tensor of shape (batch_size, in_features).\n\n        Returns:\n            torch.Tensor: Output tensor of shape (batch_size,).\n        \"\"\"\n        # Linear layer (already optimized with rocBLAS/cuBLAS)\n        x = self.matmul(x)\n        \n        # Manual max pooling with kernel_size=2: stride=kernel_size\n        # Since kernel_size is small (2), manual implementation is efficient\n        batch_size, features = x.shape\n        # Reshape to view pairs [batch, features//2, 2] for max operation\n        x_reshaped = x.view(batch_size, features // 2, 2)\n        # Compute max across the last dimension\n        x_pooled = torch.max(x_reshaped, dim=2).values\n        \n        # Sum and scale\n        result = x_pooled.sum(dim=1) * self.scale_factor\n        \n        return result"
}
```

### Tool Result
```
Successfully wrote 1408 bytes to optimized_model_v3.py
```

### Assistant (Turn 10)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model_v3.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_070023
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 2.725ms, Ref: 2.695ms, Speedup: 0.99x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_070023/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.99x
============================================================

```

### Assistant (Turn 11)
MyMy implementation is correct but shows no speedup (0.99x). I need to use a fused custom HIP kernel for better performance. Let me create a more optimized version with proper kern

**Tool Call: `write`**
```json
{
  "path": "optimized_model_v4.py",
  "content": "import os\n\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Fused kernel that combines matmul weight loading memory optimization\n# Actually, the best approach is to compile away the kernel launch overhead\n# by fusing operations using tensor operations that get compiled together\n\nfused_ops_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n__global__ void fused_maxpool_sum_scale_kernel(const float* x, float* out, int batch_size, int features, float scale) {\n    int batch_idx = blockIdx.x;\n    if (batch_idx >= batch_size) return;\n    \n    int pooled_features = features / 2;\n    \n    // Use block-level reduction\n    extern __shared__ float sdata[];\n    \n    int pooled_idx = threadIdx.x;\n    \n    // Each thread processes one pooled feature\n    // Max pool of 2 consecutive elements\n    float local_val = 0.0f;\n    if (pooled_idx < pooled_features) {\n        int base = batch_idx * features + pooled_idx * 2;\n        float a = x[base];\n        float b = x[base + 1];\n        local_val = (a > b) ? a : b;\n    }\n    \n    sdata[threadIdx.x] = local_val;\n    __syncthreads();\n    \n    // Parallel reduction (tree-based sum)\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (threadIdx.x < s) {\n            sdata[threadIdx.x] += sdata[threadIdx.x + s];\n        }\n        __syncthreads();\n    }\n    \n    // Thread 0 writes scaled result\n    if (threadIdx.x == 0) {\n        out[batch_idx] = sdata[0] * scale;\n    }\n}\n\ntorch::Tensor fused_maxpool_sum_scale_hip(torch::Tensor x, float scale_factor) {\n    int batch_size = x.size(0);\n    int features = x.size(1);\n    int pooled_features = features / 2;\n    \n    auto out = torch::zeros({batch_size}, x.options());\n    \n    // Use 256 threads per block\n    const int block_size = 256;\n    int num_blocks = batch_size;\n    int shared_mem_size = block_size * sizeof(float);\n    \n    fused_maxpool_sum_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(\n        x.data_ptr<float>(), \n        out.data_ptr<float>(), \n        batch_size, \n        features, \n        scale_factor\n    );\n    \n    return out;\n}\n\"\"\"\n\nfused_ops = load_inline(\n    name=\"fused_ops\",\n    cpp_sources=fused_ops_cpp_source,\n    functions=[\"fused_maxpool_sum_scale_hip\"],\n    verbose=False,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.matmul = nn.Linear(in_features, out_features)\n        self.fused_kernel = fused_ops\n        self.scale_factor = scale_factor\n\n    def forward(self, x):\n        \"\"\"\n        Args:\n            x (torch.Tensor): Input tensor of shape (batch_size, in_features).\n\n        Returns:\n            torch.Tensor: Output tensor of shape (batch_size,).\n        \"\"\"\n        # Linear layer uses rocBLAS/cuBLAS which is highly optimized\n        x = self.matmul(x)\n        \n        # Use fused kernel for maxpool + sum + scale\n        x = self.fused_kernel.fused_maxpool_sum_scale_hip(x, self.scale_factor)\n        \n        return x"
}
```

### Tool Result
```
Successfully wrote 3065 bytes to optimized_model_v4.py
```

### Assistant (Turn 12)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model_v4.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_070239
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Diff analysis:
   Max diff: 1545.417358 at index [74]
   Mean diff: 1514.110352
   Max relative error: 98.73%, Mean: 98.56%
   Wrong elements: 128 / 128 (100.00%)
   Error distribution: <1e-6: 0 | 1e-6~1e-4: 0 | 1e-4~1e-2: 0 | 1e-2~1: 0 | >1: 128
   Pattern: ALL elements wrong - likely algorithmic error or wrong weights
   Sample failures:
      [0]: ref=1524.415283 impl=22.431194 (diff=1501.984131)
      [1]: ref=1557.172119 impl=22.610577 (diff=1534.561523)
      [2]: ref=1531.273682 impl=21.358742 (diff=1509.914917)
      [3]: ref=1537.487305 impl=24.001263 (diff=1513.486084)
      [4]: ref=1532.040283 impl=21.929455 (diff=1510.110840)
[KernelBench] Debug tensors saved to: /workspace/kernelbench_eval_20260122_070239/debug/
[KernelBench] Correctness: False
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_070239/results.json

Error: Correctness failed on trial 1: max diff = 1545.4173583984375


Error: Command exited with code 1
```

## Expected Answer
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.max_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.sum(x, dim=1)
        x = x * self.scale_factor
        return x

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
