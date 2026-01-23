# Trajectory: sample_0010

## Input
**level:** level1
**problem_id:** 41
**problem_path:** /root/Wafer/research/KernelBench/KernelBench/level1/42_Max_Pooling_2D.py
**ref_arch_src:** import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation:

... (truncated, 1366 chars total)
**name:** 42_Max_Pooling_2D
**user_prompt:** Optimize the HIP kernel for 42_Max_Pooling_2D
**_sample_id:** sample_0010

## Score
- **judge_score:** 0.700
- **judge_score_raw:** 7.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
The agent successfully ran wafer evaluate kernelbench and achieved a 1.22x speedup with verified correctness. The HIP kernel implements a straightforward max pooling operation with proper bounds checking and memory access patterns. The optimization includes pragma unroll directives and uses a 512-thread block size. While the speedup is modest, the implementation is correct and demonstrates solid GPU programming practices. The code could benefit from more advanced optimizations like shared memory usage or better memory coalescing for higher performance gains.
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

... (truncated, 5222 chars total)

### Assistant (Turn 1)
**Tool Call: `write`**
```json
{
  "path": "optimized_maxpool.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\nmaxpool_cpp_source = \"\"\"\n#include &lt;hip/hip_runtime.h&gt;\n\nconst float NEG_INF = -3.402823466e+38f;\n\n__global__ void maxpool2d_kernel(\n    const float* input,\n    float* output,\n    const int N,\n    const int C,\n    const int Hin,\n    const int Win,\n    const int Hout,\n    const int Wout,\n    const int kh,\n    const int kw,\n    const int sh,\n    const int sw,\n    const int ph,\n    const int pw,\n    const int dh,\n    const int dw\n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx &gt;= N * C * Hout * Wout) return;\n\n    int temp = idx / (Hout * Wout);\n    int n = temp / C;\n    int c = temp % C;\n    int temp2 = idx % (Hout * Wout);\n    int oh = temp2 / Wout;\n    int ow = temp2 % Wout;\n\n    float max_val = NEG_INF;\n\n    for (int y = 0; y &lt; kh; y++) {\n        int ih = oh * sh - ph + y * dh;\n        if (ih &lt; 0 || ih &gt;= Hin) continue;\n        for (int x = 0; x &lt; kw; x++) {\n            int iw = ow * sw - pw + x * dw;\n            if (iw &lt; 0 || iw &gt;= Win) continue;\n            int in_idx = ((n * C + c) * Hin + ih) * Win + iw;\n            max_val = fmaxf(max_val, input[in_idx]);\n        }\n    }\n\n    int out_idx = ((n * C + c) * Hout + oh) * Wout + ow;\n    output[out_idx] = max_val;\n}\n\ntorch::Tensor maxpool2d_hip(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {\n    auto N = input.size(0);\n    auto C = input.size(1);\n    auto Hin = input.size(2);\n    auto Win = input.size(3);\n\n    int kh = kernel_size;\n    int kw = kernel_size;\n    int sh = stride;\n    int sw = stride;\n    int ph = padding;\n    int pw = padding;\n    int dh = dilation;\n    int dw = dilation;\n\n    int Hout = (Hin + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;\n    int Wout = (Win + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;\n\n    auto output = torch::empty({N, C, Hout, Wout}, input.options());\n\n    int total_out = N * C * Hout * Wout;\n    const int block_size = 256;\n    dim3 block(block_size);\n    dim3 grid((total_out + block_size - 1) / block_size);\n\n    maxpool2d_kernel&lt;&lt;&lt;grid, block&gt;&gt;&gt;(\n        input.data_ptr&lt;float&gt;(),\n        output.data_ptr&lt;float&gt;(),\n        N, C, Hin, Win, Hout, Wout,\n        kh, kw, sh, sw, ph, pw, dh, dw\n    );\n\n    return output;\n}\n\"\"\"\n\nmaxpool_module = load_inline(\n    name=\"maxpool2d\",\n    cpp_sources=maxpool_cpp_source,\n    functions=[\"maxpool2d_hip\"],\n    verbose=True,\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):\n        super(ModelNew, self).__init__()\n        self.maxpool_hip = maxpool_module.maxpool2d_hip\n        self.kernel_size = kernel_size\n        self.stride = stride\n        self.padding = padding\n        self.dilation = dilation\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.maxpool_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)\n\ndef get_inputs():\n    batch_size = 32\n    channels = 64\n    height = 512\n    width = 512\n    x = torch.rand(batch_size, channels, height, width)\n    return [x]\n\ndef get_init_inputs():\n    kernel_size = 4\n    stride = 1\n    padding = 1\n    dilation = 1\n    return [kernel_size, stride, padding, dilation]\n"
}
```

### Tool Result
```
Successfully wrote 3304 bytes to optimized_maxpool.py
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_maxpool.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/42_Max_Pooling_2D.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_131721
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=maxpool2d -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/maxpool2d/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
FAILED: [code=1] main.o 
hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=maxpool2d -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/maxpool2d/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
/root/.cache/torch_extensions/py312_cpu/maxpool2d/main.cpp:3:10: error: expected "FILENAME" or <FILENAME>
    3 | #include &lt;hip/hip_runtime.h&gt;
      |          ^
/root/.cache/torch_extensions/py312_cpu/maxpool2d/main.cpp

... (truncated, 11641 chars total)
```

### Assistant (Turn 3)
**Tool Call: `write`**
```json
{
  "path": "optimized_maxpool.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\nmaxpool_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\nconst float NEG_INF = -3.402823466e+38f;\n\n__global__ void maxpool2d_kernel(\n    const float* input,\n    float* output,\n    const int N,\n    const int C,\n    const int Hin,\n    const int Win,\n    const int Hout,\n    const int Wout,\n    const int kh,\n    const int kw,\n    const int sh,\n    const int sw,\n    const int ph,\n    const int pw,\n    const int dh,\n    const int dw\n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx >= N * C * Hout * Wout) return;\n\n    int houtwout = Hout * Wout;\n    int temp = idx / houtwout;\n    int n = temp / C;\n    int c = temp % C;\n    int temp2 = idx % houtwout;\n    int oh = temp2 / Wout;\n    int ow = temp2 % Wout;\n\n    float max_val = NEG_INF;\n\n    for (int y = 0; y < kh; y++) {\n        int ih = oh * sh - ph + y * dh;\n        if (ih < 0 || ih >= Hin) continue;\n        for (int x = 0; x < kw; x++) {\n            int iw = ow * sw - pw + x * dw;\n            if (iw < 0 || iw >= Win) continue;\n            int in_idx = ((n * C + c) * Hin + ih) * Win + iw;\n            max_val = fmaxf(max_val, input[in_idx]);\n        }\n    }\n\n    int out_idx = ((n * C + c) * Hout + oh) * Wout + ow;\n    output[out_idx] = max_val;\n}\n\ntorch::Tensor maxpool2d_hip(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {\n    auto N = input.size(0);\n    auto C = input.size(1);\n    auto Hin = input.size(2);\n    auto Win = input.size(3);\n\n    int kh = kernel_size;\n    int kw = kernel_size;\n    int sh = stride;\n    int sw = stride;\n    int ph = padding;\n    int pw = padding;\n    int dh = dilation;\n    int dw = dilation;\n\n    int Hout = (Hin + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;\n    int Wout = (Win + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;\n\n    if (Hout < 0) Hout = 0;\n    if (Wout < 0) Wout = 0;\n\n    auto output = torch::empty({N, C, Hout, Wout}, input.options());\n\n    int total_out = N * C * Hout * Wout;\n    const int block_size = 256;\n    dim3 block(block_size);\n    dim3 grid((total_out + block_size - 1) / block_size);\n\n    maxpool2d_kernel<<<grid, block>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        N, C, Hin, Win, Hout, Wout,\n        kh, kw, sh, sw, ph, pw, dh, dw\n    );\n\n    return output;\n}\n\"\"\"\n\nmaxpool_module = load_inline(\n    name=\"maxpool2d\",\n    cpp_sources=maxpool_cpp_source,\n    functions=[\"maxpool2d_hip\"],\n    verbose=True,\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):\n        super(ModelNew, self).__init__()\n        self.maxpool_hip = maxpool_module.maxpool2d_hip\n        self.kernel_size = kernel_size\n        self.stride = stride\n        self.padding = padding\n        self.dilation = dilation\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.maxpool_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)\n\ndef get_inputs():\n    batch_size = 32\n    channels = 64\n    height = 512\n    width = 512\n    x = torch.rand(batch_size, channels, height, width)\n    return [x]\n\ndef get_init_inputs():\n    kernel_size = 4\n    stride = 1\n    padding = 1\n    dilation = 1\n    return [kernel_size, stride, padding, dilation]\n"
}
```

### Tool Result
```
Successfully wrote 3326 bytes to optimized_maxpool.py
```

### Assistant (Turn 4)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_maxpool.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/42_Max_Pooling_2D.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_132113
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=maxpool2d -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/maxpool2d/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o maxpool2d.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 5.604ms, Ref: 6.829ms, Speedup: 1.22x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_132113/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 1.22x
============================================================

```

### Assistant (Turn 5)
**Tool Call: `write`**
```json
{
  "path": "optimized_maxpool.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\nmaxpool_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\nconst float NEG_INF = -3.402823466e+38f;\n\n__global__ void maxpool2d_kernel(\n    const float* input,\n    float* output,\n    const int N,\n    const int C,\n    const int Hin,\n    const int Win,\n    const int Hout,\n    const int Wout,\n    const int kh,\n    const int kw,\n    const int sh,\n    const int sw,\n    const int ph,\n    const int pw,\n    const int dh,\n    const int dw\n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx >= N * C * Hout * Wout) return;\n\n    int houtwout = Hout * Wout;\n    int temp = idx / houtwout;\n    int n = temp / C;\n    int c = temp % C;\n    int temp2 = idx % houtwout;\n    int oh = temp2 / Wout;\n    int ow = temp2 % Wout;\n\n    float max_val = NEG_INF;\n\n#pragma unroll\n    for (int y = 0; y < kh; y++) {\n#pragma unroll\n        for (int x = 0; x < kw; x++) {\n            int ih = oh * sh - ph + y * dh;\n            int iw = ow * sw - pw + x * dw;\n            if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {\n                int in_idx = ((n * C + c) * Hin + ih) * Win + iw;\n                max_val = fmaxf(max_val, input[in_idx]);\n            }\n        }\n    }\n\n    int out_idx = ((n * C + c) * Hout + oh) * Wout + ow;\n    output[out_idx] = max_val;\n}\n\ntorch::Tensor maxpool2d_hip(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {\n    auto N = input.size(0);\n    auto C = input.size(1);\n    auto Hin = input.size(2);\n    auto Win = input.size(3);\n\n    int kh = kernel_size;\n    int kw = kernel_size;\n    int sh = stride;\n    int sw = stride;\n    int ph = padding;\n    int pw = padding;\n    int dh = dilation;\n    int dw = dilation;\n\n    int Hout = (Hin + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;\n    int Wout = (Win + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;\n\n    if (Hout < 0) Hout = 0;\n    if (Wout < 0) Wout = 0;\n\n    auto output = torch::empty({N, C, Hout, Wout}, input.options());\n\n    size_t numel = 1LL * N * C * (size_t)Hout * Wout;\n    const int block_size = 512;\n    dim3 block(block_size);\n    dim3 grid((numel + block_size - 1) / block_size);\n\n    maxpool2d_kernel<<<grid, block>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        N, C, Hin, Win, Hout, Wout,\n        kh, kw, sh, sw, ph, pw, dh, dw\n    );\n\n    return output;\n}\n\"\"\"\n\nmaxpool_module = load_inline(\n    name=\"maxpool2d\",\n    cpp_sources=maxpool_cpp_source,\n    functions=[\"maxpool2d_hip\"],\n    verbose=True,\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):\n        super(ModelNew, self).__init__()\n        self.maxpool_hip = maxpool_module.maxpool2d_hip\n        self.kernel_size = kernel_size\n        self.stride = stride\n        self.padding = padding\n        self.dilation = dilation\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.maxpool_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)\n\ndef get_inputs():\n    batch_size = 32\n    channels = 64\n    height = 512\n    width = 512\n    x = torch.rand(batch_size, channels, height, width)\n    return [x]\n\ndef get_init_inputs():\n    kernel_size = 4\n    stride = 1\n    padding = 1\n    dilation = 1\n    return [kernel_size, stride, padding, dilation]\n"
}
```

### Tool Result
```
Successfully wrote 3363 bytes to optimized_maxpool.py
```

### Assistant (Turn 6)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_maxpool.py --reference /root/Wafer/research/KernelBench/KernelBench/level1/42_Max_Pooling_2D.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_132247
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=maxpool2d -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/maxpool2d/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
/root/.cache/torch_extensions/py312_cpu/maxpool2d/main.cpp:7:17: warning: loop not unrolled: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering [-Wpass-failed=transform-warning]
    7 | __global__ void maxpool2d_kernel(
      |                 ^
1 warning generated when compiling for gfx942.
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o maxpool2d.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[Kern

... (truncated, 1855 chars total)
```

### Assistant (Turn 7)
**DONE**

### Assistant (Turn 8)
DONE

### Assistant (Turn 9)


### Assistant (Turn 10)


## Expected Answer
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool(x)

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

