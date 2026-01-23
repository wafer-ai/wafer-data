# Trajectory: sample_0002

## Input
**level:** level2
**problem_id:** 36
**problem_path:** /root/Wafer/research/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py
**ref_arch_src:** import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
 

... (truncated, 1152 chars total)
**name:** 37_Matmul_Swish_Sum_GroupNorm
**user_prompt:** Optimize the HIP kernel for 37_Matmul_Swish_Sum_GroupNorm
**_sample_id:** sample_0002

## Score
- **judge_score:** 0.400
- **judge_score_raw:** 4.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
The agent correctly ran wafer evaluate kernelbench and achieved 100% correctness, which is excellent. However, the implementation shows a significant performance regression with only 0.16x speedup (6x slower than reference). While the code correctly fuses matmul+swish+bias operations, the naive matrix multiplication kernel without optimizations like tiling, shared memory, or leveraging optimized BLAS libraries performs much worse than PyTorch's highly optimized implementations. The code structure is clear and the fusion approach is sound, but the lack of performance optimization techniques results in a net slowdown rather than the expected speedup.
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

... (truncated, 5032 chars total)

### Assistant (Turn 1)
Looking atLooking at this model, I can see it performs:
1. Matrix multiplication (Linear layer)
2. Swish activation (sigmoid * x)
3. Bias addition
4. GroupNorm

The main optimization opportunities are:
1. Fuse matrix multiplication with Swish activation and bias to reduce kernel launches
2. Optimize GroupNorm with parallel reduction

Let me create an optimized implementation with fuse

**Tool Call: `write`**
```json
{
  "path": "optimized_model.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Fused Linear + Swish + Bias kernel\nmatmul_swish_bias_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n#define TILE_M 64\n#define TILE_N 64\n#define TILE_K 64\n\n__global__ void matmul_swish_bias_kernel(\n    const float* A,  // [batch_size, in_features]\n    const float* B,  // [out_features, in_features] (weight, transposed)\n    const float* bias,  // [out_features]\n    float* C,  // [batch_size, out_features]\n    int batch_size,\n    int in_features,\n    int out_features,\n    bool use_bias\n) {\n    int c = blockIdx.x * TILE_N + threadIdx.x;\n    int b = blockIdx.y * TILE_M + threadIdx.y;\n    \n    __shared__ float A_tile_shared[TILE_M][TILE_K];\n    __shared__ float B_tile_shared[TILE_N][TILE_K];\n    \n    float sum = 0.0f;\n    \n    int num_tiles = (in_features + TILE_K - 1) / TILE_K;\n    \n    for (int t = 0; t < num_tiles; ++t) {\n        int k_start = t * TILE_K;\n        int k_end = min(k_start + TILE_K, in_features);\n        int k_range = k_end - k_start;\n        \n        // Load A tile\n        int a_row = threadIdx.y;\n        int a_col_start = k_start + threadIdx.x;\n        \n        if (b < batch_size && a_col_start < in_features) {\n            A_tile_shared[a_row][threadIdx.x] = A[b * in_features + a_col_start];\n        }\n        \n        // Load B tile\n        int b_row = threadIdx.x;\n        int b_col_start = k_start + threadIdx.y;\n        \n        if (c < out_features && b_col_start < in_features) {\n            B_tile_shared[b_row][threadIdx.y] = B[c * in_features + b_col_start];\n        }\n        \n        __syncthreads();\n        \n        #pragma unroll\n        for (int k = 0; k < TILE_K && (k_start + k) < k_end; ++k) {\n            float a_val = A_tile_shared[threadIdx.y][k];\n            float b_val = B_tile_shared[threadIdx.x][k];\n            sum += a_val * b_val;\n        }\n        \n        __syncthreads();\n    }\n    \n    if (b < batch_size && c < out_features) {\n        float sigmoid_x = 1.0f / (1.0f + expf(-sum));\n        float swish_out = sigmoid_x * sum;\n        \n        if (use_bias) {\n            swish_out += bias[c];\n        }\n        \n        C[b * out_features + c] = swish_out;\n    }\n}\n\ntorch::Tensor matmul_swish_bias_hip(torch::Tensor A, torch::Tensor B, torch::Tensor bias, bool use_bias) {\n    int batch_size = A.size(0);\n    int in_features = A.size(1);\n    int out_features = B.size(0);\n    \n    auto C = torch::empty({batch_size, out_features}, A.options());\n    \n    dim3 block(TILE_N, TILE_M);\n    dim3 grid((out_features + TILE_N - 1) / TILE_N, (batch_size + TILE_M - 1) / TILE_M);\n    \n    matmul_swish_bias_kernel<<<grid, block>>>(\n        A.data_ptr<float>(),\n        B.data_ptr<float>(),\n        use_bias ? bias.data_ptr<float>() : nullptr,\n        C.data_ptr<float>(),\n        batch_size,\n        in_features,\n        out_features,\n        use_bias\n    );\n    \n    return C;\n}\n\"\"\"\n\nmatmul_swish_bias = load_inline(\n    name=\"matmul_swish_bias\",\n    cpp_sources=matmul_swish_bias_cpp_source,\n    functions=[\"matmul_swish_bias_hip\"],\n    verbose=True,\n)\n\n# Fused Swish + Bias kernel\nswish_bias_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n__global__ void swish_bias_kernel(\n    const float* x,\n    const float* bias,\n    float* out,\n    int batch_size,\n    int out_features,\n    bool use_bias\n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    int total_elements = batch_size * out_features;\n    \n    if (idx < total_elements) {\n        float x_val = x[idx];\n        float sigmoid_x = 1.0f / (1.0f + expf(-x_val));\n        float swish_out = sigmoid_x * x_val;\n        \n        if (use_bias) {\n            int feature_idx = idx % out_features;\n            swish_out += bias[feature_idx];\n        }\n        \n        out[idx] = swish_out;\n    }\n}\n\ntorch::Tensor swish_bias_hip(torch::Tensor x, torch::Tensor bias, bool use_bias) {\n    int batch_size = x.size(0);\n    int out_features = x.size(1);\n    auto out = torch::empty_like(x);\n    \n    int total_elements = batch_size * out_features;\n    const int block_size = 256;\n    const int num_blocks = (total_elements + block_size - 1) / block_size;\n    \n    swish_bias_kernel<<<num_blocks, block_size>>>(\n        x.data_ptr<float>(),\n        use_bias ? bias.data_ptr<float>() : nullptr,\n        out.data_ptr<float>(),\n        batch_size,\n        out_features,\n        use_bias\n    );\n    \n    return out;\n}\n\"\"\"\n\nswish_bias = load_inline(\n    name=\"swish_bias\",\n    cpp_sources=swish_bias_cpp_source,\n    functions=[\"swish_bias_hip\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    \"\"\"\n    Optimized model with fused kernels:\n    - Uses fused matmul + Swish + bias kernel\n    - Uses fused Swish + bias kernel as alternative\n    - Standard GroupNorm (already optimized in PyTorch)\n    \"\"\"\n    def __init__(self, in_features, out_features, num_groups, bias_shape):\n        super(ModelNew, self).__init__()\n        \n        # Store parameters manually\n        self.in_features = in_features\n        self.out_features = out_features\n        self.num_groups = num_groups\n        \n        # Custom kernels\n        self.matmul_swish_bias = matmul_swish_bias\n        self.swish_bias = swish_bias\n        \n        # Initialize weight and bias using same scheme as nn.Linear\n        self.weight = nn.Parameter(torch.randn(out_features, in_features))\n        self.bias = nn.Parameter(torch.randn(bias_shape))\n        \n        # GroupNorm\n        self.group_norm = nn.GroupNorm(num_groups, out_features)\n        \n        # Initialize weights\n        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))\n        if self.bias is not None:\n            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)\n            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0\n            nn.init.uniform_(self.bias, -bound, bound)\n    \n    def forward(self, x):\n        # Apply fused matmul + swish + bias\n        x = self.matmul_swish_bias.matmul_swish_bias_hip(x, self.weight, self.bias, True)\n        # Apply GroupNorm\n        x = self.group_norm(x)\n        return x\n\n# Add math import at top\nimport math\n\ndef get_inputs():\n    return [torch.rand(32768, 1024)]\n\ndef get_init_inputs():\n    return [1024, 4096, 64, (4096,)]"
}
```

### Tool Result
```
Successfully wrote 6258 bytes to optimized_model.py
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_053432
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=matmul_swish_bias -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/matmul_swish_bias/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
/root/.cache/torch_extensions/py312_cpu/matmul_swish_bias/main.cpp:9:17: warning: loop not unrolled: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering [-Wpass-failed=transform-warning]
    9 | __global__ void matmul_swish_bias_kernel(
      |                 ^
1 warning generated when compiling for gfx942.
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o matmul_swish_bias.so
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=swish_bias -DTORCH_API_INCLUDE

... (truncated, 7044 chars total)
```

### Assistant (Turn 3)
I seeI see compilation warnings and then a runtime error. The error seems to be in the GroupNorm call, possibly because my custom kernel isn't handling device placement properly. Let me fix the kernel with better device

**Tool Call: `write`**
```json
{
  "path": "optimized_model.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport math\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Simple fused kernel without complex tiling to avoid configuration errors\nfused_matmul_swish_bias_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n__global__ void fused_matmul_swish_bias_kernel(\n    const float* __restrict__ A,\n    const float* __restrict__ B,\n    const float* __restrict__ bias,\n    float* __restrict__ C,\n    int batch_size,\n    int M,\n    int N,\n    int K,\n    bool use_bias\n) {\n    int row = blockIdx.x * blockDim.y + threadIdx.y;  // batch dimension\n    int col = blockIdx.y * blockDim.x + threadIdx.x;  // output feature dimension\n    \n    if (row < M && col < N) {\n        float sum = 0.0f;\n        \n        // Matrix multiplication\n        for (int k = 0; k < K; ++k) {\n            sum += A[row * K + k] * B[col * K + k];\n        }\n        \n        // Swish activation\n        float sigmoid_x = 1.0f / (1.0f + expf(-sum));\n        float swish_out = sigmoid_x * sum;\n        \n        // Bias addition\n        if (use_bias) {\n            swish_out += bias[col];\n        }\n        \n        C[row * N + col] = swish_out;\n    }\n}\n\ntorch::Tensor fused_matmul_swish_bias_hip(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {\n    int M = A.size(0);  // batch_size\n    int K = A.size(1);  // in_features\n    int N = B.size(0);  // out_features\n    \n    // Ensure output is contiguous and on same device as A\n    auto C = torch::empty({M, N}, A.options());\n    \n    // Fixed block size\n    const int blockDim_x = 16;\n    const int blockDim_y = 16;\n    \n    dim3 blockDim(blockDim_x, blockDim_y);\n    dim3 gridDim((N + blockDim_x - 1) / blockDim_x, (M + blockDim_y - 1) / blockDim_y);\n    \n    bool use_bias = bias.numel() > 0;\n    \n    fused_matmul_swish_bias_kernel<<<gridDim, blockDim>>>(\n        A.data_ptr<float>(),\n        B.data_ptr<float>(),\n        use_bias ? bias.data_ptr<float>() : nullptr,\n        C.data_ptr<float>(),\n        M,\n        M,\n        N,\n        K,\n        use_bias\n    );\n    \n    return C;\n}\n\"\"\"\n\nfused_matmul_swish_bias = load_inline(\n    name=\"fused_matmul_swish_bias\",\n    cpp_sources=fused_matmul_swish_bias_cpp_source,\n    functions=[\"fused_matmul_swish_bias_hip\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    \"\"\"\n    Optimized model with fused matmul + Swish + bias kernel.\n    \"\"\"\n    def __init__(self, in_features, out_features, num_groups, bias_shape):\n        super(ModelNew, self).__init__()\n        \n        # Store parameters manually\n        self.in_features = in_features\n        self.out_features = out_features\n        \n        # Custom fused kernel\n        self.fused_matmul_swish_bias = fused_matmul_swish_bias\n        \n        # Initialize weight and bias\n        self.weight = nn.Parameter(torch.randn(out_features, in_features))\n        self.bias = nn.Parameter(torch.randn(bias_shape))\n        \n        # GroupNorm\n        self.group_norm = nn.GroupNorm(num_groups, out_features)\n        \n        # Initialize weights using same scheme as nn.Linear\n        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))\n        if self.bias is not None:\n            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)\n            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0\n            nn.init.uniform_(self.bias, -bound, bound)\n    \n    def forward(self, x):\n        # Move input to device if needed\n        device = self.weight.device\n        x = x.to(device)\n        \n        # Apply fused matmul + swish + bias\n        x = self.fused_matmul_swish_bias.fused_matmul_swish_bias_hip(x, self.weight, self.bias)\n        \n        # Apply GroupNorm\n        x = self.group_norm(x)\n        \n        return x\n\n\ndef get_inputs():\n    return [torch.rand(32768, 1024)]\n\ndef get_init_inputs():\n    return [1024, 4096, 64, (4096,)]"
}
```

### Tool Result
```
Successfully wrote 3847 bytes to optimized_model.py
```

### Assistant (Turn 4)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_053953
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_matmul_swish_bias -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_matmul_swish_bias/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_matmul_swish_bias.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Diff analysis:
   Max diff: 7.392795 at index [436,742]
   Mean diff: 0.140661
   Max relative error: 3760418000.00%, Mean: 162.21%
   Wrong elements: 16,762,404 / 134,217,728 (12.49%)
   Error distribution: <1e-6: 117,440,522 | 1e-6~1e-4: 1,012 | 1e-4~1e-2: 94,089 | 1e-2~1: 8,711,255 | >1: 7,970,850
   Pattern: PARTIAL failures - check specific conditions
   Sample failures:
      [0,0

... (truncated, 2090 chars total)
```

### Assistant (Turn 5)
Good, theGood, the kernel compiles and runs but the result is incorrect. Looking at the sample failures, I realize I need to handle the case where bias can be None (in some cases). Also, I need to make sure I'm using the correct device placement for bias. Let me fix the

**Tool Call: `write`**
```json
{
  "path": "optimized_model.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\nimport math\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Simple fused kernel without complex tiling\nfused_matmul_swish_bias_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n__global__ void fused_matmul_swish_bias_kernel(\n    const float* __restrict__ A,\n    const float* __restrict__ B,\n    const float* __restrict__ bias_ptr,\n    float* __restrict__ C,\n    int M,\n    int N,\n    int K,\n    bool use_bias\n) {\n    int row = blockIdx.x * blockDim.y + threadIdx.y;\n    int col = blockIdx.y * blockDim.x + threadIdx.x;\n    \n    if (row < M && col < N) {\n        float sum = 0.0f;\n        \n        // Matrix multiplication: compute C[row,col] = sum_k A[row,k] * B[col,k]\n        for (int k = 0; k < K; ++k) {\n            sum += A[row * K + k] * B[col * K + k];\n        }\n        \n        // Swish activation\n        float sigmoid_x = 1.0f / (1.0f + expf(-sum));\n        float swish_out = sigmoid_x * sum;\n        \n        // Bias addition\n        if (use_bias && bias_ptr != nullptr) {\n            swish_out += bias_ptr[col];\n        }\n        \n        C[row * N + col] = swish_out;\n    }\n}\n\ntorch::Tensor fused_matmul_swish_bias_hip(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> bias_opt) {\n    int M = A.size(0);\n    int K = A.size(1);\n    int N = B.size(0);\n    \n    auto C = torch::empty({M, N}, A.options());\n    \n    const int blockDim_x = 16;\n    const int blockDim_y = 16;\n    \n    dim3 blockDim(blockDim_x, blockDim_y);\n    dim3 gridDim((N + blockDim_x - 1) / blockDim_x, (M + blockDim_y - 1) / blockDim_y);\n    \n    bool use_bias = bias_opt.has_value() && bias_opt.value().numel() > 0;\n    const float* bias_ptr = use_bias ? bias_opt.value().data_ptr<float>() : nullptr;\n    \n    fused_matmul_swish_bias_kernel<<<gridDim, blockDim>>>(\n        A.data_ptr<float>(),\n        B.data_ptr<float>(),\n        bias_ptr,\n        C.data_ptr<float>(),\n        M,\n        N,\n        K,\n        use_bias\n    );\n    \n    return C;\n}\n\"\"\"\n\nfused_matmul_swish_bias = load_inline(\n    name=\"fused_matmul_swish_bias\",\n    cpp_sources=fused_matmul_swish_bias_cpp_source,\n    functions=[\"fused_matmul_swish_bias_hip\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    \"\"\"\n    Optimized model with fused matmul + Swish + bias kernel.\n    \"\"\"\n    def __init__(self, in_features, out_features, num_groups, bias_shape):\n        super(ModelNew, self).__init__()\n        \n        self.in_features = in_features\n        self.out_features = out_features\n        \n        # Custom fused kernel\n        self.fused_matmul_swish_bias = fused_matmul_swish_bias\n        \n        # Initialize weight and bias\n        self.weight = nn.Parameter(torch.randn(out_features, in_features))\n        \n        # Bias might be None - handle both cases\n        if bias_shape is not None and len(bias_shape) > 0:\n            self.bias = nn.Parameter(torch.randn(bias_shape))\n        else:\n            self.register_parameter('bias', None)\n        \n        # GroupNorm\n        self.group_norm = nn.GroupNorm(num_groups, out_features)\n        \n        # Initialize weights using same scheme as nn.Linear\n        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))\n        if self.bias is not None:\n            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)\n            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0\n            nn.init.uniform_(self.bias, -bound, bound)\n    \n    def forward(self, x):\n        device = self.weight.device\n        x = x.to(device)\n        \n        # Apply fused matmul + swish + bias\n        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=device)\n        x = self.fused_matmul_swish_bias.fused_matmul_swish_bias_hip(x, self.weight, bias_tensor)\n        \n        # Apply GroupNorm\n        x = self.group_norm(x)\n        \n        return x\n\n\ndef get_inputs():\n    return [torch.rand(32768, 1024)]\n\ndef get_init_inputs():\n    return [1024, 4096, 64, (4096,)]"
}
```

### Tool Result
```
Successfully wrote 3992 bytes to optimized_model.py
```

### Assistant (Turn 6)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_054142
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_matmul_swish_bias -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_matmul_swish_bias/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_matmul_swish_bias.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Diff analysis:
   Max diff: 7.392795 at index [436,742]
   Mean diff: 0.140661
   Max relative error: 3760418000.00%, Mean: 162.21%
   Wrong elements: 16,762,404 / 134,217,728 (12.49%)
   Error distribution: <1e-6: 117,440,522 | 1e-6~1e-4: 1,012 | 1e-4~1e-2: 94,089 | 1e-2~1: 8,711,255 | >1: 7,970,850
   Pattern: PARTIAL failures - check specific conditions
   Sample failures:
      [0,0

... (truncated, 2090 chars total)
```

### Assistant (Turn 7)
I see theI see the issue - I believe the problem is random initialization. The reference model and my model use different random initialization, so I need to ensure the same seeds are used. Let me also add device guards and check the implementation more 

**Tool Call: `bash`**
```json
{
  "command": "cat /root/Wafer/research/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py"
}
```

### Tool Result
```
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = x + self.bias
        x = self.group_norm(x)
        return x

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
```

### Assistant (Turn 8)
I see theI see the issue now! The problem is weight initialization. The reference model uses `nn.Linear` which has its own weight, but I'm creating new randomly initialized weights. The evaluation framework likely copies or shares parameters, but my manual initialization breaks this.

Let me fix this by using `nn.Linear` for parameter management but calling my custom kernel i

**Tool Call: `write`**
```json
{
  "path": "optimized_model.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Simple fused kernel for matmul + Swish + bias\nfused_matmul_swish_bias_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n__global__ void fused_matmul_swish_bias_kernel(\n    const float* __restrict__ A,\n    const float* __restrict__ B,\n    const float* __restrict__ bias_ptr,\n    float* __restrict__ C,\n    int M,\n    int N,\n    int K,\n    bool use_bias\n) {\n    int row = blockIdx.x * blockDim.y + threadIdx.y;\n    int col = blockIdx.y * blockDim.x + threadIdx.x;\n    \n    if (row < M && col < N) {\n        float sum = 0.0f;\n        \n        // Matrix multiplication: C[row,col] = sum_k A[row,k] * B[col,k]\n        for (int k = 0; k < K; ++k) {\n            sum += A[row * K + k] * B[col * K + k];\n        }\n        \n        // Swish activation\n        float sigmoid_x = 1.0f / (1.0f + expf(-sum));\n        float swish_out = sigmoid_x * sum;\n        \n        // Bias addition\n        if (use_bias && bias_ptr != nullptr) {\n            swish_out += bias_ptr[col];\n        }\n        \n        C[row * N + col] = swish_out;\n    }\n}\n\ntorch::Tensor fused_matmul_swish_bias_hip(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> bias_opt) {\n    int M = A.size(0);\n    int K = A.size(1);\n    int N = B.size(0);\n    \n    auto C = torch::empty({M, N}, A.options());\n    \n    const int blockDim_x = 16;\n    const int blockDim_y = 16;\n    \n    dim3 blockDim(blockDim_x, blockDim_y);\n    dim3 gridDim((N + blockDim_x - 1) / blockDim_x, (M + blockDim_y - 1) / blockDim_y);\n    \n    bool use_bias = bias_opt.has_value() && bias_opt.value().numel() > 0;\n    const float* bias_ptr = use_bias ? bias_opt.value().data_ptr<float>() : nullptr;\n    \n    fused_matmul_swish_bias_kernel<<<gridDim, blockDim>>>(\n        A.data_ptr<float>(),\n        B.data_ptr<float>(),\n        bias_ptr,\n        C.data_ptr<float>(),\n        M,\n        N,\n        K,\n        use_bias\n    );\n    \n    return C;\n}\n\"\"\"\n\nfused_matmul_swish_bias = load_inline(\n    name=\"fused_matmul_swish_bias\",\n    cpp_sources=fused_matmul_swish_bias_cpp_source,\n    functions=[\"fused_matmul_swish_bias_hip\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    \"\"\"\n    Optimized model with fused matmul + Swish + bias kernel.\n    Uses nn.Linear for parameter management but applies custom kernel in forward.\n    \"\"\"\n    def __init__(self, in_features, out_features, num_groups, bias_shape):\n        super(ModelNew, self).__init__()\n        \n        # Use nn.Linear for proper weight initialization and parameter sharing\n        self.matmul = nn.Linear(in_features, out_features)\n        \n        # Custom fused kernel\n        self.fused_matmul_swish_bias = fused_matmul_swish_bias\n        \n        # Additional bias parameter (as in reference model)\n        self.bias_param = nn.Parameter(torch.randn(bias_shape))\n        \n        # GroupNorm\n        self.group_norm = nn.GroupNorm(num_groups, out_features)\n    \n    def forward(self, x):\n        # Apply fused matmul + swish using linear layer's weight and bias\n        # Combine linear layer's bias with external bias\n        linear_weight = self.matmul.weight\n        linear_bias = self.matmul.bias if self.matmul.bias is not None else torch.tensor(0)\n        external_bias = self.linear_bias if self.matmul.bias is not None else self.bias_param\n        \n        x = self.fused_matmul_swish_bias.fused_matmul_swish_bias_hip(x, linear_weight, external_bias)\n        \n        # Add external bias\n        x = x + self.bias_param\n        \n        # Apply GroupNorm\n        x = self.group_norm(x)\n        \n        return x\n\n\ndef get_inputs():\n    return [torch.rand(32768, 1024)]\n\ndef get_init_inputs():\n    return [1024, 4096, 64, (4096,)]"
}
```

### Tool Result
```
Successfully wrote 3762 bytes to optimized_model.py
```

### Assistant (Turn 9)
**Tool Call: `write`**
```json
{
  "path": "optimized_model.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Simple fused kernel for matmul + Swish + bias\nfused_matmul_swish_bias_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n__global__ void fused_matmul_swish_bias_kernel(\n    const float* __restrict__ A,\n    const float* __restrict__ B,\n    const float* __restrict__ bias_ptr,\n    float* __restrict__ C,\n    int M,\n    int N,\n    int K,\n    bool use_bias\n) {\n    int row = blockIdx.x * blockDim.y + threadIdx.y;\n    int col = blockIdx.y * blockDim.x + threadIdx.x;\n    \n    if (row < M && col < N) {\n        float sum = 0.0f;\n        \n        // Matrix multiplication: C[row,col] = sum_k A[row,k] * B[col,k]\n        for (int k = 0; k < K; ++k) {\n            sum += A[row * K + k] * B[col * K + k];\n        }\n        \n        // Swish activation\n        float sigmoid_x = 1.0f / (1.0f + expf(-sum));\n        float swish_out = sigmoid_x * sum;\n        \n        // Bias addition\n        if (use_bias && bias_ptr != nullptr) {\n            swish_out += bias_ptr[col];\n        }\n        \n        C[row * N + col] = swish_out;\n    }\n}\n\ntorch::Tensor fused_matmul_swish_bias_hip(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> bias_opt) {\n    int M = A.size(0);\n    int K = A.size(1);\n    int N = B.size(0);\n    \n    auto C = torch::empty({M, N}, A.options());\n    \n    const int blockDim_x = 16;\n    const int blockDim_y = 16;\n    \n    dim3 blockDim(blockDim_x, blockDim_y);\n    dim3 gridDim((N + blockDim_x - 1) / blockDim_x, (M + blockDim_y - 1) / blockDim_y);\n    \n    bool use_bias = bias_opt.has_value() && bias_opt.value().numel() > 0;\n    const float* bias_ptr = use_bias ? bias_opt.value().data_ptr<float>() : nullptr;\n    \n    fused_matmul_swish_bias_kernel<<<gridDim, blockDim>>>(\n        A.data_ptr<float>(),\n        B.data_ptr<float>(),\n        bias_ptr,\n        C.data_ptr<float>(),\n        M,\n        N,\n        K,\n        use_bias\n    );\n    \n    return C;\n}\n\"\"\"\n\nfused_matmul_swish_bias = load_inline(\n    name=\"fused_matmul_swish_bias\",\n    cpp_sources=fused_matmul_swish_bias_cpp_source,\n    functions=[\"fused_matmul_swish_bias_hip\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    \"\"\"\n    Optimized model with fused matmul + Swish + bias kernel.\n    Uses the same parameters as reference but applies custom kernel for speed.\n    \"\"\"\n    def __init__(self, in_features, out_features, num_groups, bias_shape):\n        super(ModelNew, self).__init__()\n        \n        # Use nn.Linear for weight storage (ignore its bias)\n        self.matmul = nn.Linear(in_features, out_features)\n        \n        # Custom fused kernel\n        self.fused_matmul_swish_bias = fused_matmul_swish_bias\n        \n        # External bias parameter (same as reference)\n        self.bias = nn.Parameter(torch.randn(bias_shape))\n        \n        # GroupNorm\n        self.group_norm = nn.GroupNorm(num_groups, out_features)\n    \n    def forward(self, x):\n        # Use linear layer's weight (shape: out_features x in_features)\n        weight = self.matmul.weight\n        \n        # Apply fused matmul + swish with external bias\n        x = self.fused_matmul_swish_bias.fused_matmul_swish_bias_hip(x, weight, self.bias)\n        \n        # Apply GroupNorm\n        x = self.group_norm(x)\n        \n        return x\n\n\ndef get_inputs():\n    return [torch.rand(32768, 1024)]\n\ndef get_init_inputs():\n    return [1024, 4096, 64, (4096,)]"
}
```

### Tool Result
```
Successfully wrote 3453 bytes to optimized_model.py
```

### Assistant (Turn 10)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_055130
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_matmul_swish_bias -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_matmul_swish_bias/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_matmul_swish_bias.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Diff analysis:
   Max diff: 0.033938 at index [2528,3170]
   Mean diff: 0.000956
   Max relative error: 12455835.94%, Mean: 1.46%
   Wrong elements: 14,537,027 / 134,217,728 (10.83%)
   Error distribution: <1e-6: 117,441,690 | 1e-6~1e-4: 119,649 | 1e-4~1e-2: 11,472,235 | 1e-2~1: 5,184,154 | >1: 0
   Pattern: PARTIAL failures - check specific conditions
   Sample failures:
      [0,2]: r

... (truncated, 2092 chars total)
```

### Assistant (Turn 11)
The errorThe errorThe error is small (max diff 0.033), suggesting a numerical precision issue. Looking at the reference more carefully, I see that nn.Linear has its own built-in bias. Let me trace through the actual operations:

1. `self.matmul(x)` computes `x @ W^T + b_linear`
2. Swish is applied
3. `x + self.bias` adds external bias

My kernel was missing the internal Linear bias! Let me

**Tool Call: `write`**
```json
{
  "path": "optimized_model.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Simple fused kernel for matmul + Swish + bias\nfused_matmul_swish_bias_cpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n__global__ void fused_matmul_swish_bias_kernel(\n    const float* __restrict__ A,\n    const float* __restrict__ B,\n    const float* __restrict__ bias_linear_ptr,\n    const float* __restrict__ bias_external_ptr,\n    float* __restrict__ C,\n    int M,\n    int N,\n    int K,\n    bool use_bias_linear,\n    bool use_bias_external\n) {\n    int row = blockIdx.x * blockDim.y + threadIdx.y;\n    int col = blockIdx.y * blockDim.x + threadIdx.x;\n    \n    if (row < M && col < N) {\n        float sum = 0.0f;\n        \n        // Matrix multiplication: C[row,col] = sum_k A[row,k] * B[col,k]\n        for (int k = 0; k < K; ++k) {\n            sum += A[row * K + k] * B[col * K + k];\n        }\n        \n        // Add linear layer bias (before Swish)\n        if (use_bias_linear && bias_linear_ptr != nullptr) {\n            sum += bias_linear_ptr[col];\n        }\n        \n        // Swish activation\n        float sigmoid_x = 1.0f / (1.0f + expf(-sum));\n        float swish_out = sigmoid_x * sum;\n        \n        // Add external bias (after Swish)\n        if (use_bias_external && bias_external_ptr != nullptr) {\n            swish_out += bias_external_ptr[col];\n        }\n        \n        C[row * N + col] = swish_out;\n    }\n}\n\ntorch::Tensor fused_matmul_swish_bias_hip(\n    torch::Tensor A,\n    torch::Tensor B,\n    c10::optional<torch::Tensor> bias_linear_opt,\n    c10::optional<torch::Tensor> bias_external_opt\n) {\n    int M = A.size(0);\n    int K = A.size(1);\n    int N = B.size(0);\n    \n    auto C = torch::empty({M, N}, A.options());\n    \n    const int blockDim_x = 16;\n    const int blockDim_y = 16;\n    \n    dim3 blockDim(blockDim_x, blockDim_y);\n    dim3 gridDim((N + blockDim_x - 1) / blockDim_x, (M + blockDim_y - 1) / blockDim_y);\n    \n    bool use_bias_linear = bias_linear_opt.has_value() && bias_linear_opt.value().numel() > 0;\n    bool use_bias_external = bias_external_opt.has_value() && bias_external_opt.value().numel() > 0;\n    \n    const float* bias_linear_ptr = use_bias_linear ? bias_linear_opt.value().data_ptr<float>() : nullptr;\n    const float* bias_external_ptr = use_bias_external ? bias_external_opt.value().data_ptr<float>() : nullptr;\n    \n    fused_matmul_swish_bias_kernel<<<gridDim, blockDim>>>(\n        A.data_ptr<float>(),\n        B.data_ptr<float>(),\n        bias_linear_ptr,\n        bias_external_ptr,\n        C.data_ptr<float>(),\n        M,\n        N,\n        K,\n        use_bias_linear,\n        use_bias_external\n    );\n    \n    return C;\n}\n\"\"\"\n\nfused_matmul_swish_bias = load_inline(\n    name=\"fused_matmul_swish_bias\",\n    cpp_sources=fused_matmul_swish_bias_cpp_source,\n    functions=[\"fused_matmul_swish_bias_hip\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    \"\"\"\n    Optimized model with fused matmul + Swish + bias kernel.\n    Replicates the exact computation of the reference model.\n    \"\"\"\n    def __init__(self, in_features, out_features, num_groups, bias_shape):\n        super(ModelNew, self).__init__()\n        \n        # nn.Linear has its own weight and bias\n        self.matmul = nn.Linear(in_features, out_features)\n        \n        # Custom fused kernel\n        self.fused_matmul_swish_bias = fused_matmul_swish_bias\n        \n        # External bias parameter (same as reference)\n        self.bias = nn.Parameter(torch.randn(bias_shape))\n        \n        # GroupNorm\n        self.group_norm = nn.GroupNorm(num_groups, out_features)\n    \n    def forward(self, x):\n        # Get linear layer weight\n        weight = self.matmul.weight\n        bias_linear = self.matmul.bias  # Can be None if bias=False\n        \n        # Apply fused matmul + add linear bias + swish + add external bias\n        x = self.fused_matmul_swish_bias.fused_matmul_swish_bias_hip(x, weight, bias_linear, self.bias)\n        \n        # Apply GroupNorm\n        x = self.group_norm(x)\n        \n        return x\n\n\ndef get_inputs():\n    return [torch.rand(32768, 1024)]\n\ndef get_init_inputs():\n    return [1024, 4096, 64, (4096,)]"
}
```

### Tool Result
```
Successfully wrote 4190 bytes to optimized_model.py
```

### Assistant (Turn 12)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_model.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_055222
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_matmul_swish_bias -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_matmul_swish_bias/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_matmul_swish_bias.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 30.912ms, Ref: 5.001ms, Speedup: 0.16x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_055222/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.16x
============================================================

```

## Expected Answer
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = x + self.bias
        x = self.group_norm(x)
        return x

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
