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
- **judge_score:** 0.300
- **judge_score_raw:** 3.000
- **has_response:** 1.000

### judge_score - Details
**reasoning:**
```
The agent correctly ran wafer evaluate kernelbench and achieved 100% correctness, which is essential. However, the kernel shows poor performance with only 0.01x speedup (actually 100x slower than reference). While the code implements a reasonable fusion strategy combining matmul+maxpool+sum+scale operations and uses vectorization with float4, the kernel design has significant performance issues. The approach of using one block per batch element with 256 threads per block may not be optimal for the large matrix dimensions (32768x32768), leading to poor GPU utilization and memory access patterns. The kernel passes correctness but fails to deliver any performance improvement.
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
I'll optimize this architecture by fusing the matmul with the subsequent operations (maxpool, sum, scale) into a single custom HIP kernel. This will significantly reduce memory traffic by avoiding writing intermediate results to memory.

Let me create the optimized kernel:

**Tool Call: `write`**
```json
{
  "path": "optimized_kernel.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Fused kernel: matmul + maxpool + sum + scale\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n\n#define BLOCK_SIZE 256\n\n// Fused kernel: matmul + maxpool + sum + scale\n// Performs: output = scale * sum(maxpool(matmul(x, W^T)))\n__global__ void fused_matmul_maxpool_sum_scale_kernel(\n    const float* __restrict__ x,      // input: (batch_size, in_features)\n    const float* __restrict__ W,      // weight: (out_features, in_features)\n    const float* __restrict__ bias,   // bias: (out_features,)\n    float* __restrict__ output,       // output: (batch_size,)\n    const int batch_size,\n    const int in_features,\n    const int out_features,\n    const int kernel_size,            // maxpool kernel size (2)\n    const float scale_factor\n) {\n    int batch_idx = blockIdx.x;\n    int tid = threadIdx.x;\n    \n    if (batch_idx >= batch_size) return;\n    \n    // Each thread block processes one batch element\n    // and computes the sum of maxpooled outputs\n    __shared__ float shared_sum[BLOCK_SIZE];\n    \n    float thread_sum = 0.0f;\n    \n    // Process output_features in chunks\n    // Each thread processes multiple output elements\n    for (int out_idx = tid; out_idx < out_features; out_idx += BLOCK_SIZE) {\n        // Compute maxpool over [out_idx*kernel_size, (out_idx+1)*kernel_size)\n        // Since kernel_size is small (2), we can unroll this\n        float max_val = -INFINITY;\n        \n        for (int k = 0; k < kernel_size; k++) {\n            int full_out_idx = out_idx * kernel_size + k;\n            if (full_out_idx >= out_features) continue;\n            \n            // Compute dot product: x[batch_idx, :] @ W[full_out_idx, :]\n            float dot = bias ? bias[full_out_idx] : 0.0f;\n            \n            // Vectorized dot product - process 4 elements at a time\n            int i = 0;\n            for (; i < in_features - 3; i += 4) {\n                float4 x_vec = *reinterpret_cast<const float4*>(&x[batch_idx * in_features + i]);\n                float4 w_vec = *reinterpret_cast<const float4*>(&W[full_out_idx * in_features + i]);\n                \n                dot += x_vec.x * w_vec.x;\n                dot += x_vec.y * w_vec.y;\n                dot += x_vec.z * w_vec.z;\n                dot += x_vec.w * w_vec.w;\n            }\n            \n            // Handle remaining elements\n            for (; i < in_features; i++) {\n                dot += x[batch_idx * in_features + i] * W[full_out_idx * in_features + i];\n            }\n            \n            // Apply maxpool\n            max_val = fmaxf(max_val, dot);\n        }\n        \n        // Accumulate to thread sum\n        thread_sum += max_val;\n    }\n    \n    // Reduce within block using shared memory\n    shared_sum[tid] = thread_sum;\n    __syncthreads();\n    \n    // Parallel reduction in shared memory\n    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            shared_sum[tid] += shared_sum[tid + s];\n        }\n        __syncthreads();\n    }\n    \n    // Write final result for this batch element\n    if (tid == 0) {\n        output[batch_idx] = shared_sum[0] * scale_factor;\n    }\n}\n\ntorch::Tensor fused_matmul_maxpool_sum_scale(\n    torch::Tensor x,\n    torch::Tensor W,\n    torch::Tensor bias,\n    const float scale_factor,\n    const int kernel_size\n) {\n    const int batch_size = x.size(0);\n    const int in_features = x.size(1);\n    const int out_features = W.size(0);\n    \n    // Allocate output tensor of shape (batch_size,)\n    auto output = torch::zeros({batch_size}, x.options());\n    \n    const dim3 blocks(batch_size);\n    const dim3 threads(BLOCK_SIZE);\n    \n    fused_matmul_maxpool_sum_scale_kernel<<<blocks, threads>>>(\n        x.data_ptr<float>(),\n        W.data_ptr<float>(),\n        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,\n        output.data_ptr<float>(),\n        batch_size,\n        in_features,\n        out_features,\n        kernel_size,\n        scale_factor\n    );\n    \n    return output;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_matmul_maxpool_sum_scale\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_matmul_maxpool_sum_scale\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features)\n        self.kernel_size = kernel_size\n        self.scale_factor = scale_factor\n        self.fused_op = fused_op\n        \n        # Initialize with same distribution as reference\n        nn.init.uniform_(self.linear.weight, -0.1, 0.1)\n        nn.init.uniform_(self.linear.bias, -0.1, 0.1)\n    \n    def forward(self, x):\n        # Use fused kernel that combines matmul, maxpool, sum, and scale\n        return self.fused_op.fused_matmul_maxpool_sum_scale(\n            x, \n            self.linear.weight, \n            self.linear.bias,\n            self.scale_factor,\n            self.kernel_size\n        )\n\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features)]\n\n\ndef get_init_inputs():\n    return [in_features, out_features, kernel_size, scale_factor]\n"
}
```

### Tool Result
```
Successfully wrote 5223 bytes to optimized_kernel.py
```

### Assistant (Turn 2)
Now let's test the optimized kernel:

**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_164643
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_matmul_maxpool_sum_scale -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_matmul_maxpool_sum_scale/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_matmul_maxpool_sum_scale.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Diff analysis:
   Max diff: inf at index [0]
   Mean diff: inf
   Max relative error: inf%, Mean: inf%
   Wrong elements: 128 / 128 (100.00%)
   Error distribution: <1e-6: 0 | 1e-6~1e-4: 0 | 1e-4~1e-2: 0 | 1e-2~1: 0 | >1: 128
   Pattern: ALL elements wrong - likely algorithmic error or wrong weights
   Sample failures:
      [0]: ref=1533.442139 impl=-inf (diff=inf)

... (truncated, 1994 chars total)
```

### Assistant (Turn 3)
The kernel is producing all -inf values, which means there's a logical error in my maxpool implementation. Let me analyze and fix the issue:

**Tool Call: `write`**
```json
{
  "path": "optimized_kernel.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Fused kernel: matmul + maxpool + sum + scale\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n// Kernel configuration\n#define BLOCK_SIZE 256\n#define UNROLL_FACTOR 4\n\n// Fused kernel: linear + maxpool + sum + scale\n// Performs: output = scale * sum(maxpool(linear(x)))\n// Note: maxpool only operates along feature dimension\n__global__ void fused_kernel(\n    const float* __restrict__ x,      // input: (batch_size, in_features)\n    const float* __restrict__ W,      // weight: (out_features, in_features)  \n    const float* __restrict__ bias,   // bias: (out_features,)\n    float* __restrict__ output,       // output: (batch_size,)\n    int batch_size,\n    int in_features,\n    int out_features,\n    int kernel_size,                  // maxpool kernel size (2)\n    float scale_factor\n) {\n    int batch_idx = blockIdx.x;\n    int tid = threadIdx.x;\n    \n    if (batch_idx >= batch_size) return;\n    \n    // Each block processes one batch element\n    __shared__ float shared_buffer[BLOCK_SIZE];\n    \n    // Process output features in tiles\n    // Each thread computes multiple maxpooled outputs\n    float thread_total = 0.0f;\n    \n    // Loop through output feature pairs (maxpool groups)\n    for (int out_tile_start = tid * kernel_size; out_tile_start < out_features; \n         out_tile_start += BLOCK_SIZE * kernel_size) {\n        \n        // Process up to kernel_size outputs (for maxpool)\n        for (int k = 0; k < kernel_size; k++) {\n            int out_idx = out_tile_start + k;\n            if (out_idx >= out_features) continue;\n            \n            // Compute dot product: x[batch_idx, :] @ W[out_idx, :]\n            float dot_val = (bias != nullptr) ? bias[out_idx] : 0.0f;\n            \n            // Vectorized dot product - process 4 elements at a time\n            int i = 0;\n            for (; i < in_features - 3; i += 4) {\n                float4 x_vec = *reinterpret_cast<const float4*>(&x[batch_idx * in_features + i]);\n                float4 w_vec = *reinterpret_cast<const float4*>(&W[out_idx * in_features + i]);\n                \n                dot_val += x_vec.x * w_vec.x;\n                dot_val += x_vec.y * w_vec.y;\n                dot_val += x_vec.z * w_vec.z;\n                dot_val += x_vec.w * w_vec.w;\n            }\n            \n            // Handle remaining elements\n            for (; i < in_features; i++) {\n                dot_val += x[batch_idx * in_features + i] * W[out_idx * in_features + i];\n            }\n            \n            // For maxpool: take max over kernel_size adjacent features\n            if (k == 0) {\n                // Store first value for this maxpool group\n                shared_buffer[tid] = dot_val;\n            } else {\n                // Update max value for this maxpool group\n                shared_buffer[tid] = fmaxf(shared_buffer[tid], dot_val);\n            }\n        }\n        \n        // Accumulate the max value from this group\n        thread_total += shared_buffer[tid];\n    }\n    \n    // Global reduction for this batch element\n    shared_buffer[tid] = thread_total;\n    __syncthreads();\n    \n    // Parallel reduction in shared memory\n    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            shared_buffer[tid] += shared_buffer[tid + s];\n        }\n        __syncthreads();\n    }\n    \n    // Write final result for this batch \n    if (tid == 0) {\n        output[batch_idx] = shared_buffer[0] * scale_factor;\n    }\n}\n\ntorch::Tensor fused_forward(\n    torch::Tensor x,      // (batch_size, in_features)\n    torch::Tensor W,      // (out_features, in_features)\n    torch::Tensor bias,   // (out_features,)\n    float scale_factor,\n    int kernel_size\n) {\n    int batch_size = x.size(0);\n    int in_features = x.size(1);\n    int out_features = W.size(0);\n    \n    // Allocate output tensor (batch_size,)\n    auto output = torch::zeros({batch_size}, x.options());\n    \n    dim3 blocks(batch_size);\n    dim3 threads(BLOCK_SIZE);\n    \n    fused_kernel<<<blocks, threads>>>(\n        x.data_ptr<float>(),\n        W.data_ptr<float>(),\n        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,\n        output.data_ptr<float>(),\n        batch_size,\n        in_features,\n        out_features,\n        kernel_size,\n        scale_factor\n    );\n    \n    return output;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_kernel\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_forward\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features, bias=True)\n        self.kernel_size = kernel_size\n        self.scale_factor = scale_factor\n        self.fused_op = fused_op\n    \n    def forward(self, x):\n        # Call fused kernel that combines linear, maxpool, sum, and scale\n        return self.fused_op.fused_forward(\n            x, self.linear.weight, self.linear.bias, \n            self.scale_factor, self.kernel_size\n        )\n\n\nbatch_size = 128\nin_features = 32768\nout_features = 32768\nkernel_size = 2\nscale_factor = 0.5\n\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features)]\n\n\ndef get_init_inputs():\n    return [in_features, out_features, kernel_size, scale_factor]\n"
}
```

### Tool Result
```
Successfully wrote 5372 bytes to optimized_kernel.py
```

### Assistant (Turn 4)
**Tool Call: `write`**
```json
{
  "path": "optimized_kernel.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Fused kernel: matmul + maxpool + sum + scale\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n\n#define BLOCK_SIZE 256\n\n// Fused kernel: linear + maxpool + sum + scale\n__global__ void fused_kernel(\n    const float* __restrict__ x,      // input: (batch_size, in_features)\n    const float* __restrict__ W,      // weight: (out_features, in_features)\n    const float* __restrict__ bias,   // bias: (out_features,)\n    float* __restrict__ output,       // output: (batch_size,)\n    int batch_size,\n    int in_features,\n    int out_features,\n    int kernel_size,                  // maxpool kernel size (2)\n    float scale_factor\n) {\n    // Each block processes one batch element\n    __shared__ float shared_sum[BLOCK_SIZE];\n    \n    int batch_idx = blockIdx.x;\n    int tid = threadIdx.x;\n    \n    if (batch_idx >= batch_size) return;\n    \n    float thread_result = 0.0f;\n    \n    // Process output features in tiles\n    // Each thread computes the maxpooled result for a region\n    for (int out_start = tid * kernel_size; out_start < out_features; \n         out_start += BLOCK_SIZE * kernel_size) {\n        \n        float max_val = -1e30f;  // Very small number for max initialization\n        \n        // Process maxpool group [out_start, out_start + kernel_size)\n        for (int k = 0; k < kernel_size && (out_start + k) < out_features; k++) {\n            int out_idx = out_start + k;\n            \n            // Compute linear transformation: out = W @ x + bias[out_idx]\n            float dot = 0.0f;\n            \n            // Add bias if present\n            if (bias != nullptr) {\n                dot = bias[out_idx];\n            }\n            \n            // Vectorized matmul: process 4 elements at a time\n            int i = 0;\n            for (; i <= in_features - 4; i += 4) {\n                float4 x_val = *reinterpret_cast<const float4*>(&x[batch_idx * in_features + i]);\n                float4 w_val = *reinterpret_cast<const float4*>(&W[out_idx * in_features + i]);\n                \n                dot += x_val.x * w_val.x;\n                dot += x_val.y * w_val.y;\n                dot += x_val.z * w_val.z;\n                dot += x_val.w * w_val.w;\n            }\n            \n            // Handle remaining elements\n            for (; i < in_features; i++) {\n                dot += x[batch_idx * in_features + i] * W[out_idx * in_features + i];\n            }\n            \n            // Update maxpool (take maximum in the group)\n            max_val = fmaxf(max_val, dot);\n        }\n        \n        // Accumulate the max value for this group\n        // Only accumulate if we processed this group\n        if (out_start < out_features) {\n            thread_result += max_val;\n        }\n    }\n    \n    // Store thread contribution to shared memory\n    shared_sum[tid] = thread_result;\n    __syncthreads();\n    \n    // Parallel reduction in shared memory\n    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            shared_sum[tid] += shared_sum[tid + s];\n        }\n        __syncthreads();\n    }\n    \n    // Final result: write scaled sum for this batch element\n    if (tid == 0) {\n        output[batch_idx] = shared_sum[0] * scale_factor;\n    }\n}\n\ntorch::Tensor fused_forward(\n    torch::Tensor x,\n    torch::Tensor W,\n    torch::Tensor bias,\n    float scale_factor,\n    int kernel_size\n) {\n    // Validate dimensions\n    auto batch_size = x.size(0);\n    auto in_features = x.size(1);\n    auto out_features = W.size(0);\n    \n    // Allocate output of shape (batch_size,)\n    auto output = torch::zeros({batch_size}, x.options());\n    \n    // Launch one block per batch element\n    dim3 blocks(batch_size);\n    dim3 threads(BLOCK_SIZE);\n    \n    // Launch kernel\n    fused_kernel<<<blocks, threads>>>(\n        x.data_ptr<float>(),\n        W.data_ptr<float>(),\n        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,\n        output.data_ptr<float>(),\n        batch_size,\n        in_features,\n        out_features,\n        kernel_size,\n        scale_factor\n    );\n    \n    return output;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_kernel\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_forward\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        # Use same initialization as reference\n        self.linear = nn.Linear(in_features, out_features, bias=True)\n        self.kernel_size = kernel_size\n        self.scale_factor = scale_factor\n        self.fused_op = fused_op\n    \n    def forward(self, x):\n        # Fused kernel combining linear + maxpool + sum + scale\n        return self.fused_op.fused_forward(\n            x, self.linear.weight, self.linear.bias,  \n            self.scale_factor, self.kernel_size\n        )\n\n\nbatch_size = 128\nin_features = 32768\nout_features = 32768\nkernel_size = 2\nscale_factor = 0.5\n\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features)]\n\n\ndef get_init_inputs():\n    return [in_features, out_features, kernel_size, scale_factor]\n"
}
```

### Tool Result
```
Successfully wrote 5175 bytes to optimized_kernel.py
```

### Assistant (Turn 5)


### Assistant (Turn 6)
<|tool_call_begin|>functions.bash:4<|tool_call_argument_begin|>{"command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive", "timeout": 300}<|tool_call_end|><|tool_calls_section_end|>

### Assistant (Turn 7)


### Assistant (Turn 8)
<|tool_calls_section_end|>

### Assistant (Turn 9)
The user wants me to optimize a PyTorch architecture with custom HIP/ROCm kernels. The architecture is:

```python
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
```

The operations in the forward pass:
1. `x = self.matmul(x)`: Linear layer (matmul + bias) - shape: (batch_size, out_features)
2. `x = self.max_pool(x.unsqueeze(1)).squeeze(1)`: MaxPool1d with kernel_size - This pools groups of `kernel_size` features along the feature dimension. Since `kernel_size=2`, it takes max of every 2 features.
3. `x = torch.sum(x, dim=1)`: Sum along feature dimension - shape: (batch_size,)
4. `x = x * self.scale_factor`: Scale by a factor - shape: (batch_size,)

The input shape is (batch_size=128, in_features=32768), matmul produces (128, 32768), maxpool with kernel_size=2 reduces it to (128, 16384), sum reduces it to (128,), scale keeps shape (128,).

The optimization strategy:
- This is a perfect candidate for kernel fusion! We're doing matmul → maxpool → sum → scale.
- The matmul produces 32768 features per batch item, then maxpool reduces by 2x.
- Then we sum all features.
- The key insight: we don't need to write the full matmul result to memory! We can compute matmul and immediately apply maxpool and sum before writing anything.

Let me think about the maxpool operation more carefully. In PyTorch, MaxPool1d with kernel_size=2 and default stride will have stride=2. So it pools every 2 elements along the feature dimension, producing half the features (out_features // 2).

But wait, let's verify the reference implementation's parameters:
- in_features = 32768
- out_features = 32768
- kernel_size = 2

So the operations are:
1. Linear: (128, 32768) @ (32768, 32768) + bias = (128, 32768)
2. MaxPool1d(kernel_size=2): (128, 32768) -> (128, 16384) (taking max of pairs)
3. Sum(dim=1): (128, 16384) -> (128,)
4. Scale: (128,) -> (128,)

For kernel fusion:
- Each batch element can be processed independently
- For each batch element, we need to:
  - Compute linear transformation for each output feature (32768 features)
  - For every group of 2 features, take the max
  - Sum all the max values
  - Multiply by scale factor

Optimization opportunities:
1. **Fuse all operations**: Instead of writing intermediate tensors, compute everything in registers/shared memory
2. **Vectorization**: Use float4 for efficient memory access
3. **Tiling**: Use shared memory for intermediate results
4. **Loop unrolling**: The kernel_size is small (2), so unroll the maxpool comparison

Let me design the kernel:

**Kernel Design:**
- Each thread block processes one batch element (batch_size=128, so 128 blocks)
- Each thread computes multiple output features and performs the maxpool+sum
- Use shared memory for reduction across threads in a block

**Detailed kernel logic:**
For a batch element `b`:
- Each thread `t` in block `b` will compute outputs for certain output features:
  - Let's say thread `t` computes features in `[t*stride, (t+1)*stride)` where stride is chosen to cover all features
  - Actually, since we need to maxpool over kernel_size=2, each thread should compute a pair of features and take their max
  - Or better: each thread computes multiple pairs

Let me rethink:
- We have 32768 output features before pooling → 16384 after pooling
- Each of the 16384 pooled features requires computing 2 linear outputs and taking max
- So we have 16384 "maxpool groups", each producing 1 value

We can:
- Assign each thread to handle multiple maxpool groups
- Each maxpool group computes 2 linear outputs, takes max, and adds to running sum
- After all groups processed, perform block-level reduction to sum all thread contributions
- Scale the final result

Let me write the actual kernel:

```c++
__global__ void fused_kernel(
    const float* x,    // [batch_size, in_features]
    const float* W,    // [out_features, in_features]  (out_features = 32768)
    const float* bias, // [out_features]
    float* output,     // [batch_size]
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    // Each block processes one batch element
    // Each thread processes multiple maxpool groups
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ float shared_sum[BLOCK_SIZE];
    float thread_sum = 0.0f;
    
    // Number of maxpool groups = out_features / kernel_size = 16384
    int num_groups = out_features / kernel_size;
    
    // Each thread handles multiple groups
    for (int group = tid; group < num_groups; group += blockDim.x) {
        // Compute two linear outputs for this group (kernel_size = 2)
        float vals[2];
        
        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            int out_idx = group * kernel_size + k;
            
            // Compute dot product for out_idx
            float dot = 0.0f;
            if (bias) dot = bias[out_idx];
            
            // Vectorized dot product
            int i = 0;
            for (; i <= in_features - 4; i += 4) {
                float4 x_vec = *reinterpret_cast<const float4*>(&x[batch_idx * in_features + i]);
                float4 w_vec = *reinterpret_cast<const float4*>(&W[out_idx * in_features + i]);
                dot += x_vec.x * w_vec.x + x_vec.y * w_vec.y + x_vec.z * w_vec.z + x_vec.w * w_vec.w;
            }
            for (; i < in_features; i++) {
                dot += x[batch_idx * in_features + i] * W[out_idx * in_features + i];
            }
            
            vals[k] = dot;
        }
        
        // Max pooling over kernel_size elements (2)
        float max_val = vals[0];
        #pragma unroll
        for (int k = 1; k < kernel_size; k++) {
            max_val = fmaxf(max_val, vals[k]);
        }
        
        thread_sum += max_val;
    }
    
    // Reduction
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = shared_sum[0] * scale_factor;
    }
}
```

Wait, I need to be careful about the maxpool groups. Let me verify the PyTorch MaxPool1d behavior:

MaxPool1d(kernel_size=2) on a 1D tensor of length 32768 will produce length 16384 (since stride defaults to kernel_size). It takes max over elements [0,1], [2,3], [4,5], etc.

So the groups are indeed (0,1), (2,3), (4,5), ..., (32766, 32767). So I need to compute max over each pair.

My kernel above should work: each thread computes a group (pair), takes max, adds to thread sum. Then block reduction sums all thread sums.

Let me refine the kernel to make sure I handle edge cases and avoid register pressure:

Actually, I can simplify. Instead of storing all `vals[k]` in registers, I can compute max on the fly:

```c++
float max_val = -INFINITY;
#pragma unroll
for (int k = 0; k < kernel_size; k++) {
    int out_idx = group * kernel_size + k;
    // compute dot...
    float dot = ...;
    max_val = fmaxf(max_val, dot);
}
thread_sum += max_val;
```

This is better - less register usage.

Let me write the final kernel code and the PyTorch wrapper.

Now for the Python part, I need to:
1. Load the HIP code with `load_inline`
2. Create the `ModelNew` class that uses the fused kernel
3. Set up weight initialization to match the reference

The reference uses default Linear initialization, which is uniform. Let me check the reference code:

```python
def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
```

So the inputs to `__init__` are just the parameters. The Linear layer will use its default initialization. In PyTorch, `nn.Linear` uses Kaiming uniform initialization by default.

Let me write the full optimized kernel file.

I'll name it `optimized_kernel.py` and include everything needed.

After writing the file, I MUST run:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation optimized_kernel.py --reference /root/Wafer/research

/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive
```

Let me write the file:</think>

**Tool Call: `write`**
```json
{
  "path": "optimized_kernel.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Fused kernel: matmul + maxpool + sum + scale\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <math.h>\n\n#define BLOCK_SIZE 256\n\n// Fused kernel: linear + maxpool + sum + scale\n// Performs: output = scale * sum(maxpool(linear(x)))\n__global__ void fused_kernel(\n    const float* __restrict__ x,      // input: (batch_size, in_features)\n    const float* __restrict__ W,      // weight: (out_features, in_features)\n    const float* __restrict__ bias,   // bias: (out_features,)\n    float* __restrict__ output,       // output: (batch_size,)\n    int batch_size,\n    int in_features,\n    int out_features,\n    int kernel_size,                  // maxpool kernel size (2)\n    float scale_factor\n) {\n    // Each block processes one batch element\n    int batch_idx = blockIdx.x;\n    int tid = threadIdx.x;\n    \n    if (batch_idx >= batch_size) return;\n    \n    __shared__ float shared_sum[BLOCK_SIZE];\n    float thread_sum = 0.0f;\n    \n    // Number of maxpool groups = out_features / kernel_size\n    int num_groups = out_features / kernel_size;\n    \n    // Each thread processes multiple groups\n    for (int group = tid; group < num_groups; group += blockDim.x) {\n        float max_val = -1e30f;  // Very low initial value\n        \n        // Process each element in the maxpool group\n        #pragma unroll\n        for (int k = 0; k < kernel_size; k++) {\n            int out_idx = group * kernel_size + k;\n            \n            // Compute linear output: out = dot(x, W[out_idx]) + bias[out_idx]\n            float dot = (bias != nullptr) ? bias[out_idx] : 0.0f;\n            \n            // Vectorized dot product: process 4 elements at a time\n            int i = 0;\n            for (; i <= in_features - 4; i += 4) {\n                float4 x_vec = *reinterpret_cast<const float4*>(&x[batch_idx * in_features + i]);\n                float4 w_vec = *reinterpret_cast<const float4*>(&W[out_idx * in_features + i]);\n                \n                dot += x_vec.x * w_vec.x;\n                dot += x_vec.y * w_vec.y;\n                dot += x_vec.z * w_vec.z;\n                dot += x_vec.w * w_vec.w;\n            }\n            \n            // Handle remaining elements\n            for (; i < in_features; i++) {\n                dot += x[batch_idx * in_features + i] * W[out_idx * in_features + i];\n            }\n            \n            // Update max value for this group\n            max_val = fmaxf(max_val, dot);\n        }\n        \n        // Accumulate the max value for this group\n        thread_sum += max_val;\n    }\n    \n    // Store thread contribution in shared memory\n    shared_sum[tid] = thread_sum;\n    __syncthreads();\n    \n    // Parallel reduction in shared memory\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            shared_sum[tid] += shared_sum[tid + s];\n        }\n        __syncthreads();\n    }\n    \n    // Write final result for this batch element\n    if (tid == 0) {\n        output[batch_idx] = shared_sum[0] * scale_factor;\n    }\n}\n\ntorch::Tensor fused_forward(\n    torch::Tensor x,\n    torch::Tensor W,\n    torch::Tensor bias,\n    float scale_factor,\n    int kernel_size\n) {\n    auto batch_size = x.size(0);\n    auto in_features = x.size(1);\n    auto out_features = W.size(0);\n    \n    // Allocate output tensor of shape (batch_size,)\n    auto output = torch::zeros({batch_size}, x.options());\n    \n    // Launch one block per batch element\n    dim3 blocks(batch_size);\n    dim3 threads(BLOCK_SIZE);\n    \n    fused_kernel<<<blocks, threads>>>(\n        x.data_ptr<float>(),\n        W.data_ptr<float>(),\n        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,\n        output.data_ptr<float>(),\n        batch_size,\n        in_features,\n        out_features,\n        kernel_size,\n        scale_factor\n    );\n    \n    return output;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_kernel\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_forward\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features, bias=True)\n        self.kernel_size = kernel_size\n        self.scale_factor = scale_factor\n        self.fused_op = fused_op\n        \n        # Initialize with same distribution as reference (uniform)\n        nn.init.uniform_(self.linear.weight, -0.1, 0.1)\n        nn.init.uniform_(self.linear.bias, -0.1, 0.1)\n    \n    def forward(self, x):\n        # Fused forward: linear + maxpool + sum + scale\n        return self.fused_op.fused_forward(\n            x, self.linear.weight, self.linear.bias,\n            self.scale_factor, self.kernel_size\n        )\n\n\nbatch_size = 128\nin_features = 32768\nout_features = 32768\nkernel_size = 2\nscale_factor = 0.5\n\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features)]\n\n\ndef get_init_inputs():\n    return [in_features, out_features, kernel_size, scale_factor]\n"
}
```

### Tool Result
```
Successfully wrote 5063 bytes to optimized_kernel.py
```

### Assistant (Turn 10)
Now let me test the kernel:

**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_164949
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_kernel -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_kernel/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_kernel.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Diff analysis:
   Max diff: 26616.214844 at index [5]
   Mean diff: 26228.212891
   Max relative error: 1750.37%, Mean: 1707.33%
   Wrong elements: 128 / 128 (100.00%)
   Error distribution: <1e-6: 0 | 1e-6~1e-4: 0 | 1e-4~1e-2: 0 | 1e-2~1: 0 | >1: 128
   Pattern: ALL elements wrong - likely algorithmic error or wrong weights
   Sample failures:
      [0]: ref=1533.442139 impl=27911.660156 (diff=26378.218750)
      [1]:

... (truncated, 2062 chars total)
```

### Assistant (Turn 11)
Let me fix the initialization issue and simplify the kernel logic:

**Tool Call: `write`**
```json
{
  "path": "optimized_kernel.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nos.environ[\"CXX\"] = \"hipcc\"\n\n# Fused kernel: matmul + maxpool + sum + scale\ncpp_source = \"\"\"\n#include <hip/hip_runtime.h>\n#include <math.h>\n\n#define BLOCK_SIZE 256\n\n// Fused kernel: linear + maxpool + sum + scale\n// Performs: output = scale * sum(maxpool(linear(x)))\n__global__ void fused_kernel(\n    const float* __restrict__ x,      // input: (batch_size, in_features)\n    const float* __restrict__ W,      // weight: (out_features, in_features)\n    const float* __restrict__ bias,   // bias: (out_features,)\n    float* __restrict__ output,       // output: (batch_size,)\n    int batch_size,\n    int in_features,\n    int out_features,\n    int kernel_size,                  // maxpool kernel size (2)\n    float scale_factor\n) {\n    // Each block processes one batch element\n    int batch_idx = blockIdx.x;\n    int tid = threadIdx.x;\n    \n    if (batch_idx >= batch_size) return;\n    \n    __shared__ float shared_sum[BLOCK_SIZE];\n    float thread_sum = 0.0f;\n    \n    // Number of maxpool groups = out_features / kernel_size\n    int num_groups = out_features / kernel_size;\n    \n    // Each thread processes multiple maxpool groups\n    for (int group = tid; group < num_groups; group += blockDim.x) {\n        float max_val = -1e30f;  // Very low initial value\n        \n        // Process each element in the maxpool group\n        #pragma unroll\n        for (int k = 0; k < kernel_size; k++) {\n            int out_idx = group * kernel_size + k;\n            \n            // Compute linear output: out = dot(x, W[out_idx]) + bias[out_idx]\n            float dot = (bias != nullptr) ? bias[out_idx] : 0.0f;\n            \n            // Vectorized dot product: process 4 elements at a time\n            int i = 0;\n            for (; i <= in_features - 4; i += 4) {\n                float4 x_vec = *reinterpret_cast<const float4*>(&x[batch_idx * in_features + i]);\n                float4 w_vec = *reinterpret_cast<const float4*>(&W[out_idx * in_features + i]);\n                \n                dot += x_vec.x * w_vec.x;\n                dot += x_vec.y * w_vec.y;\n                dot += x_vec.z * w_vec.z;\n                dot += x_vec.w * w_vec.w;\n            }\n            \n            // Handle remaining elements\n            for (; i < in_features; i++) {\n                dot += x[batch_idx * in_features + i] * W[out_idx * in_features + i];\n            }\n            \n            // Update max value for this group\n            max_val = fmaxf(max_val, dot);\n        }\n        \n        // Accumulate the max value for this group\n        thread_sum += max_val;\n    }\n    \n    // Store thread contribution in shared memory\n    shared_sum[tid] = thread_sum;\n    __syncthreads();\n    \n    // Parallel reduction in shared memory\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            shared_sum[tid] += shared_sum[tid + s];\n        }\n        __syncthreads();\n    }\n    \n    // Write final result for this batch element\n    if (tid == 0) {\n        output[batch_idx] = shared_sum[0] * scale_factor;\n    }\n}\n\ntorch::Tensor fused_forward(\n    torch::Tensor x,\n    torch::Tensor W,\n    torch::Tensor bias,\n    float scale_factor,\n    int kernel_size\n) {\n    auto batch_size = x.size(0);\n    auto in_features = x.size(1);\n    auto out_features = W.size(0);\n    \n    // Allocate output tensor of shape (batch_size,)\n    auto output = torch::zeros({batch_size}, x.options());\n    \n    // Launch one block per batch element\n    dim3 blocks(batch_size);\n    dim3 threads(BLOCK_SIZE);\n    \n    fused_kernel<<<blocks, threads>>>(\n        x.data_ptr<float>(),\n        W.data_ptr<float>(),\n        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,\n        output.data_ptr<float>(),\n        batch_size,\n        in_features,\n        out_features,\n        kernel_size,\n        scale_factor\n    );\n    \n    return output;\n}\n\"\"\"\n\nfused_op = load_inline(\n    name=\"fused_kernel\",\n    cpp_sources=cpp_source,\n    functions=[\"fused_forward\"],\n    verbose=True,\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self, in_features, out_features, kernel_size, scale_factor):\n        super(ModelNew, self).__init__()\n        self.linear = nn.Linear(in_features, out_features, bias=True)\n        self.kernel_size = kernel_size\n        self.scale_factor = scale_factor\n        self.fused_op = fused_op\n        # Use PyTorch default initialization (same as reference)\n    \n    def forward(self, x):\n        # Fused forward: linear + maxpool + sum + scale\n        return self.fused_op.fused_forward(\n            x, self.linear.weight, self.linear.bias,\n            self.scale_factor, self.kernel_size\n        )\n\n\nbatch_size = 128\nin_features = 32768\nout_features = 32768\nkernel_size = 2\nscale_factor = 0.5\n\n\ndef get_inputs():\n    return [torch.rand(batch_size, in_features)]\n\n\ndef get_init_inputs():\n    return [in_features, out_features, kernel_size, scale_factor]\n"
}
```

### Tool Result
```
Successfully wrote 4950 bytes to optimized_kernel.py
```

### Assistant (Turn 12)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation optimized_kernel.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py --benchmark --defensive",
  "timeout": 300
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_165200
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
[1/2] hipcc -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=fused_kernel -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.12/site-packages/torch/include -isystem /opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cpu/fused_kernel/main.cpp -o main.o -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC
[2/2] hipcc main.o -shared -L/opt/venv/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o fused_kernel.so
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 306.221ms, Ref: 2.740ms, Speedup: 0.01x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_165200/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 0.01x
============================================================

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
