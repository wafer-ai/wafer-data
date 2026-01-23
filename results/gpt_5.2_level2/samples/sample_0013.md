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
Agent correctly ran wafer evaluate kernelbench with --defensive flag and achieved excellent results. The optimization demonstrates high-quality engineering: (1) Brilliant algebraic insight reducing Linear(8192→8192) + AvgPool1d to equivalent Linear(8192→512) by averaging weight/bias groups, cutting GEMM cost ~16x; (2) Custom HIP kernel fusing GELU+scale+max operations; (3) Verified correctness (100% pass); (4) Exceptional 9.50x speedup (0.137ms vs 1.304ms); (5) Clean, well-documented code with proper state management. Only minor deduction for slightly verbose implementation, but overall this represents expert-level GPU optimization.
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
  "path": "model_new_98.py",
  "content": "import os\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.cpp_extension import load_inline\n\n# Compile HIP extension\nos.environ.setdefault(\"CXX\", \"hipcc\")\n\nhip_src = r\"\"\"\n#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n#include <ATen/cuda/CUDAContext.h>\n\n__device__ __forceinline__ float gelu_exact(float x) {\n    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))\n    const float inv_sqrt2 = 0.7071067811865475244f;\n    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));\n}\n\n__global__ void gelu_scale_max_kernel(const float* __restrict__ inp,\n                                     float* __restrict__ out,\n                                     int ncols,\n                                     float scale) {\n    // One block per row.\n    const int row = (int)blockIdx.x;\n    const int tid = (int)threadIdx.x;\n\n    float maxv = -INFINITY;\n    const int row_off = row * ncols;\n\n    // Assume ncols is multiple of 4 for vector loads (true for 512).\n    const int idx4 = tid * 4;\n    if (idx4 + 3 < ncols) {\n        const float4 v = *reinterpret_cast<const float4*>(inp + row_off + idx4);\n        float a0 = gelu_exact(v.x) * scale;\n        float a1 = gelu_exact(v.y) * scale;\n        float a2 = gelu_exact(v.z) * scale;\n        float a3 = gelu_exact(v.w) * scale;\n        maxv = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));\n    } else {\n        // Tail (shouldn't happen for ncols=512)\n        for (int j = idx4; j < ncols; ++j) {\n            float a = gelu_exact(inp[row_off + j]) * scale;\n            maxv = fmaxf(maxv, a);\n        }\n    }\n\n    extern __shared__ float sdata[];\n    sdata[tid] = maxv;\n    __syncthreads();\n\n    // Parallel reduction (max)\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) {\n            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);\n        }\n        __syncthreads();\n    }\n\n    if (tid == 0) {\n        out[row] = sdata[0];\n    }\n}\n\ntorch::Tensor gelu_scale_max_hip(torch::Tensor input, double scale) {\n    TORCH_CHECK(input.is_cuda(), \"input must be CUDA/HIP tensor\");\n    TORCH_CHECK(input.dtype() == torch::kFloat32, \"FP32 only\");\n    TORCH_CHECK(input.dim() == 2, \"input must be 2D\");\n\n    auto inp = input.contiguous();\n    const int64_t B = inp.size(0);\n    const int64_t N = inp.size(1);\n\n    auto out = torch::empty({B}, inp.options());\n\n    // Use 128 threads so each thread processes 4 elements for N=512.\n    const int threads = 128;\n    TORCH_CHECK((threads * 4) >= N, \"threads*4 must cover N\");\n\n    const dim3 blocks((uint32_t)B);\n    const size_t shmem = threads * sizeof(float);\n\n    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream().stream();\n    hipLaunchKernelGGL(gelu_scale_max_kernel, blocks, dim3(threads), shmem, stream,\n                       (const float*)inp.data_ptr<float>(),\n                       (float*)out.data_ptr<float>(),\n                       (int)N,\n                       (float)scale);\n\n    return out;\n}\n\"\"\"\n\n# Build extension once.\n_gelu_scale_max = load_inline(\n    name=\"gelu_scale_max_ext_98\",\n    cpp_sources=hip_src,\n    functions=[\"gelu_scale_max_hip\"],\n    extra_cflags=[\"-O3\"],\n    with_cuda=True,\n    verbose=False,\n)\n\n\nclass ModelNew(nn.Module):\n    \"\"\"Optimized model:\n\n    Exploits algebraic equivalence:\n      Linear(8192->8192) + AvgPool1d(k=16, stride=16)\n    == Linear(8192->512) with weights/bias averaged over groups of 16 output rows.\n\n    Then uses a fused HIP kernel for GELU + scale + max-reduction.\n    \"\"\"\n\n    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):\n        super().__init__()\n        self.matmul = nn.Linear(in_features, out_features, bias=True)\n        self.pool_kernel_size = int(pool_kernel_size)\n        self.scale_factor = float(scale_factor)\n\n        assert out_features % self.pool_kernel_size == 0\n        self.pooled_features = out_features // self.pool_kernel_size\n\n        # Cached aggregated weights/bias to avoid building them every forward.\n        self.register_buffer(\"_w_agg\", torch.empty(0), persistent=False)\n        self.register_buffer(\"_b_agg\", torch.empty(0), persistent=False)\n        self._agg_ready = False\n\n        self._ext = _gelu_scale_max\n\n    def _rebuild_agg(self):\n        k = self.pool_kernel_size\n        w = self.matmul.weight  # (out_features, in_features)\n        b = self.matmul.bias    # (out_features,)\n\n        # AvgPool1d on length dimension corresponds to averaging over groups of k output channels.\n        w_agg = w.view(self.pooled_features, k, w.size(1)).mean(dim=1)\n        b_agg = b.view(self.pooled_features, k).mean(dim=1)\n\n        # Keep contiguous for GEMM and kernel reads.\n        self._w_agg = w_agg.contiguous()\n        self._b_agg = b_agg.contiguous()\n        self._agg_ready = True\n\n    def load_state_dict(self, state_dict, strict: bool = True):\n        out = super().load_state_dict(state_dict, strict=strict)\n        self._agg_ready = False\n        return out\n\n    def forward(self, x):\n        if (not self._agg_ready) or (self._w_agg.device != x.device):\n            self._rebuild_agg()\n\n        # Reduced GEMM: (B, in_features) x (pooled_features, in_features)^T\n        y = F.linear(x, self._w_agg, self._b_agg)\n        return self._ext.gelu_scale_max_hip(y, self.scale_factor)\n"
}
```

### Tool Result
```
Successfully wrote 5253 bytes to model_new_98.py
```

### Assistant (Turn 2)
**Tool Call: `bash`**
```json
{
  "command": "wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive",
  "timeout": 600
}
```

### Tool Result
```
Using target: local-mi300x
Connecting to root@localhost:22...
Created run directory: /root/.wafer/workspaces/kernelbench/kernelbench_eval_20260122_113441
Running KernelBench evaluation in Docker container...
Docker command: docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /root/.wafer/workspaces/ker...
Requirement already satisfied: uv in /opt/venv/lib/python3.12/site-packages (0.9.26)
Successfully preprocessed all matching files.
[KernelBench] Modules loaded successfully
[KernelBench] Models instantiated (seed=42)
[KernelBench] Correctness: True
[KernelBench] Running benchmarks...
[KernelBench] New: 0.137ms, Ref: 1.304ms, Speedup: 9.50x
[KernelBench] Results written to /workspace/kernelbench_eval_20260122_113441/results.json

============================================================
Result: PASS
Correctness: 1/1 (100.0%)
Speedup: 9.50x
============================================================

```

### Assistant (Turn 3)
Implemented `ModelNew` in `model_new_98.py` with two key optimizations:

1. **Algorithmic rewrite (major win):**  
   `Linear(8192→8192) + AvgPool1d(k=16, stride=16)` is exactly equivalent to a single `Linear(8192→512)` where weights/bias are the mean over each group of 16 output rows.  
   `ModelNew` keeps the original `self.matmul` so KernelBench can load/copy weights, then lazily builds cached aggregated weights once and uses the reduced GEMM in forward.

2. **Custom HIP kernel fusion:**  
   Fused **GELU (exact erf)** + **scale** + **max-reduction** over the 512 pooled features into one HIP kernel (`gelu_scale_max_hip`), reducing kernel launches and intermediate memory traffic.

### Test result (MANDATORY command)
Ran:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Output:
- Correctness: **True**
- New: **0.137 ms**, Ref: **1.304 ms**
- Speedup: **9.50×**
- Result: **PASS**

DONE

### Assistant (Turn 4)
Implemented `ModelNew` in `model_new_98.py` with:

- **Exact algebraic rewrite:** `Linear(8192→8192) + AvgPool1d(k=16,stride=16)` → `Linear(8192→512)` by averaging each group of 16 output rows of the original weight/bias (cached after `load_state_dict` / first forward).
- **Custom HIP kernel fusion:** one HIP kernel computes **GELU (exact erf) + scale + max-reduction** over the 512 pooled features.

Ran the required command:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Result: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

DONE

### Assistant (Turn 5)
Implemented `ModelNew` in `model_new_98.py` with:

- **Exact algebraic rewrite:** `Linear(8192→8192) + AvgPool1d(k=16,stride=16)` → `Linear(8192→512)` by averaging each group of 16 output rows of the original weight/bias (cached after `load_state_dict` / first forward).
- **Custom HIP kernel fusion:** one HIP kernel computes **GELU (exact erf) + scale + max-reduction** over the 512 pooled features.

Ran the required command:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Result: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

DONE

### Assistant (Turn 6)
Implemented `ModelNew` in `model_new_98.py` with:

- **Exact algebraic rewrite:** `Linear(8192→8192) + AvgPool1d(k=16,stride=16)` is exactly equivalent to a `Linear(8192→512)` whose weights/bias are the mean over each group of 16 output rows. `ModelNew` keeps the original `self.matmul` so KernelBench can load/copy weights, then lazily builds/caches the aggregated weights.
- **Custom HIP fusion kernel:** a HIP kernel computes **GELU (exact erf)** + **scale** + **max-reduction** in one pass.

Required benchmark command executed:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Output: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

DONE

### Assistant (Turn 7)
Implemented `ModelNew` in `model_new_98.py` with:

- **Exact algebraic rewrite:** `Linear(8192→8192) + AvgPool1d(k=16,stride=16)` → `Linear(8192→512)` by averaging each group of 16 output rows of the original weight/bias (cached; rebuilt after `load_state_dict` / on first forward).
- **Custom HIP kernel fusion:** one HIP kernel performs **GELU (exact erf) + scale + max-reduction** over the 512 pooled features.

Ran the required command:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Result: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

DONE

### Assistant (Turn 8)
Implemented `ModelNew` in `model_new_98.py` with:

- **Exact algebraic rewrite:** `Linear(8192→8192) + AvgPool1d(k=16,stride=16)` is exactly equivalent to a `Linear(8192→512)` whose weights/bias are the mean over each group of 16 output rows. `ModelNew` keeps the original `self.matmul` so KernelBench can load/copy weights, then lazily builds/caches the aggregated weights.
- **Custom HIP fused kernel:** a HIP kernel computes **GELU (exact erf) + scale + max-reduction** in one pass.

Executed the required benchmark:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Outcome: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

DONE

### Assistant (Turn 9)
Implemented `ModelNew` in `model_new_98.py` with:

- **Exact algebraic rewrite:** `Linear(8192→8192) + AvgPool1d(k=16,stride=16)` → `Linear(8192→512)` by averaging each group of 16 output rows of the original weight/bias (cached; rebuilt after `load_state_dict` / on first forward).
- **Custom HIP fused kernel:** fused **GELU (exact erf) + scale + max-reduction** over the 512 pooled features.

Required command executed:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Result: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

DONE

### Assistant (Turn 10)
Implemented `ModelNew` in `model_new_98.py` using:

- **Algebraic reduction (exact):** `Linear(8192→8192) + AvgPool1d(k=16,stride=16)` is equivalent to `Linear(8192→512)` with weights/bias averaged over each block of 16 output rows (cached after `load_state_dict` / first forward). This cuts the GEMM cost ~16×.
- **Custom HIP kernel:** fused **GELU (exact erf) + scale + max** into one HIP kernel.

Ran the required evaluation:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Output: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

DONE

### Assistant (Turn 11)
Implemented `ModelNew` in `model_new_98.py` with:

- **Exact algebraic rewrite:** `Linear(8192→8192) + AvgPool1d(k=16,stride=16)` → `Linear(8192→512)` by averaging each contiguous block of 16 output rows in the original linear weight/bias (cached; rebuilt after `load_state_dict` / on first forward).
- **Custom HIP fused kernel:** fused **GELU (exact erf) + scale + max-reduction** over the 512 pooled features.

Ran the required command:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Result: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

DONE

### Assistant (Turn 12)
Implemented `ModelNew` in `model_new_98.py` with:

- **Exact algebraic rewrite:** `Linear(8192→8192) + AvgPool1d(k=16, stride=16)` is exactly equivalent to a `Linear(8192→512)` where each output row (and bias) is the mean of the corresponding group of 16 original output rows. `ModelNew` keeps `self.matmul` so KernelBench can load/copy weights, then lazily builds/caches the aggregated weights.
- **Custom HIP fusion kernel:** a HIP kernel fuses **GELU (exact erf) + scale + max-reduction** into one pass.

Ran the required evaluation:
```bash
wafer evaluate kernelbench --target local-mi300x --implementation model_new_98.py --reference /root/Wafer/research/KernelBench/KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py --benchmark --defensive
```

Outcome: **PASS**, Correctness **True**, Speedup **9.50×** (New **0.137ms** vs Ref **1.304ms**).

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
