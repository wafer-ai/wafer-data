# KernelBench HIP — MI300X

LLM-generated HIP kernels on AMD MI300X, covering a [subset](https://x.com/elliotarledge/status/2012368833970053461) of problems from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) across 4 difficulty levels.

**Leaderboard:** [kernelarena.ai/eval?suite=kernelbench-hip](https://kernelarena.ai/eval?suite=kernelbench-hip)

## Requirements

- Python 3.10+
- PyTorch with ROCm 7.0
- AMD MI300X GPU

## Methodology

- **Harness:** Basic agent loop with `bash` and `write` tool access only — no IDE integration
- **Correctness:** `torch.allclose` with `rtol=1e-3`, `atol=1e-3` against reference PyTorch implementations
- **Libraries:** Models are allowed to use existing libraries (e.g. composable_kernel, Triton, hipBLASLt)
- **Scoring:** `score = (1 × correctness) + speedup`
- **Models:** 11 models from Anthropic, OpenAI, Google, xAI, Moonshot, and Z.AI
- **Tasks:** 41 kernels across 4 difficulty levels

## Structure

```
kernelbench-hip-mi300x/
├── index.json
├── solutions/
│   └── {task}/              # e.g. level1_1_Square_matrix_multiplication_
│       └── {model}/         # e.g. opus_4.5
│           └── kernel.py    # submitted solution
```

## Reproducibility

- Solutions are Python `.py` files at `solutions/{task}/{model}/kernel.py`
- Reference implementations are the original KernelBench PyTorch code
