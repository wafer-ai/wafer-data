# Wafer Data

Public benchmark datasets, traces, and static evaluation artifacts from Wafer.

This repository is for reusable data artifacts. Product apps, dashboards, and benchmark
runners live in their own repositories.

## Datasets

### `hip-benchmarks-results/`

Traces, kernels, and results from Wafer's HIP benchmark runs.

```text
hip-benchmarks-results/
├── kernels/
└── results/
```

### `kernel-arena/exports/`

Static public exports used by KernelArena.

```text
kernel-arena/exports/
├── kernelbench-hip-mi300x/
└── waferbench-nvfp4-b200/
```

The original KernelArena data repository history is preserved in
`kernel-arena/archive/kernel-arena-full-history.bundle`.

## Repository Scope

Keep this repository limited to public benchmark data and static artifacts. Do not add:

- application source code
- private/internal docs
- credentials or environment files
- customer data
- generated cache files such as `__pycache__/` or `.pyc`
