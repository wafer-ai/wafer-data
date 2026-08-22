# WaferBench NVFP4 — Speedup Results

**Date:** 2026-03-11
**Hardware:** 8x NVIDIA B200
**CUDA:** nvcc 13.1 (compile), PyTorch 2.7.0+cu128 (runtime)
**Reference:** FlashInfer 0.2.6.post1+cu128sm100 (public API, production code path)
**Methodology:** `bench_sustained` (TK 2.0 convention) — 500 warmup, 100 reps, 2 CUDA events, L2 cycling, ref-first execution order
**All tasks:** BF16 input, NVFP4 output (block_size=16, E4M3 scales)

**Passed:** 22 / 24 (92%)

## Results

| Task | Config | GPT-5.4 | Claude-4.6-Opus | Composer-1.5 | Gemini-3.1-Pro |
| --- | --- | --- | --- | --- | --- |
| `add_rmsnorm_fp4quant b128×h2048` | 2D BF16, non-swizzled, global_scale | **1.454x** | 1.428x | 1.163x | 1.452x |
| `add_rmsnorm_fp4quant b128×h4096` | 2D BF16, non-swizzled, global_scale | 1.458x | 1.421x | 1.355x | **1.503x** |
| `add_rmsnorm_fp4quant b128×h8192` | 2D BF16, non-swizzled, global_scale | 1.252x | 1.251x | 0.412x | **1.302x** |
| `nvfp4_quantize m128×k14336` | 2D BF16, non-swizzled | 0.931x | **1.710x** | 0.988x | 1.213x |
| `silu_mul_fp4quant b8×m256×k7168` | 3D BF16, swizzled output | 0.956x | **1.370x** | FAIL | 1.053x |
| `silu_mul_fp4quant b8×m256×k14336` | 3D BF16, swizzled output | 1.002x | FAIL | 0.235x | **1.120x** |

Score = (1 × correct) + speedup. Incorrect kernels get score 0.

| Task | GPT-5.4 | Claude-4.6-Opus | Composer-1.5 | Gemini-3.1-Pro |
| --- | --- | --- | --- | --- |
| `add_rmsnorm_fp4quant b128×h2048` | **2.454** | 2.428 | 2.163 | 2.452 |
| `add_rmsnorm_fp4quant b128×h4096` | 2.458 | 2.421 | 2.355 | **2.503** |
| `add_rmsnorm_fp4quant b128×h8192` | 2.252 | 2.251 | 1.412 | **2.302** |
| `nvfp4_quantize m128×k14336` | 1.931 | **2.710** | 1.988 | 2.213 |
| `silu_mul_fp4quant b8×m256×k7168` | 1.956 | **2.370** | 0 | 2.053 |
| `silu_mul_fp4quant b8×m256×k14336` | 2.002 | 0 | 1.235 | **2.120** |
| **Total** | **13.053** | **12.180** | **9.153** | **13.643** |

## Failures

| Task | Model | Reason |
| --- | --- | --- |
| `silu_mul_fp4quant b8×m256×k14336` | Claude-4.6-Opus | Incorrect output (correctness.py fail) |
| `silu_mul_fp4quant b8×m256×k7168` | Composer-1.5 | Incorrect output (correctness.py fail) |

## Methodology

**`bench_sustained` (TK 2.0 convention):**
500 warmup + 100 reps, 2 CUDA events wrapping all reps (avg = total / N),
L2 cache cycling via input groups, 500ms thermal cooldown between kernels, bitwise-identical random inputs. Follows the
[ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2) benchmarking convention.

**Correctness:** `correctness.py` patches the kernel into FlashInfer, then runs FlashInfer's own pytest suite at the benchmark shape/dtype.
Tolerances are FlashInfer's upstream values (unmodified).

**Reference-first execution order:** `compare.py` benchmarks FlashInfer
*before* loading the custom kernel via `torch.utils.cpp_extension.load`.
