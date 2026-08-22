import os
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import json
import torch
from bench import bench_sustained, compute_num_input_groups, CyclingCallable, print_results
import flashinfer
BLOCK_SIZE = 16
FP8_E4M3_MAX, FP4_E2M1_MAX = 448.0, 6.0
B, M, K = 8, 256, 14336

print(f"SiLU+Mul FP4 Quant  B={B} M={M} K={K}")
print("=" * 60)

def make_inputs(seed):
    torch.manual_seed(seed)
    x = torch.randn(B, M, 2 * K, device="cuda", dtype=torch.bfloat16)
    mask = torch.full((B,), M, device="cuda", dtype=torch.int32)
    ref_y = flashinfer.activation.silu_and_mul(x)
    amax = ref_y.abs().amax(dim=(1, 2)).float()
    gs = (FP8_E4M3_MAX * FP4_E2M1_MAX / amax).to(torch.float32)
    return x, mask, gs

input_bytes = B * M * 2 * K * 2
n_groups = compute_num_input_groups(input_bytes)
print(f"L2 cycling: {n_groups} group(s)\n")

inputs = []
for g in range(n_groups):
    inputs.append(make_inputs(42 + g))

groups_ref = [
    lambda x=x, m=mask, s=gs:
        flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize(x, m, s)
    for x, mask, gs in inputs
]
print("Benchmarking FlashInfer reference (clean CUDA driver state) ...")
ref_results = bench_sustained(
    {"flashinfer": CyclingCallable(groups_ref)}, warmup=500, rep=100)

from torch.utils.cpp_extension import load
print("\nCompiling kernel.cu ...")
mod = load(name="silu_mul_fp4quant_b8xm256xk14336", sources=["kernel.cu"],
           extra_cuda_cflags=["-O3", "-use_fast_math"], verbose=False)
print("Done.\n")

groups_custom = [
    lambda x=x, m=mask, s=gs:
        mod.silu_and_mul_scaled_nvfp4_experts_quantize(x, m, s)
    for x, mask, gs in inputs
]
print("Benchmarking custom kernel ...")
custom_results = bench_sustained(
    {"custom": CyclingCallable(groups_custom)}, warmup=500, rep=100)

print_results("Results", {**ref_results, **custom_results})

kernel_us = custom_results["custom"].avg_us
ref_us = ref_results["flashinfer"].avg_us
speedup = ref_us / kernel_us

print(f"\nEVAL_RESULT_JSON:{json.dumps({
    'correct': None,
    'speedup': round(speedup, 4),
    'runtime_ms': round(kernel_us / 1000, 4),
    'reference_runtime_ms': round(ref_us / 1000, 4),
    'score': round(1.0 + speedup, 4),
})}")
