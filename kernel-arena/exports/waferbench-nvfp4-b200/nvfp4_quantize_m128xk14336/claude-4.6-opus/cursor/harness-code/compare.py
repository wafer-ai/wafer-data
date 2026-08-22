import os
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import json
import torch
from bench import bench_sustained, compute_num_input_groups, CyclingCallable, print_results
from flashinfer import fp4_quantize
BLOCK_SIZE = 16
FP8_E4M3_MAX, FP4_E2M1_MAX = 448.0, 6.0
M, K = 128, 14336

print(f"FP4 Quantize  M={M} K={K}")
print("=" * 60)

def global_scale_for(x):
    return (FP8_E4M3_MAX * FP4_E2M1_MAX) / x.float().abs().nan_to_num().max().item()

input_bytes = M * K * 2
n_groups = compute_num_input_groups(input_bytes)
print(f"L2 cycling: {n_groups} group(s)\n")

inputs = []
for g in range(n_groups):
    torch.manual_seed(42 + g)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    gs = torch.tensor(global_scale_for(x), dtype=torch.float32, device="cuda")
    inputs.append((x, gs))

groups_ref = [
    lambda x=x, s=gs:
        fp4_quantize(x, s, BLOCK_SIZE, False, False)
    for x, gs in inputs
]
print("Benchmarking FlashInfer reference (clean CUDA driver state) ...")
ref_results = bench_sustained(
    {"flashinfer": CyclingCallable(groups_ref)}, warmup=500, rep=100)

from torch.utils.cpp_extension import load
print("\nCompiling kernel.cu ...")
mod = load(name="nvfp4_quantize_m128xk14336", sources=["kernel.cu"],
           extra_cuda_cflags=["-O3", "-use_fast_math"], verbose=False)
print("Done.\n")

groups_custom = [
    lambda x=x, s=gs:
        mod.fp4_quantize(x, global_scale=s, sf_vec_size=BLOCK_SIZE,
                         sf_use_ue8m0=False, is_sf_swizzled_layout=False)
    for x, gs in inputs
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
