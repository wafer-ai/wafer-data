import os
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import json
import torch
from bench import bench_sustained, compute_num_input_groups, CyclingCallable, print_results
import flashinfer
BLOCK_SIZE = 16
FP8_E4M3_MAX, FP4_E2M1_MAX = 448.0, 6.0
batch, hidden = 128, 8192

print(f"Add+RMSNorm+FP4Quant  batch={batch} hidden={hidden}")
print("=" * 60)

def rms_norm(x, w, eps=1e-6):
    xf = x.float()
    return ((xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()).to(x.dtype)

def global_scale_for(normed):
    return (FP8_E4M3_MAX * FP4_E2M1_MAX) / normed.float().abs().max().item()

input_bytes = batch * hidden * 2 * 3
n_groups = compute_num_input_groups(input_bytes)
print(f"L2 cycling: {n_groups} group(s)\n")

inputs = []
for g in range(n_groups):
    torch.manual_seed(42 + g)
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(hidden, device="cuda", dtype=torch.bfloat16).abs() + 0.1
    gs = torch.tensor([global_scale_for(rms_norm(r + x, w))],
                       dtype=torch.float32, device="cuda")
    inputs.append((x, r, w, gs))

groups_ref = [
    lambda x=x, r=r, w=w, s=gs:
        flashinfer.add_rmsnorm_fp4quant(x, r.clone(), w, global_scale=s,
                                        eps=1e-6, block_size=BLOCK_SIZE,
                                        scale_format="e4m3")
    for x, r, w, gs in inputs
]
print("Benchmarking FlashInfer reference (clean CUDA driver state) ...")
ref_results = bench_sustained(
    {"flashinfer": CyclingCallable(groups_ref)}, warmup=500, rep=100)

from torch.utils.cpp_extension import load
print("\nCompiling kernel.cu ...")
mod = load(name="add_rmsnorm_fp4quant_b128xh8192", sources=["kernel.cu"],
           extra_cuda_cflags=["-O3", "-use_fast_math"], verbose=False)
print("Done.\n")

groups_custom = [
    lambda x=x, r=r, w=w, s=gs:
        mod.add_rmsnorm_fp4quant(x, r.clone(), w, global_scale=s,
                                 eps=1e-6, block_size=BLOCK_SIZE)
    for x, r, w, gs in inputs
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
