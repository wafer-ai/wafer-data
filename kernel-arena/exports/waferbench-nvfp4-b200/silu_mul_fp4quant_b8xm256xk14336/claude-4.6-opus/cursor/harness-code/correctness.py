import os, sys
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

from torch.utils.cpp_extension import load

print("Compiling kernel...")
mod = load(name="silu_mul_fp4quant_b8xm256xk14336", sources=["kernel.cu"],
           extra_cuda_cflags=["-O3", "-use_fast_math"], verbose=False)
print("Done.")

import flashinfer.activation as _act
import flashinfer
_act.silu_and_mul_scaled_nvfp4_experts_quantize = mod.silu_and_mul_scaled_nvfp4_experts_quantize
flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize = mod.silu_and_mul_scaled_nvfp4_experts_quantize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flashinfer_tests"))
import pytest
sys.exit(pytest.main([
    "flashinfer_tests/tests/utils/test_fp4_quantize.py",
    "-v", "--tb=short",
    "-k", "test_silu_and_mul_scaled_nvfp4_experts_quantize and batch_shape5 and dtype1",
]))
