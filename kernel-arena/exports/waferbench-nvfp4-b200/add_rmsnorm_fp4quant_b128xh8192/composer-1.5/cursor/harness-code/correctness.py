import os, sys
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

from torch.utils.cpp_extension import load

print("Compiling kernel...")
mod = load(name="add_rmsnorm_fp4quant_b128xh8192", sources=["kernel.cu"],
           extra_cuda_cflags=["-O3", "-use_fast_math"], verbose=False)
print("Done.")

import flashinfer.cute_dsl.add_rmsnorm_fp4quant as _m
import flashinfer.cute_dsl as _c
import flashinfer.norm as _n
_m.add_rmsnorm_fp4quant = mod.add_rmsnorm_fp4quant
_c.add_rmsnorm_fp4quant = mod.add_rmsnorm_fp4quant
_n.add_rmsnorm_fp4quant = mod.add_rmsnorm_fp4quant

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flashinfer_tests"))
import pytest
sys.exit(pytest.main([
    "flashinfer_tests/tests/norm/test_add_rmsnorm_fp4_quant_cute_dsl.py",
    "-v", "--tb=short",
    "-k", "test_add_rmsnorm_fp4quant_2d and 8192-128 and dtype1",
]))
