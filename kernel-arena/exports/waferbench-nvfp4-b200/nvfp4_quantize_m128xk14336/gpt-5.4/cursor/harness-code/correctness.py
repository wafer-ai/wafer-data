import os, sys
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

from torch.utils.cpp_extension import load

print("Compiling kernel...")
mod = load(name="nvfp4_quantize_m128xk14336", sources=["kernel.cu"],
           extra_cuda_cflags=["-O3", "-use_fast_math"], verbose=False)
print("Done.")

import flashinfer.fp4_quantization as _fq
import flashinfer
_fq.fp4_quantize = mod.fp4_quantize
flashinfer.fp4_quantize = mod.fp4_quantize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flashinfer_tests"))
import pytest
sys.exit(pytest.main([
    "flashinfer_tests/tests/utils/test_fp4_quantize.py",
    "-v", "--tb=short",
    "-k", "test_fp4_quantization and False-False and dtype1 and shape5",
]))
