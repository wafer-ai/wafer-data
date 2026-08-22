import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set compiler to hipcc
os.environ["CXX"] = "hipcc"
os.environ["ROCBLAS_TENSILE_LIB_PATH"] = "/opt/rocm/rocblas/lib/library"

bmm_source = """
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

hipblasHandle_t handle = nullptr;

torch::Tensor bmm_hip(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);

    auto C = torch::empty({batch_size, m, n}, A.options());

    if (handle == nullptr) {
        hipblasCreate(&handle);
        // Allow atomics for potentially faster kernels on MI300X
        hipblasSetAtomicsMode(handle, HIPBLAS_ATOMICS_ALLOWED);
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    long long strideA = (long long)m * k;
    long long strideB = (long long)k * n;
    long long strideC = (long long)m * n;

    // Call SgemmStridedBatched
    // Transpose logic: C^T = B^T * A^T
    // We want C (mxn). We compute C^T (nxm) in col-major.
    // Passing B as first matrix (nxk) and A as second (kxm).
    
    hipblasSgemmStridedBatched(
        handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        n, m, k,
        &alpha,
        B.data_ptr<float>(), n, strideB,
        A.data_ptr<float>(), k, strideA,
        &beta,
        C.data_ptr<float>(), n, strideC,
        batch_size
    );
    
    return C;
}
"""

bmm_ext = load_inline(
    name="bmm_ext_blas_v2",
    cpp_sources=bmm_source,
    functions=["bmm_hip"],
    extra_ldflags=["-lhipblas"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_ext = bmm_ext

    def forward(self, A, B):
        return self.bmm_ext.bmm_hip(A, B)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []
