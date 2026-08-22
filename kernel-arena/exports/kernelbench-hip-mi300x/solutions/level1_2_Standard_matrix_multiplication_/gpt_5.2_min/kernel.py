import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

cpp_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP32(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

struct LtGemmCache {
    hipblasLtHandle_t lt = nullptr;
    hipblasLtMatmulDesc_t opDesc = nullptr;
    hipblasLtMatrixLayout_t aLayout = nullptr;
    hipblasLtMatrixLayout_t bLayout = nullptr;
    hipblasLtMatrixLayout_t cLayout = nullptr;
    hipblasLtMatmulAlgo_t algo;
    void* workspace = nullptr;
    size_t workspaceSize = 0;
    int64_t lastM = -1, lastN = -1, lastK = -1;
    bool initialized = false;
};

static LtGemmCache& cache() {
    static LtGemmCache c;
    return c;
}

static void ensure_initialized(int64_t M, int64_t N, int64_t K) {
    // We compute C_cm (N x M) = B_cm (N x K) * A_cm (K x M)
    // so lt matmul uses m=N, n=M, k=K.
    auto &c = cache();
    if (c.initialized && c.lastM == M && c.lastN == N && c.lastK == K) return;

    // Destroy previous (if any)
    if (c.workspace) { hipFree(c.workspace); c.workspace = nullptr; }
    if (c.cLayout) { hipblasLtMatrixLayoutDestroy(c.cLayout); c.cLayout = nullptr; }
    if (c.bLayout) { hipblasLtMatrixLayoutDestroy(c.bLayout); c.bLayout = nullptr; }
    if (c.aLayout) { hipblasLtMatrixLayoutDestroy(c.aLayout); c.aLayout = nullptr; }
    if (c.opDesc) { hipblasLtMatmulDescDestroy(c.opDesc); c.opDesc = nullptr; }
    if (!c.lt) hipblasLtCreate(&c.lt);

    hipblasLtMatmulDescCreate(&c.opDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);

    // No transpose for the column-major views
    hipblasOperation_t trans = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(c.opDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans));
    hipblasLtMatmulDescSetAttribute(c.opDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans));

    const int64_t m = N;
    const int64_t n = M;
    const int64_t k = K;

    // Column-major layouts
    const int64_t lda = m; // B_cm: (m x k)
    const int64_t ldb = k; // A_cm: (k x n)
    const int64_t ldc = m; // C_cm: (m x n)

    hipblasLtMatrixLayoutCreate(&c.aLayout, HIP_R_32F, m, k, lda);
    hipblasLtMatrixLayoutCreate(&c.bLayout, HIP_R_32F, k, n, ldb);
    hipblasLtMatrixLayoutCreate(&c.cLayout, HIP_R_32F, m, n, ldc);

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);

    // Allow some workspace for better kernels
    c.workspaceSize = 32 * 1024 * 1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.workspaceSize, sizeof(c.workspaceSize));

    hipblasLtMatmulHeuristicResult_t heur;
    int returned = 0;

    auto st = hipblasLtMatmulAlgoGetHeuristic(
        c.lt,
        c.opDesc,
        c.aLayout,
        c.bLayout,
        c.cLayout,
        c.cLayout,
        pref,
        1,
        &heur,
        &returned);

    hipblasLtMatmulPreferenceDestroy(pref);

    TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS && returned > 0, "hipblasLtMatmulAlgoGetHeuristic failed");
    c.algo = heur.algo;

    // Allocate workspace if needed by selected algo
    c.workspaceSize = heur.workspaceSize;
    if (c.workspaceSize > 0) {
        hipError_t e = hipMalloc(&c.workspace, c.workspaceSize);
        TORCH_CHECK(e == hipSuccess, "hipMalloc(workspace) failed");
    }

    c.lastM = M; c.lastN = N; c.lastK = K;
    c.initialized = true;
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_FP32(A);
    CHECK_FP32(B);

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    ensure_initialized(M, N, K);
    auto &c = cache();

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major trick as before:
    // C_cm (N x M) = B_cm (N x K) * A_cm (K x M)
    const void* A_ptr = (const void*)B.data_ptr<float>(); // A operand in Lt matmul
    const void* B_ptr = (const void*)A.data_ptr<float>(); // B operand in Lt matmul
    void* C_ptr = (void*)C.data_ptr<float>();

    auto st = hipblasLtMatmul(
        c.lt,
        c.opDesc,
        &alpha,
        A_ptr, c.aLayout,
        B_ptr, c.bLayout,
        &beta,
        C_ptr, c.cLayout,
        C_ptr, c.cLayout,
        &c.algo,
        c.workspace,
        c.workspaceSize,
        stream);

    TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, "hipblasLtMatmul failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_hip", &matmul_hip, "hipBLASLt FP32 matmul (row-major wrapper)");
}
"""

matmul_ext = load_inline(
    name="matmul_hipblaslt_ext",
    cpp_sources=cpp_src,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.dtype != torch.float32:
            A = A.float()
        if B.dtype != torch.float32:
            B = B.float()
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        return matmul_ext.matmul_hip(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []
