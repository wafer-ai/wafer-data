import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

# One-shot fused Linear (GEMM) + (1/divisor) scaling + Bias + GELU using hipBLASLt epilogue.

cpp_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

static hipblasLtHandle_t get_handle() {
    static hipblasLtHandle_t handle = nullptr;
    static bool inited = false;
    if(!inited) {
        hipblasLtCreate(&handle);
        inited = true;
    }
    return handle;
}

// Cache for fixed shapes (KernelBench uses fixed batch/in/out)
struct CachedPlan {
    bool valid = false;
    int64_t m=0,n=0,k=0;
    hipblasLtMatmulDesc_t matmulDesc = nullptr;
    hipblasLtMatrixLayout_t layoutA = nullptr;
    hipblasLtMatrixLayout_t layoutB = nullptr;
    hipblasLtMatrixLayout_t layoutD = nullptr;
    hipblasLtMatmulAlgo_t algo;
    size_t workspaceSize = 0;
};

static CachedPlan& get_plan(int64_t m, int64_t n, int64_t k) {
    static CachedPlan plan;
    if(plan.valid && plan.m==m && plan.n==n && plan.k==k) return plan;

    if(plan.matmulDesc) hipblasLtMatmulDescDestroy(plan.matmulDesc);
    if(plan.layoutA) hipblasLtMatrixLayoutDestroy(plan.layoutA);
    if(plan.layoutB) hipblasLtMatrixLayoutDestroy(plan.layoutB);
    if(plan.layoutD) hipblasLtMatrixLayoutDestroy(plan.layoutD);

    plan = CachedPlan{};
    plan.m=m; plan.n=n; plan.k=k;

    // Column-major trick:
    // X is (batch,k) row-major => view as (k,n=batch) column-major.
    // W is (m=out,k) row-major => view as (k,m) column-major, then op(A)=T.

    hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;
    hipblasDatatype_t scaleType = HIPBLAS_R_32F;
    hipblasLtMatmulDescCreate(&plan.matmulDesc, computeType, scaleType);

    hipblasOperation_t opA = HIPBLAS_OP_T;
    hipblasOperation_t opB = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(plan.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtMatmulDescSetAttribute(plan.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
    hipblasLtMatmulDescSetAttribute(plan.matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

    // Layouts (all column-major)
    // A: (k,m) with lda=k
    // B: (k,n) with ldb=k
    // D: (m,n) with ldd=m
    hipblasLtMatrixLayoutCreate(&plan.layoutA, HIPBLAS_R_32F, k, m, k);
    hipblasLtMatrixLayoutCreate(&plan.layoutB, HIPBLAS_R_32F, k, n, k);
    hipblasLtMatrixLayoutCreate(&plan.layoutD, HIPBLAS_R_32F, m, n, m);

    hipblasLtOrder_t col = HIPBLASLT_ORDER_COL;
    hipblasLtMatrixLayoutSetAttribute(plan.layoutA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &col, sizeof(col));
    hipblasLtMatrixLayoutSetAttribute(plan.layoutB, HIPBLASLT_MATRIX_LAYOUT_ORDER, &col, sizeof(col));
    hipblasLtMatrixLayoutSetAttribute(plan.layoutD, HIPBLASLT_MATRIX_LAYOUT_ORDER, &col, sizeof(col));

    // Heuristic algorithm
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    size_t maxWorkspace = 64 * 1024 * 1024; // 64MB
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspace, sizeof(maxWorkspace));

    hipblasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    hipblasLtMatmulAlgoGetHeuristic(get_handle(), plan.matmulDesc,
                                    plan.layoutA, plan.layoutB,
                                    plan.layoutD, plan.layoutD,
                                    pref, 1, &heur, &returned);
    hipblasLtMatmulPreferenceDestroy(pref);

    TORCH_CHECK(returned > 0, "hipBLASLt heuristic failed to find an algorithm");
    plan.algo = heur.algo;
    plan.workspaceSize = heur.workspaceSize;

    plan.valid = true;
    return plan;
}

torch::Tensor linear_div_gelu_hip(torch::Tensor x, torch::Tensor w, torch::Tensor b, double divisor) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "tensors must be CUDA/HIP");
    TORCH_CHECK(x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat && b.scalar_type()==at::kFloat, "FP32 only");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous(), "contiguous only");

    // x: (batch,k), w: (m,k), b: (m)
    int64_t batch = x.size(0);
    int64_t k = x.size(1);
    int64_t m = w.size(0);
    TORCH_CHECK(w.size(1) == k, "weight shape mismatch");
    TORCH_CHECK(b.numel() == m, "bias shape mismatch");

    int64_t n = batch;

    auto out = torch::empty({batch, m}, x.options());

    auto &plan = get_plan(m, n, k);

    // Set bias pointer each call
    void* biasPtr = (void*)b.data_ptr<float>();
    hipblasLtMatmulDescSetAttribute(plan.matmulDesc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr));

    float alpha = 1.0f / (float)divisor;
    float beta = 0.0f;

    // Workspace
    torch::Tensor workspace;
    void* workspacePtr = nullptr;
    if(plan.workspaceSize > 0) {
        workspace = torch::empty({(long long)plan.workspaceSize}, x.options().dtype(torch::kUInt8));
        workspacePtr = workspace.data_ptr();
    }

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();

    // Pointer mapping using column-major views:
    // A points to w buffer (row-major m x k) viewed as col-major (k x m)
    // B points to x buffer (row-major batch x k) viewed as col-major (k x batch)
    // D points to out buffer (row-major batch x m) viewed as col-major (m x batch)
    const void* A = (const void*)w.data_ptr<float>();
    const void* B = (const void*)x.data_ptr<float>();
    void* D = (void*)out.data_ptr<float>();

    hipblasStatus_t st = hipblasLtMatmul(get_handle(),
                                        plan.matmulDesc,
                                        &alpha,
                                        A, plan.layoutA,
                                        B, plan.layoutB,
                                        &beta,
                                        D, plan.layoutD,
                                        D, plan.layoutD,
                                        &plan.algo,
                                        workspacePtr,
                                        plan.workspaceSize,
                                        stream);
    TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, "hipblasLtMatmul failed");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_div_gelu_hip", &linear_div_gelu_hip, "Fused Linear+Div+Bias+GELU (hipBLASLt)");
}
"""

ext = load_inline(
    name="linear_div_gelu_hipblaslt_ext",
    cpp_sources=cpp_source,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_include_paths=["/opt/rocm/include"],
    extra_ldflags=["-lhipblaslt"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = float(divisor)

    def forward(self, x):
        # Use fused hipBLASLt matmul epilogue (scale=1/divisor, bias, gelu)
        return ext.linear_div_gelu_hip(x.contiguous(), self.linear.weight.contiguous(), self.linear.bias.contiguous(), self.divisor)


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [input_size, output_size, divisor]
