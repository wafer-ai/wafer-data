import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use hipblaslt for optimized GEMM - more performant for large matrices
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <ATen/hip/HIPContext.h>

static hipblasLtHandle_t handle = nullptr;
static hipblasLtMatmulDesc_t matmulDesc = nullptr;
static hipblasLtMatrixLayout_t layoutA = nullptr;
static hipblasLtMatrixLayout_t layoutB = nullptr;
static hipblasLtMatrixLayout_t layoutC = nullptr;
static hipblasLtMatmulPreference_t pref = nullptr;
static bool initialized = false;

// Cached dimensions
static int cached_M = 0;
static int cached_K = 0;
static int cached_N = 0;

void cleanup_hipblaslt() {
    if (layoutA) hipblasLtMatrixLayoutDestroy(layoutA);
    if (layoutB) hipblasLtMatrixLayoutDestroy(layoutB);
    if (layoutC) hipblasLtMatrixLayoutDestroy(layoutC);
    if (matmulDesc) hipblasLtMatmulDescDestroy(matmulDesc);
    if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    layoutA = nullptr;
    layoutB = nullptr;
    layoutC = nullptr;
    matmulDesc = nullptr;
    pref = nullptr;
}

void init_hipblaslt(int M, int K, int N) {
    if (!initialized) {
        hipblasLtCreate(&handle);
        initialized = true;
    }
    
    if (cached_M != M || cached_K != K || cached_N != N) {
        cleanup_hipblaslt();
        cached_M = M;
        cached_K = K;
        cached_N = N;
        
        // Create matmul descriptor
        hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        
        // Set operation types - both non-transpose since we handle row-major
        hipblasOperation_t opA = HIPBLAS_OP_N;
        hipblasOperation_t opB = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
        
        // Create matrix layouts for C = A * B in row-major
        // In column-major: C^T = B^T * A^T
        // B^T is N x K (B viewed as column-major)
        // A^T is K x M (A viewed as column-major)
        // C^T is N x M (C viewed as column-major)
        hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_32F, N, K, N);  // B^T: N x K, ldb = N
        hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_32F, K, M, K);  // A^T: K x M, lda = K
        hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F, N, M, N);  // C^T: N x M, ldc = N
        
        // Create preference
        hipblasLtMatmulPreferenceCreate(&pref);
        size_t workspace_size = 32 * 1024 * 1024;  // 32MB workspace
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                               &workspace_size, sizeof(workspace_size));
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    init_hipblaslt(M, K, N);
    
    auto C = torch::zeros({M, N}, A.options());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Find best algorithm
    hipblasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    
    hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutB, layoutA, layoutC, layoutC,
        pref, 1, &heuristicResult, &returnedResults
    );
    
    // Allocate workspace
    void* workspace = nullptr;
    if (heuristicResult.workspaceSize > 0) {
        (void)hipMalloc(&workspace, heuristicResult.workspaceSize);
    }
    
    // Get current stream  
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Execute matmul: C^T = B^T * A^T
    hipblasLtMatmul(
        handle,
        matmulDesc,
        &alpha,
        B.data_ptr<float>(), layoutB,  // B^T
        A.data_ptr<float>(), layoutA,  // A^T
        &beta,
        C.data_ptr<float>(), layoutC,  // C^T
        C.data_ptr<float>(), layoutC,  // D = C
        &heuristicResult.algo,
        workspace,
        heuristicResult.workspaceSize,
        stream
    );
    
    if (workspace) {
        (void)hipFree(workspace);
    }
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B);
"""

matmul_module = load_inline(
    name="matmul_hip",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_hip_source,
    functions=["matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-I/opt/rocm/include"],
    extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_hip(A, B)


def get_inputs():
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
