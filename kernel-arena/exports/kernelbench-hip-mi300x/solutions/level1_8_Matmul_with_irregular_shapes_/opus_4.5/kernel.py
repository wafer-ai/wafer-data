import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use hipBLASLt for high-performance matrix multiplication with better tuning
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <ATen/hip/HIPContext.h>
#include <iostream>

static hipblasLtHandle_t ltHandle = nullptr;
static bool initialized = false;

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    if (!initialized) {
        hipblasLtCreate(&ltHandle);
        initialized = true;
    }
    
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();
    
    // Create operation descriptors
    hipblasLtMatmulDesc_t matmulDesc;
    hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    
    hipblasOperation_t opA = HIPBLAS_OP_N;
    hipblasOperation_t opB = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB));
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA));
    
    // Create matrix layouts (column-major layout)
    // For row-major C = A * B: we do C^T = B^T * A^T in column-major
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    
    // B layout: K x N in row-major => N x K in column-major (transposed)
    hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_32F, N, K, N);
    // A layout: M x K in row-major => K x M in column-major (transposed)
    hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_32F, K, M, K);
    // C layout: M x N in row-major => N x M in column-major
    hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F, N, M, N);
    
    // Create preference for algorithm selection
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulPreferenceCreate(&preference);
    size_t workspaceSize = 32 * 1024 * 1024;  // 32 MB workspace
    hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    
    // Get algorithms
    hipblasLtMatmulHeuristicResult_t heuristicResult[8];
    int returnedAlgoCount;
    hipblasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC, preference, 8, heuristicResult, &returnedAlgoCount);
    
    if (returnedAlgoCount > 0) {
        void* workspace = nullptr;
        if (heuristicResult[0].workspaceSize > 0) {
            hipMalloc(&workspace, heuristicResult[0].workspaceSize);
        }
        
        hipblasLtMatmul(
            ltHandle,
            matmulDesc,
            &alpha,
            B.data_ptr<float>(), layoutA,
            A.data_ptr<float>(), layoutB,
            &beta,
            C.data_ptr<float>(), layoutC,
            C.data_ptr<float>(), layoutC,
            &heuristicResult[0].algo,
            workspace,
            heuristicResult[0].workspaceSize,
            stream
        );
        
        if (workspace) {
            hipFree(workspace);
        }
    }
    
    // Cleanup
    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtMatmulDescDestroy(matmulDesc);
    
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
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lhipblaslt"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_hip(A, B)


def get_inputs():
    M = 8205
    K = 2949
    N = 5921
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
