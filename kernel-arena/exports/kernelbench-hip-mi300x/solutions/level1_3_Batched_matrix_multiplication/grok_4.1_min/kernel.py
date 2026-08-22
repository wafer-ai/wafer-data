import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void batched_matmul_kernel(const float *A, const float *B, float *C, int batch_size, int M, int N, int K, int TM, int TN, int TK) {
    int batch = blockIdx.z;
    if (batch >= batch_size) return;
    
    extern __shared__ float sdata[];
    float *shA = sdata;
    float *shB = shA + TM * TK;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TM + ty;
    int col = blockIdx.x * TN + tx;
    
    if (row >= M || col >= N) return;
    
    float acc = 0.0f;
    int num_ktile = (K + TK - 1) / TK;
    
    for (int ktile = 0; ktile < num_ktile; ++ktile) {
        int kstart = ktile * TK;
        
        // Load A tile
        int num_step_a = (TK + TN - 1) / TN;
        #pragma unroll
        for (int step = 0; step < num_step_a; ++step) {
            int k_local = step * TN + tx;
            if (k_local < TK) {
                int gk = kstart + k_local;
                if (gk < K) {
                    shA[ty * TK + k_local] = A[batch * (size_t)M * K + (size_t)row * K + gk];
                } else {
                    shA[ty * TK + k_local] = 0.0f;
                }
            }
        }
        
        // Load B tile
        int num_step_b = (TK + TM - 1) / TM;
        #pragma unroll
        for (int step = 0; step < num_step_b; ++step) {
            int k_local = step * TM + ty;
            if (k_local < TK) {
                int gk = kstart + k_local;
                if (gk < K) {
                    shB[k_local * TN + tx] = B[batch * (size_t)K * N + (size_t)gk * N + col];
                } else {
                    shB[k_local * TN + tx] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute
        for (int kk = 0; kk < TK; ++kk) {
            acc += shA[ty * TK + kk] * shB[kk * TN + tx];
        }
        
        __syncthreads();
    }
    
    C[batch * (size_t)M * N + (size_t)row * N + col] = acc;
}

torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B) {
    int64_t batch_size64 = A.size(0);
    int64_t M64 = A.size(1);
    int64_t K64 = A.size(2);
    int64_t N64 = B.size(2);
    
    int batch_size = static_cast<int>(batch_size64);
    int M = static_cast<int>(M64);
    int N = static_cast<int>(N64);
    int K = static_cast<int>(K64);
    
    torch::Tensor C = torch::zeros({batch_size64, M64, N64}, A.options());
    
    const int TM = 32;
    const int TN = 32;
    const int TK = 64;
    
    dim3 blockDim(TN, TM);
    dim3 gridDim((N + TN - 1) / TN, (M + TM - 1) / TM, batch_size);
    
    size_t shmem_bytes = ((size_t)TM * TK + (size_t)TK * TN) * sizeof(float);
    
    batched_matmul_kernel<<<gridDim, blockDim, shmem_bytes>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, M, N, K, TM, TN, TK);
    
    return C;
}
"""

batched_bmm = load_inline(
    name="batched_bmm",
    cpp_sources=cpp_source,
    functions=["batched_matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_bmm = batched_bmm

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_bmm.batched_matmul_hip(A, B)


batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []
