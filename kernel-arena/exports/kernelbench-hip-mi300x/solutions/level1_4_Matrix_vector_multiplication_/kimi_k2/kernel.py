import torch
import torch.nn as nn
import os

# Set compiler to hipcc for ROCm
os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# Optimized HIP kernel with float4 vectorization for max bandwidth
matvec_hip_source = """
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>

#define BLOCK_SIZE 256
#define VECTOR_SIZE 4  // Use float4 for 4x vectorization

// Helper to load float4 safely
__device__ void load_float4(const float* ptr, float& x, float& y, float& z, float& w) {
    float4 vec = *reinterpret_cast<const float4*>(ptr);
    x = vec.x;
    y = vec.y;
    z = vec.z;
    w = vec.w;
}

__global__ void matvec_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    __shared__ float shared_mem[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    float sum = 0.0f;
    const float* __restrict__ A_row = A + row * K;
    
    // Process vectorized elements with float4
    // This loads 4 elements per thread
    for (int i = tid * VECTOR_SIZE; i < K / VECTOR_SIZE * VECTOR_SIZE; i += blockDim.x * VECTOR_SIZE) {
        float4 a_vec = *reinterpret_cast<const float4*>(&A_row[i]);
        float4 b_vec = *reinterpret_cast<const float4*>(&B[i]);
        
        sum += a_vec.x * b_vec.x;
        sum += a_vec.y * b_vec.y;
        sum += a_vec.z * b_vec.z;
        sum += a_vec.w * b_vec.w;
    }
    
    // Handle remaining elements if K is not divisible by 4
    int remaining = tid + (K / VECTOR_SIZE) * VECTOR_SIZE;
    if (remaining < K) {
        sum += A_row[remaining] * B[remaining];
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Fast reduction: reduce to 32 elements (one per warp)
    if (tid < 32) {
        for (int i = tid + 32; i < BLOCK_SIZE; i += 32) {
            shared_mem[tid] += shared_mem[i];
        }
    }
    __syncthreads();
    
    // Final warp reduction with inline operations
    if (tid < 32) {
        float val = shared_mem[tid];
        // Manual unrolled reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset);
        }
        if (tid == 0) {
            atomicAdd_system(&C[row * 1 + 0], val);
        }
    }
}

torch::Tensor matvec_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    
    int M = A.size(0);
    int K = A.size(1);
    
    TORCH_CHECK(B.size(0) == K, "B dim 0 must match A dim 1");
    TORCH_CHECK(B.size(1) == 1, "B must be Kx1");
    
    auto C = torch::zeros({M, 1}, A.options());
    
    const int block_size = BLOCK_SIZE;
    const int grid_size = M;  // One block per output row
    
    hipStream_t stream = c10::hip::getCurrentHIPStream(A.device().index());
    
    hipLaunchKernelGGL(
        matvec_kernel,
        dim3(grid_size),
        dim3(block_size),
        0, // shared memory
        stream,
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        K
    );
    
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        TORCH_CHECK(false, "HIP kernel launch error: ", hipGetErrorString(error));
    }
    
    return C;
}
"""

matvec = load_inline(
    name="matvec",
    cpp_sources=matvec_hip_source,
    functions=["matvec_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matvec = matvec
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        return self.matvec.matvec_hip(A, B)

def get_inputs():
    M = 256 * 8  # 2048
    K = 131072 * 8  # 1048576
    A = torch.randn(M, K, dtype=torch.float32).cuda()
    B = torch.randn(K, 1, dtype=torch.float32).cuda()
    return [A, B]

def get_init_inputs():
    return []