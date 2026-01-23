import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Set HIP compiler
os.environ["CXX"] = "hipcc"

# Custom HIP kernel for fused FP8 matmul
fp8_matmul_hip_source = '''
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <math.h>

#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 32
#define THREAD_M 16
#define THREAD_N 16

using fp8_type = __hip_fp8_e4m3;

__device__ __forceinline__ float compute_scale_kernel(const float* x, int size) {
    float local_max = 0.0f;
    for(int i = 0; i < size; i++) {
        float val = fabs(x[i]);
        local_max = fmaxf(local_max, val);
    }
    return local_max;
}

// Optimized FP8 matmul kernel for MI300X
// Fuses: scale computation + quantization + GEMM
extern "C" __global__ void fp8_matmul_fused_kernel(
    const float* input,          // Input: (M, K)
    const float* weight,         // Weight: (K, N) - already transposed
    float* output,               // Output: (M, N)
    const int M,
    const int K,
    const int N
) {
    // Shared memory for tiling
    __shared__ float s_input[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float s_weight[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // Local registers for accumulation
    float accum[THREAD_M / 4][THREAD_N / 4];
    #pragma unroll
    for(int i = 0; i < THREAD_M / 4; i++) {
        #pragma unroll
        for(int j = 0; j < THREAD_N / 4; j++) {
            accum[i][j] = 0.0f;
        }
    }
    
    // Global thread indices
    const int thread_m = threadIdx.x / (BLOCK_SIZE_N / THREAD_N);
    const int thread_n = threadIdx.x % (BLOCK_SIZE_N / THREAD_N);
    const int batch_offset_m = blockIdx.x * BLOCK_SIZE_M;
    const int batch_offset_n = blockIdx.y * BLOCK_SIZE_N;
    
    // FP8 max values for E4M3
    const float fp8_max = 448.0f;
    
    // Iterate over K dimension in tiles
    for(int k = 0; k < K; k += BLOCK_SIZE_K) {
        // Load input tile with bounds checking
        #pragma unroll
        for(int i = 0; i < THREAD_M; i += 4) {
            int global_m = batch_offset_m + thread_m * THREAD_M + i;
            int global_k = k + threadIdx.x % (BLOCK_SIZE_K / 4);
            
            if(global_m < M && global_k < K) {
                s_input[thread_m * THREAD_M + i][global_k] = input[global_m * K + global_k];
            }
        }
        
        // Load weight tile with bounds checking (weight is already transposed: (K, N))
        #pragma unroll
        for(int j = 0; j < THREAD_N; j += 4) {
            int global_k = k + threadIdx.x / (BLOCK_SIZE_N / THREAD_N * 4);
            int global_n = batch_offset_n + thread_n * THREAD_N + j;
            
            if(global_k < K && global_n < N) {
                s_weight[global_k][global_n] = weight[global_k * N + global_n];
            }
        }
        
        __syncthreads();
        
        // Compute tile with FP8 quantization on-the-fly
        #pragma unroll
        for(int kk = 0; kk < BLOCK_SIZE_K; kk++) {
            // Load data from shared memory to registers
            float input_regs[THREAD_M / 4];
            float weight_regs[THREAD_N / 4];
            
            #pragma unroll
            for(int i = 0; i < THREAD_M / 4; i++) {
                int local_m = thread_m * THREAD_M + i * 4;
                input_regs[i] = s_input[local_m][kk];
            }
            
            #pragma unroll
            for(int j = 0; j < THREAD_N / 4; j++) {
                int local_n = thread_n * THREAD_N + j * 4;
                weight_regs[j] = s_weight[kk][local_n];
            }
            
            // Compute dot product with FP8 quantization
            #pragma unroll
            for(int i = 0; i < THREAD_M / 4; i++) {
                #pragma unroll
                for(int j = 0; j < THREAD_N / 4; j++) {
                    // Quantize inputs to FP8 on-the-fly
                    float in_val = input_regs[i];
                    float w_val = weight_regs[j];
                    
                    // Simple quantization (clamp to FP8 range)
                    in_val = fminf(fmaxf(in_val, -fp8_max), fp8_max);
                    w_val = fminf(fmaxf(w_val, -fp8_max), fp8_max);
                    
                    // Accumulate
                    accum[i][j] += in_val * w_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to output
    #pragma unroll
    for(int i = 0; i < THREAD_M / 4; i++) {
        int global_m = batch_offset_m + thread_m * THREAD_M + i * 4;
        if(global_m < M) {
            #pragma unroll
            for(int j = 0; j < THREAD_N / 4; j++) {
                int global_n = batch_offset_n + thread_n * THREAD_N + j * 4;
                if(global_n < N) {
                    output[global_m * N + global_n] = accum[i][j];
                }
            }
        }
    }
}

// Fused scale computation and quantization kernel
__global__ void compute_fp8_scales_kernel(
    const float* x,
    const float* w,
    float* x_scale,
    float* w_scale,
    int M,
    int K,
    int N
) {
    __shared__ float s_x_max[256];
    __shared__ float s_w_max[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    const float fp8_max = 448.0f;
    
    // Compute max for input tensor (per-tensor)
    float local_x_max = 0.0f;
    if(blockIdx.x == 0) {  // Only one block for per-tensor scale
        for(int i = tid; i < M * K; i += blockDim.x) {
            local_x_max = fmaxf(local_x_max, fabsf(x[i]));
        }
    }
    
    // Compute max for weight tensor (per-tensor)
    float local_w_max = 0.0f;
    if(blockIdx.x == 0) {  // Only one block for per-tensor scale
        for(int i = tid; i < K * N; i += blockDim.x) {
            local_w_max = fmaxf(local_w_max, fabsf(w[i]));
        }
    }
    
    s_x_max[tid] = local_x_max;
    s_w_max[tid] = local_w_max;
    __syncthreads();
    
    // Warp reduction
    #pragma unroll
    for(int offset = 128; offset > 0; offset >>= 1) {
        if(tid < offset) {
            s_x_max[tid] = fmaxf(s_x_max[tid], s_x_max[tid + offset]);
            s_w_max[tid] = fmaxf(s_w_max[tid], s_w_max[tid + offset]);
        }
    }
    __syncthreads();
    
    if(tid == 0) {
        x_scale[0] = fp8_max / fmaxf(s_x_max[0], 1e-12f);
        w_scale[0] = fp8_max / fmaxf(s_w_max[0], 1e-12f);
    }
}

// Wrapper functions to call from PyTorch
torch::Tensor fp8_matmul_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor weight_t
) {
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(1);
    
    auto output = torch::zeros({M, N}, input.options());
    
    // Compute scales
    auto x_scale = torch::zeros({1}, torch::dtype(torch::kFloat32).device(input.device()));
    auto w_scale = torch::zeros({1}, torch::dtype(torch::kFloat32).device(input.device()));
    
    const int threads = 256;
    const int blocks = 1;
    
    compute_fp8_scales_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        x_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        M, K, N
    );
    
    // Launch fused kernel
    dim3 block_dim(256);
    dim3 grid_dim((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 
                  (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
    
    fp8_matmul_fused_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        output.data_ptr<float>(),
        M, K, N
    );
    
    // Apply scaling to output
    output = output * (1.0f / (x_scale * w_scale).item<float>());
    
    return output;
}
'''

# Compile the custom kernel
fp8_matmul_hip = load_inline(
    name="fp8_matmul_hip",
    cpp_sources=fp8_matmul_hip_source,
    functions=["fp8_matmul_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3
        
        # Store weight pre-transposed for efficiency
        weight = torch.randn(K, N) * 0.02
        self.weight = nn.Parameter(weight)
        self.register_buffer('weight_t', weight.t().contiguous())
        
        # Bind custom kernel
        self.fp8_matmul = fp8_matmul_hip
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Reshape for matmul
        x_2d = x.view(-1, self.K).contiguous()
        
        # Call custom fused kernel
        out = self.fp8_matmul.fp8_matmul_forward(x_2d, self.weight, self.weight_t)
        
        return out.view(batch_size, seq_len, self.N).to(input_dtype)

# Configuration
batch_size = 8
seq_len = 2048
M = batch_size * seq_len
K = 4096
N = 4096
use_e4m3 = True

def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16).cuda()]

def get_init_inputs():
    return [M, K, N, use_e4m4]