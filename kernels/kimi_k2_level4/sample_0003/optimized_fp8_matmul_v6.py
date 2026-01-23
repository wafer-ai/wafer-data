import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Set HIP compiler
os.environ["CXX"] = "hipcc"

# Optimized FP8 matmul kernel without complex warp operations
fp8_gemm_source = '''
#include <hip/hip_runtime.h>
#include <math.h>

// Simplified FP8 GEMM kernel optimized for MI300X
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define THREAD_M 4
#define THREAD_N 4

__global__ void fp8_gemm_mi300x(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    float fp8_max,
    int M, int K, int N
) {
    // Block indices
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int thread_m = threadIdx.x / 16;
    const int thread_n = threadIdx.x % 16;
    
    // Global coordinates
    const int global_m = block_m * BLOCK_SIZE_M + thread_m * THREAD_M;
    const int global_n = block_n * BLOCK_SIZE_N + thread_n * THREAD_N;
    
    // Accumulators
    float accum[THREAD_M][THREAD_N];
    #pragma unroll
    for(int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for(int j = 0; j < THREAD_N; j++) {
            accum[i][j] = 0.0f;
        }
    }
    
    // Main K loop
    for(int k = 0; k < K; k++) {
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++) {
            const int idx_a = (global_m + i) * K + k;
            float a_val = (global_m + i < M && k < K) ? A[idx_a] : 0.0f;
            a_val = fmaxf(-fp8_max, fminf(fp8_max, a_val));
            
            #pragma unroll
            for(int j = 0; j < THREAD_N; j++) {
                const int idx_b = k * N + (global_n + j);
                float b_val = (k < K && global_n + j < N) ? B[idx_b] : 0.0f;
                b_val = fmaxf(-fp8_max, fminf(fp8_max, b_val));
                
                accum[i][j] += a_val * b_val;
            }
        }
    }
    
    // Store results
    #pragma unroll
    for(int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for(int j = 0; j < THREAD_N; j++) {
            if(global_m + i < M && global_n + j < N) {
                C[(global_m + i) * N + (global_n + j)] = accum[i][j];
            }
        }
    }
}

// Wrapper
torch::Tensor fp8_gemm_wrapper(
    torch::Tensor A,
    torch::Tensor B,
    float fp8_max
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 grid((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 
              (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
    dim3 block(256);
    
    fp8_gemm_mi300x<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        fp8_max,
        M, K, N
    );
    
    return C;
}
'''

# Compile
fp8_gemm = load_inline(
    name="fp8_gemm_mi300x_v6",
    cpp_sources=fp8_gemm_source,
    functions=["fp8_gemm_wrapper"],
    verbose=True,
    extra_cflags=["-O3"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3
        
        # FP8 format specifications
        if use_e4m3:
            self.fp8_max = 448.0
        else:
            self.fp8_max = 57344.0
        
        # Initialize weights
        weight = torch.randn(K, N, dtype=torch.float32) * 0.02
        self.weight = nn.Parameter(weight)
        self.register_buffer('weight_t', weight.t().contiguous())
        
        # Bind kernel
        self.gemm = fp8_gemm
        
    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        amax = x.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        return scale
        
    def quantize_to_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_scaled = x * scale
        x_clamped = x_scaled.clamp(-self.fp8_max, self.fp8_max)
        return x_clamped.to(torch.float8_e4m3fn if self.use_e4m3 else torch.float8_e5m2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        batch_size, seq_len, K = x.shape
        
        # Reshape input
        x_2d = x.view(-1, self.K).contiguous().float()
        
        # Compute scales
        x_scale = self.compute_scale(x_2d)
        w_scale = self.compute_scale(self.weight_t)
        
        # Pre-scale for quantization simulation
        x_scaled = x_2d * x_scale
        w_scaled = self.weight_t * w_scale
        
        # Call custom kernel
        out = self.gemm.fp8_gemm_wrapper(x_scaled, w_scaled, self.fp8_max)
        
        # Post-scale (inverse quantization)
        inv_scale = 1.0 / (x_scale * w_scale)
        out = out * inv_scale
        
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
    return [M, K, N, use_e4m3]