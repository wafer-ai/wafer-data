import os
import torch
import torch.nn as nn

# Reference model that works on MI300X (using regular matmul instead of torch._scaled_mm)
class ReferenceModel(nn.Module):
    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3
        
        if use_e4m3:
            self.fp8_dtype = torch.float8_e4m3fn
            self.fp8_max = 448.0
        else:
            self.fp8_dtype = torch.float8_e5m2
            self.fp8_max = 57344.0
        
        # Weight matrix
        self.weight = nn.Parameter(torch.randn(K, N) * 0.02)
        
    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        amax = x.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        return scale
        
    def quantize_to_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_scaled = x * scale
        x_clamped = x_scaled.clamp(-self.fp8_max, self.fp8_max)
        return x_clamped.to(self.fp8_dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        x_2d = x.view(-1, self.K)
        x_scale = self.compute_scale(x_2d)
        x_fp8 = self.quantize_to_fp8(x_2d, x_scale)
        
        w_t = self.weight.t().contiguous()
        w_scale = self.compute_scale(w_t)
        w_fp8 = self.quantize_to_fp8(w_t, w_scale)
        
        # Use regular matmul instead of torch._scaled_mm (works on MI300X)
        x_dequant = x_fp8.to(input_dtype) / x_scale
        w_dequant = w_fp8.to(input_dtype) / w_scale
        out = torch.matmul(x_dequant, w_dequant)
        
        return out.view(batch_size, seq_len, self.N)

# Optimized implementation using custom HIP kernel
from torch.utils.cpp_extension import load_inline

# Set HIP compiler
os.environ["CXX"] = "hipcc"

# Simple optimized FP8 kernel
custom_fp8_source = '''
#include <hip/hip_runtime.h>
#include <math.h>

#define BLOCK_SIZE 64

// GEMM kernel optimized for MI300X
__global__ void gemm_mi300x(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if(row < M && col < N) {
        float sum = 0.0f;
        for(int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Wrapper
torch::Tensor gemm_custom(
    torch::Tensor A,
    torch::Tensor B
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    gemm_mi300x<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}
'''

# Compile kernel
fp8_gemm_custom = load_inline(
    name="fp8_gemm_custom_v7",
    cpp_sources=custom_fp8_source,
    functions=["gemm_custom"],
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
        
        # FP8 specs
        if use_e4m3:
            self.fp8_dtype = torch.float8_e4m3fn
            self.fp8_max = 448.0
        else:
            self.fp8_dtype = torch.float8_e5m2
            self.fp8_max = 57344.0
        
        # Store weight pre-transposed
        weight = torch.randn(K, N, dtype=torch.float32) * 0.02
        self.weight = nn.Parameter(weight)
        self.register_buffer('weight_t', weight.t().contiguous())
        
        # Bind custom kernel
        self.gemm = fp8_gemm_custom
        
    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        amax = x.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        return scale
        
    def quantize_to_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_scaled = x * scale
        x_clamped = x_scaled.clamp(-self.fp8_max, self.fp8_max)
        return x_clamped.to(self.fp8_dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        batch_size, seq_len, K = x.shape
        
        # Reshape
        x_2d = x.view(-1, self.K).contiguous().float()
        
        # Compute scales
        x_scale = self.compute_scale(x_2d)
        w_scale = self.compute_scale(self.weight_t)
        
        # FP8 quantization simulation (clamp and scale)
        x_sim = x_2d * x_scale
        w_sim = self.weight_t * w_scale
        
        # Clamp to FP8 range
        x_sim = x_sim.clamp(-self.fp8_max, self.fp8_max)
        w_sim = w_sim.clamp(-self.fp8_max, self.fp8_max)
        
        # Custom GEMM
        out = self.gemm.gemm_custom(x_sim, w_sim)
        
        # Inverse scaling
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