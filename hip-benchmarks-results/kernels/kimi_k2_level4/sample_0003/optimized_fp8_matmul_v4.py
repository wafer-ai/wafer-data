import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Clear cache
import shutil
if os.path.exists('/root/.cache/torch_extensions'):
    shutil.rmtree('/root/.cache/torch_extensions')

# Set HIP compiler
os.environ["CXX"] = "hipcc"
os.environ["CC"] = "hipcc"

# Simple FP8 matmul kernel
fp8_matmul_kernel_source = '''
#include <hip/hip_runtime.h>
#include <math.h>

__device__ __forceinline__ float clamp_fp8(float x, float max_val) {
    return fminf(fmaxf(x, -max_val), max_val);
}

// Global max reduction
__global__ void compute_global_max(const float* x, float* max_out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    
    float local_max = 0.0f;
    for(int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(x[i]));
    }
    
    sdata[tid] = local_max;
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        max_out[0] = fmaxf(sdata[0], 1e-12f);
    }
}

// Main FP8 matmul kernel
__global__ void fp8_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    float scale_A,
    float scale_B,
    float fp8_max,
    int M, int K, int N
) {
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int tm = threadIdx.x / 4;
    const int tn = threadIdx.x % 4;
    
    const int M_offset = bm * 128 + tm * 4;
    const int N_offset = bn * 128 + tn * 4;
    
    float accum[4] = {0.0f};
    
    for(int k = 0; k < K; k++) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if((M_offset + 0) < M && k < K) a0 = A[(M_offset + 0) * K + k];
        if((M_offset + 1) < M && k < K) a1 = A[(M_offset + 1) * K + k];
        if((M_offset + 2) < M && k < K) a2 = A[(M_offset + 2) * K + k];
        if((M_offset + 3) < M && k < K) a3 = A[(M_offset + 3) * K + k];
        
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
        if((N_offset + 0) < N && k < K) b0 = B[k * N + (N_offset + 0)];
        if((N_offset + 1) < N && k < K) b1 = B[k * N + (N_offset + 1)];
        if((N_offset + 2) < N && k < K) b2 = B[k * N + (N_offset + 2)];
        if((N_offset + 3) < N && k < K) b3 = B[k * N + (N_offset + 3)];
        
        // Quantize and accumulate
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            float a_val = (i == 0) ? a0 : ((i == 1) ? a1 : ((i == 2) ? a2 : a3));
            a_val = a_val * scale_A;
            a_val = clamp_fp8(a_val, fp8_max);
            
            #pragma unroll
            for(int j = 0; j < 4; j++) {
                float b_val = (j == 0) ? b0 : ((j == 1) ? b1 : ((j == 2) ? b2 : b3));
                b_val = b_val * scale_B;
                b_val = clamp_fp8(b_val, fp8_max);
                
                accum[j] += a_val * b_val;
            }
        }
    }
    
    #pragma unroll
    for(int j = 0; j < 4; j++) {
        if((M_offset + tm) < M && (N_offset + j) < N) {
            C[(M_offset + tm) * N + (N_offset + j)] = accum[j];
        }
    }
}

// Main wrapper
torch::Tensor fp8_matmul_kimi(
    torch::Tensor input,
    torch::Tensor weight_t,
    float fp8_max
) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight_t.size(0);
    
    // Compute scales
    auto x_scale = torch::zeros({1}, torch::dtype(torch::kFloat32).device(input.device()));
    auto w_scale = torch::zeros({1}, torch::dtype(torch::kFloat32).device(input.device()));
    
    compute_global_max<<<1, 256>>>(input.data_ptr<float>(), x_scale.data_ptr<float>(), M * K);
    compute_global_max<<<1, 256>>>(weight_t.data_ptr<float>(), w_scale.data_ptr<float>(), K * N);
    
    // Launch GEMM (hardcode max=448.0 for E4M3)
    dim3 grid((M + 128 - 1) / 128, (N + 128 - 1) / 128);
    dim3 block(256);
    
    auto output = torch::zeros({M, N}, input.options());
    
    fp8_gemm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        output.data_ptr<float>(),
        x_scale.item<float>(),
        w_scale.item<float>(),
        fp8_max,
        M, K, N
    );
    
    float inv_scale = 1.0f / (x_scale.item<float>() * w_scale.item<float>());
    output = output * inv_scale;
    
    return output;
}
'''

# Compile kernel
fp8_matmul_kimi = load_inline(
    name="fp8_matmul_kimi_v4",
    cpp_sources=fp8_matmul_kernel_source,
    functions=["fp8_matmul_kimi"],
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
        
        # Initialize weight
        weight = torch.randn(K, N) * 0.02
        self.weight = nn.Parameter(weight)
        # Pre-transpose weight for efficiency
        self.register_buffer('weight_t', weight.t().contiguous())
        
        # Bind custom kernel
        self.kimigemm = fp8_matmul_kimi
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Reshape input
        x_2d = x.view(-1, self.K).contiguous()
        
        # Call custom FP8 matmul
        out = self.kimigemm.fp8_matmul_kimi(x_2d, self.weight_t, self.fp8_max)
        
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