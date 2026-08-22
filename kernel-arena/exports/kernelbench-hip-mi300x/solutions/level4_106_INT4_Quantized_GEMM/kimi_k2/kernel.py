import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set up HIP compilation
os.environ["CXX"] = "hipcc"

# Revised HIP kernel - Fixed stream API for ROCm
int4_gemm_hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Simplified fused INT4 GEMM kernel
// Each block computes a tile of the output matrix

extern "C" __global__ 
void int4_gemm_fused_simple(
    const __half* __restrict__ x,      // Input: (batch_seq_len, K)
    const uint8_t* __restrict__ w_packed, // Packed weights: (N, K/2)
    const __half* __restrict__ scales, // Scales: (N, num_groups)
    __half* __restrict__ y,            // Output: (batch_seq_len, N)
    int batch_seq_len,                 // Number of rows
    int K,                            // Inner dimension
    int N,                            // Output columns
    int num_groups,                   // K / group_size
    int group_size                    // Group size for quantization
) {
    // Block tile sizes
    const int BM = 64;    // Block rows
    const int BN = 64;    // Block cols
    
    // Thread configuration
    int tid = threadIdx.x;
    int thread_row = tid / 4;
    int thread_col = tid % 4;
    
    // Block position
    int block_row = blockIdx.x * BM;
    int block_col = blockIdx.y * BN;
    
    if (block_row >= batch_seq_len || block_col >= N) return;
    
    // Accumulator registers
    float accum[8] = {0.0f};
    
    // Loop over K dimension
    for (int k = 0; k < K; k++) {
        // Load input X
        int x_row = block_row + thread_row / 2;
        int x_col = k;
        
        if (x_row >= batch_seq_len) continue;
        
        __half x_h = x[x_row * K + x_col];
        float x_f = __half2float(x_h);
        
        if (x_f == 0.0f) continue;
        
        // Process output columns
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int out_col = block_col + thread_col * 16 + i * 8 + j * 4;
                
                if (out_col >= N) continue;
                
                // Load packed weight
                int w_row = out_col;
                int w_col_packed = k / 2;
                
                if (w_col_packed >= K/2) continue;
                
                uint8_t packed = w_packed[w_row * (K/2) + w_col_packed];
                
                // Unpack INT4
                int w_int = (k % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                
                // Get scale and dequantize
                int k_group = k / group_size;
                float scale = __half2float(scales[w_row * num_groups + k_group]);
                float w_dequant = scale * (static_cast<float>(w_int) - 8.0f);
                
                // Accumulate
                accum[i*4 + j*2 + 0] += x_f * w_dequant;
            }
        }
    }
    
    // Store results
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int out_col = block_col + thread_col * 16 + i * 8 + j * 4;
            if (out_col < N) {
                int out_row = block_row + thread_row / 2;
                if (out_row < batch_seq_len) {
                    y[out_row * N + out_col] = __float2half(accum[i*4 + j*2 + 0]);
                }
            }
        }
    }
}

// Reference kernel - grid-stride loop
extern "C" __global__ 
void int4_gemm_reference(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ scales,
    __half* __restrict__ y,
    int batch_seq_len,
    int K,
    int N,
    int num_groups,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= batch_seq_len) return;
    
    // Each thread computes one row
    for (int col = 0; col < N; col++) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; k++) {
            // Load input
            __half x_h = x[row * K + k];
            float x_f = __half2float(x_h);
            if (x_f == 0.0f) continue;
            
            // Load packed weight
            int w_col_packed = k / 2;
            uint8_t packed = w_packed[col * (K/2) + w_col_packed];
            
            // Unpack INT4
            int w_int = (k % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            
            // Dequantize with scale
            int k_group = k / group_size;
            float scale = __half2float(scales[col * num_groups + k_group]);
            float w_dequant = scale * (static_cast<float>(w_int) - 8.0f);
            
            sum += x_f * w_dequant;
        }
        
        y[row * N + col] = __float2half(sum);
    }
}

torch::Tensor int4_gemm_forward(
    torch::Tensor x,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    int K,
    int N,
    int group_size
) {
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int batch_seq_len = batch_size * seq_len;
    
    // Allocate output
    auto y = torch::zeros({batch_seq_len, N}, x.options());
    
    int num_groups = K / group_size;
    
    // Launch configuration for reference kernel
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    
    // 2D grid for better work distribution
    dim3 grid(
        (batch_seq_len + 63) / 64,
        (N + 63) / 64
    );
    
    // Use the reference kernel for correctness
    hipLaunchKernelGGL(
        int4_gemm_reference,
        grid, block, 0, 0,
        x.data_ptr<__half>(),
        weight_packed.data_ptr<uint8_t>(),
        scales.data_ptr<__half>(),
        y.data_ptr<__half>(),
        batch_seq_len, K, N, num_groups, group_size
    );
    
    return y.view({batch_size, seq_len, N});
}
"""

# Load the HIP kernel
print("Compiling INT4 GEMM kernel...")
int4_gemm_fused = load_inline(
    name="int4_gemm_fused",
    cpp_sources=int4_gemm_hip_source,
    functions=["int4_gemm_forward"],
    verbose=True,
    extra_cflags=["-O3", "-D__HIP_PLATFORM_AMD__", "--offload-arch=gfx942"],
    extra_ldflags=["-lamdhip64"],
)
print("Kernel compilation completed!")

class ModelNew(nn.Module):
    def __init__(self, K: int, N: int, group_size: int = 128):
        super().__init__()
        self.K = K
        self.N = N
        self.group_size = group_size
        self.num_groups = K // group_size
        
        assert K % group_size == 0, "K must be divisible by group_size"
        assert K % 2 == 0, "K must be even for INT4 packing"
        
        # Same initialization as original
        self.register_buffer(
            "weight_packed",
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        )
        
        self.register_buffer(
            "scales",
            torch.randn(N, self.num_groups, dtype=torch.float16).abs() * 0.1
        )
        
        # Store the custom kernel
        self.fused_gemm = int4_gemm_fused
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Convert to FP16 (as expected by kernel)
        x_fp16 = x.half()
        
        # Use the fused kernel - this fuses unpacking, dequantization, and GEMM
        out = self.fused_gemm.int4_gemm_forward(
            x_fp16,  # Pass as (batch, seq_len, K)
            self.weight_packed,
            self.scales,
            self.K,
            self.N,
            self.group_size
        )
        
        return out

# Keep the same input generation functions
def get_inputs():
    batch_size = 4
    seq_len = 2048
    K = 4096
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float)]

def get_init_inputs():
    return [4096, 11008, 128]