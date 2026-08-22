import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: Softmax (across channel dimension) + 2x MaxPool
# Combines three operations into one kernel to reduce memory traffic
#
# Algorithm (for each output position):
# 1. Read 4x4x4 region for each channel
# 2. Apply softmax across channels at each 2x2x2 sub-region
# 3. Take max across the 2x2x2 sub-regions (first pool)
# 4. Take max of the four results (second pool)
softmax_pool_fused_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>
#include <float.h>

__global__ void softmax_double_pool_kernel(
    const float* input,  // Shape: [B, C, D1, H1, W1]
    float* output,       // Shape: [B, C, D2, H2, W2]
    int batch_size,
    int channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width
) {
    // Global thread position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_depth * out_height * out_width;
    
    if (idx >= total_outputs) return;
    
    // Decode output position
    int w = idx % out_width;
    int idx_flat = idx / out_width;
    int h = idx_flat % out_height;
    idx_flat /= out_height;
    int d = idx_flat % out_depth;
    int b = idx_flat / out_depth;
    
    // Shared memory: store input region for this position
    // Need channels * 64 values (4x4x4 region per channel)
    extern __shared__ float shared_mem[];
    float* region = shared_mem;
    
    // Base input coordinates
    int base_d = d * 4;
    int base_h = h * 4;
    int base_w = w * 4;
    
    // Load 4x4x4 region for each channel
    for (int c = 0; c < channels; c++) {
        for (int ld = 0; ld < 4; ld++) {
            int id = base_d + ld;
            if (id >= in_depth) {
                // Pad with -inf for maxpool
                for (int lh = 0; lh < 4; lh++) {
                    for (int lw = 0; lw < 4; lw++) {
                        region[c * 64 + ld * 16 + lh * 4 + lw] = -FLT_MAX;
                    }
                }
                continue;
            }
            
            for (int lh = 0; lh < 4; lh++) {
                int ih = base_h + lh;
                row_loop_start:;
                if (ih >= in_height) {
                    for (int lw = 0; lw < 4; lw++) {
                        region[c * 64 + ld * 16 + lh * 4 + lw] = -FLT_MAX;
                    }
                    continue;
                }
                
                for (int lw = 0; lw < 4; lw++) {
                    int iw = base_w + lw;
                    if (iw < in_width) {
                        int in_idx = ((b * channels + c) * in_depth + id) * in_height + ih;
                        in_idx = in_idx * in_width + iw;
                        region[c * 64 + ld * 16 + lh * 4 + lw] = input[in_idx];
                    } else {
                        region[c * 64 + ld * 16 + lh * 4 + lw] = -FLT_MAX;
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Processing: Softmax across channels, then 2x maxpool
    // For each output position:
    // - First pool: 2x2x2 -> 4 results (4 quadrants)
    // - Second pool: max of 4 results -> 1 per channel
    
    for (int c = 0; c < channels; c++) {
        // First maxpool: process 4 quadrants of 2x2x2
        float pool_results[4];  // Results from first pool
        
        for (int q = 0; q < 4; q++) {
            int q_d = q / 2;    // 0 or 1
            int q_h = q % 2;    // 0 or 1
            
            float max_val = -FLT_MAX;
            
            for (int ld = 0; ld < 2; ld++) {
                int sd = q_d * 2 + ld;  // Source depth
                for (int lh = 0; lh < 2; lh++) {
                    int sh = q_h * 2 + lh;  // Source height
                    for (int lw = 0; lw < 2; lw++) {
                        int sw = lw;  // Source width (0 or 1 for left quadrant)
                        if (q == 1 || q == 3) sw += 2;  // Add 2 for right quadrant
                        
                        float val = region[c * 64 + sd * 16 + sh * 4 + sw];
                        if (val > max_val) max_val = val;
                    }
                }
            }
            
            pool_results[q] = max_val;
        }
        
        // Second pool: max of 4 results
        float pooled_val = -FLT_MAX;
        for (int q = 0; q < 4; q++) {
            if (pool_results[q] > pooled_val) pooled_val = pool_results[q];
        }
        
        // Now we have the pooled value for this channel at this position
        // Wait - we need to apply softmax ACROSS CHANNELS, then pool!
        // But I'm already pooled...
        // 
        // The issue is: we need softmax on the 4x4x4xCHANNEL values, then pool
        // But that's expensive.
        //
        // Correct approach:
        // For each 2x2x2 region (16 regions total for 4x4x4):
        //   - Collect all channel values at positions in that region
        //   - Apply softmax across channels
        //   - Take max of softmax results (this gives us first pool output)
        // - Take max of the 4 results (second pool)
        
        // Let's retry with correct ordering
    }
    
    // Correct implementation:
    // For each of the 4 quadrants (2x2x2 regions), pool with softmax
    
    float quadrant_results[4];  // After first pool (after softmax)
    
    for (int q = 0; q < 4; q++) {
        int q_d = q / 2;    // 0 or 1
        int q_h = q % 2;    // 0 or 1
        
        // For this quadrant, we need to:
        // 1. Find values from all channels at each position in 2x2x2 region
        // 2. Apply softmax at each position across channels
        // 3. Take the max of the softmax values (maxpool after softmax)
        
        float max_softmax = -FLT_MAX;
        
        for (int ld = 0; ld < 2; ld++) {
            for (int lh = 0; lh < 2; lh++) {
                for (int lw = 0; lw < 2; lw++) {
                    int sd = q_d * 2 + ld;
                    int sh = q_h * 2 + lh;
                    int sw = lw;
                    if (q == 1 || q == 3) sw += 2;
                    
                    // Collect values from all channels at this position
                    float values[16];  // Max 16 channels
                    for (int c = 0; c < channels; c++) {
                        values[c] = region[c * 64 + sd * 16 + sh * 4 + sw];
                    }
                    
                    // Find max for softmax stability
                    float local_max = -FLT_MAX;
                    for (int c = 0; c < channels; c++) {
                        if (values[c] > local_max) local_max = values[c];
                    }
                    
                    // Compute exp sum
                    float exp_sum = 0.0f;
                    for (int c = 0; c < channels; c++) {
                        exp_sum += expf(values[c] - local_max);
                    }
                    
                    // Find maximum softmax value (the pooling)
                    for (int c = 0; c < channels; c++) {
                        float softmax_val = expf(values[c] - local_max) / exp_sum;
                        if (softmax_val > max_softmax) max_softmax = softmax_val;
                    }
                }
            }
        }
        
        quadrant_results[q] = max_softmax;
    }
    
    // Second pool: max of quadrant results
    float final_results[16];  // We need to store per channel after second pool?
    // Wait, we lost channel info...
    
    // The issue: After softmax + maxpool, we still have CHANNELS number of outputs
    // But maxpool reduces spatial dims, NOT channels!
    // My algorithm needs to preserve channels throughout.
    
    // Let me restart: Read the full 4x4x4xCHANNEL tensor
    // Then apply softmax across channels (preserves channel dim)
    // Then apply maxpool 2x (reduces spatial dims only)
    
    // Actually, let's do it position by position:
    // For each channel c:
    //   - Perform softmax at ALL 64 positions in 4x4x4 region
    //   - This gives us softmax-normalized channel values at each position
    //   - Then maxpool those 64 values down to 1 (first pool: 4, then second: 1)
    //
    // But maxpool should happen across spatial dims only!
    // So for each channel c, we maxpool across the 4x4x4 values (but they're already softmax-normalized)
    
    // Store softmax values in shared memory
    for (int ld = 0; ld < 4; ld++) {
        for (int lh = 0; lh < 4; lh++) {
            for (int lw = 0; lw < 4; lw++) {
                int base_pos = ld * 16 + lh * 4 + lw;
                
                // Find max across channels
                float max_val = -FLT_MAX;
                for (int c = 0; c < channels; c++) {
                    float val = region[c * 64 + base_pos];
                    if (val > max_val) max_val = val;
                }
                
                // Compute exp sum
                float exp_sum = 0.0f;
                for (int c = 0; c < channels; c++) {
                    exp_sum += expf(region[c * 64 + base_pos] - max_val);
                }
                
                // Store softmax values
                for (int c = 0; c < channels; c++) {
                    region[c * 64 + base_pos] = expf(region[c * 64 + base_pos] - max_val) / exp_sum;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Now apply 2x maxpool for each channel
    for (int c = 0; c < channels; c++) {
        // First pool: 4x4x4 -> 2x2x2
        float pool1[8];
        for (int i = 0; i < 8; i++) {
            int pd = i / 4;     // 0 or 1
            int ph = (i / 2) % 2; // 0 or 1
            int pw = i % 2;     // 0 or 1
            
            float max_val = -FLT_MAX;
            for (int ld = 0; ld < 2; ld++) {
                int sd = pd * 2 + ld;
                for (int lh = 0; lh < 2; lh++) {
                    int sh = ph * 2 + lh;
                    for (int lw = 0; lw < 2; lw++) {
                        int sw = pw * 2 + lw;
                        float val = region[c * 64 + sd * 16 + sh * 4 + sw];
                        if (val > max_val) max_val = val;
                    }
                }
            }
            pool1[i] = max_val;
        }
        
        // Second pool: 2x2x2 -> 1
        float final_val = -FLT_MAX;
        for (int i = 0; i < 8; i++) {
            if (pool1[i] > final_val) final_val = pool1[i];
        }
        
        // Write output
        int out_idx = ((b * channels + c) * out_depth + d) * out_height + h;
        out_idx = out_idx * out_width + w;
        output[out_idx] = final_val;
    }
}

torch::Tensor softmax_double_pool_hip(torch::Tensor input, int pool_kernel_size) {
    // Input shape: [B, C, D, H, W]
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    // Compute output size based on pooling operations
    // Using floor division (ceil_mode=False default)
    int mid_depth = in_depth / pool_kernel_size;
    int mid_height = in_height / pool_kernel_size;
    int mid_width = in_width / pool_kernel_size;
    
    int out_depth = mid_depth / pool_kernel_size;
    int out_height = mid_height / pool_kernel_size;
    int out_width = mid_width / pool_kernel_size;
    
    int total_outputs = batch_size * out_depth * out_height * out_width;
    
    // Create output tensor on same device as input
    auto output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, 
                               input.options());
    
    // Kernel launch
    const int block_size = 64;  // Reduced to fit more shared memory
    const int num_blocks = (total_outputs + block_size - 1) / block_size;
    int shared_size = channels * 64 * sizeof(float);  // channels * 4*4*4 region
    
    softmax_double_pool_kernel<<<num_blocks, block_size, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl;
    }
    
    return output;
}
"""

softmax_pool_fused = load_inline(
    name="softmax_pool_fused",
    cpp_sources=softmax_pool_fused_cpp_source,
    functions=["softmax_double_pool_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    Uses fused kernel for softmax+2xmaxpool to reduce memory transfers.
    """
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, pool_kernel_size=2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.softmax_pool_fused = softmax_pool_fused
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = self.conv(x)
        # Fused softmax + 2x maxpool for efficiency
        x = self.softmax_pool_fused.softmax_double_pool_hip(x, self.pool_kernel_size)
        return x