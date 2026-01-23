
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <algorithm>

#define WARP_SIZE 64

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float temp = __shfl_down(val, offset, WARP_SIZE);
        if (temp > val) val = temp;
    }
    return val;
}

__global__ void max_reduction_case1_kernel(const float* __restrict__ input, float* __restrict__ output, int A, int B, int C) {
    int idx = blockIdx.x * (int)blockDim.x + threadIdx.x;
    int num_outputs = A * C;
    if (idx < num_outputs) {
        int a = idx / C;
        int c = idx % C;
        const float* input_ptr = input + (a * B) * C + c;
        float max_val = -3.402823466e+38f; // -FLT_MAX
        for (int b = 0; b < B; ++b) {
            float val = *input_ptr;
            if (val > max_val) {
                max_val = val;
            }
            input_ptr += C;
        }
        output[idx] = max_val;
    }
}

__global__ void max_reduction_case2_kernel(const float* __restrict__ input, float* __restrict__ output, int A, int B, int C) {
    int a_c_idx = blockIdx.x;
    if (a_c_idx >= A * C) return;
    int a = a_c_idx / C;
    int c = a_c_idx % C;

    int tid = threadIdx.x;
    float max_val = -3.402823466e+38f;

    const float* input_ptr = input + (a * B) * C + c;
    for (int b = tid; b < B; b += blockDim.x) {
        float val = input_ptr[b * C];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Shared memory for warp-level results
    static __shared__ float shared_max[64]; // Max 1024 threads / 64 = 16 warps
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    max_val = warpReduceMax(max_val);

    if (lane_id == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp_id == 0) {
        float val = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_max[lane_id] : -3.402823466e+38f;
        val = warpReduceMax(val);
        if (tid == 0) {
            output[a_c_idx] = val;
        }
    }
}

torch::Tensor max_reduction_hip(torch::Tensor x, int64_t dim) {
    if (dim < 0) dim += x.dim();
    
    auto sizes = x.sizes();
    int A = 1;
    for (int i = 0; i < dim; ++i) A *= sizes[i];
    int B = sizes[dim];
    int C = 1;
    for (int i = dim + 1; i < x.dim(); ++i) C *= sizes[i];

    auto output_sizes = sizes.vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto out = torch::empty(output_sizes, x.options());

    if (C >= 64) {
        const int block_size = 256;
        int num_outputs = A * C;
        const int num_blocks = (num_outputs + block_size - 1) / block_size;
        max_reduction_case1_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), A, B, C);
    } else {
        const int block_size = 256; // Must be a multiple of WARP_SIZE
        int num_blocks = A * C;
        max_reduction_case2_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), A, B, C);
    }

    return out;
}
