
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void mean_reduction_element_kernel(const float* x, float* out, int outer, int reduction, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = outer * inner;
    if (idx < total_output_elements) {
        int outer_idx = idx / inner;
        int inner_idx = idx % inner;
        float sum = 0;
        const float* base_ptr = x + (outer_idx * reduction) * inner + inner_idx;
        
        int k = 0;
        for (; k <= reduction - 8; k += 8) {
            sum += base_ptr[k * inner];
            sum += base_ptr[(k + 1) * inner];
            sum += base_ptr[(k + 2) * inner];
            sum += base_ptr[(k + 3) * inner];
            sum += base_ptr[(k + 4) * inner];
            sum += base_ptr[(k + 5) * inner];
            sum += base_ptr[(k + 6) * inner];
            sum += base_ptr[(k + 7) * inner];
        }
        for (; k < reduction; k++) {
            sum += base_ptr[k * inner];
        }
        out[idx] = sum / (float)reduction;
    }
}

template <int BLOCK_SIZE>
__global__ void mean_reduction_block_kernel(const float* x, float* out, int outer, int reduction, int inner) {
    int out_idx = blockIdx.x;
    int outer_idx = out_idx / inner;
    int inner_idx = out_idx % inner;
    int tid = threadIdx.x;

    float sum = 0;
    const float* base_ptr = x + (outer_idx * reduction) * inner + inner_idx;
    for (int k = tid; k < reduction; k += BLOCK_SIZE) {
        sum += base_ptr[k * inner];
    }

    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[out_idx] = shared_sum[0] / (float)reduction;
    }
}

torch::Tensor mean_reduction_hip(torch::Tensor x, int64_t dim) {
    if (dim < 0) dim += x.dim();
    
    auto shape = x.sizes();
    int64_t outer = 1;
    for (int i = 0; i < dim; i++) outer *= shape[i];
    int64_t reduction = shape[dim];
    int64_t inner = 1;
    for (int i = dim + 1; i < x.dim(); i++) inner *= shape[i];

    auto out_shape = shape.vec();
    out_shape.erase(out_shape.begin() + dim);
    auto out = torch::empty(out_shape, x.options());

    int64_t total_output_elements = outer * inner;
    
    if (inner >= 64) {
        const int block_size = 256;
        const int num_blocks = (total_output_elements + block_size - 1) / block_size;
        mean_reduction_element_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), (int)outer, (int)reduction, (int)inner);
    } else {
        const int block_size = 256;
        mean_reduction_block_kernel<block_size><<<total_output_elements, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), (int)outer, (int)reduction, (int)inner);
    }

    return out;
}
