#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cub/cub.cuh>

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    uint32_t bits = __float_as_uint(val);
    uint8_t sign = (bits >> 28) & 8;
    val = fabsf(val);
    uint8_t encoded;
    if (val <= 1.75f) {
        encoded = __float2int_rn(val * 2.0f);
    } else {
        encoded = (val <= 2.5f) ? 4 : ((val < 3.5f) ? 5 : ((val <= 5.0f) ? 6 : 7));
    }
    return sign | encoded;
}

__device__ __forceinline__ uint8_t float_to_e4m3(float val) {
    if (val <= 0.0f) return 0;
    if (val > E4M3_MAX) val = E4M3_MAX;
    float min_normal = 1.0f / 64.0f;
    if (val < min_normal) {
        int man = __float2int_rn(val * 64.0f * 8.0f);
        if (man > 7) man = 7; if (man < 0) man = 0;
        return (uint8_t)man;
    }
    int exp_unbiased = (int)floorf(log2f(val));
    if (exp_unbiased < -6) exp_unbiased = -6;
    if (exp_unbiased >  8) exp_unbiased = 8;
    int biased_exp = exp_unbiased + E4M3_BIAS;
    if (biased_exp < 1) biased_exp = 1; if (biased_exp > 15) biased_exp = 15;
    float pow2 = exp2f((float)exp_unbiased);
    int man = __float2int_rn((val / pow2 - 1.0f) * 8.0f);
    if (man > 7) { man = 0; biased_exp++; }
    if (man < 0) man = 0;
    if (biased_exp > 15) { biased_exp = 15; man = 6; }
    return (uint8_t)((biased_exp << 3) | man);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    int biased_exp = (bits >> 3) & 0xF; int man = bits & 0x7;
    if (biased_exp == 0) return (float(man) / 8.0f) * (1.0f / 64.0f);
    return (1.0f + float(man) / 8.0f) * exp2f(float(biased_exp - E4M3_BIAS));
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t nibble) {
    uint8_t sign = (nibble >> 3) & 1;
    uint8_t exp  = (nibble >> 1) & 3;
    uint8_t man  = nibble & 1;
    float val;
    if (exp == 0) { val = man * 0.5f; }
    else { val = (1.0f + man * 0.5f) * float(1 << (exp - 1)); }
    return sign ? -val : val;
}

__global__ void nvfp4_dequant_kernel(
    const uint8_t* __restrict__ fp4_packed,
    const uint8_t* __restrict__ block_scales,
    float inv_global_scale,
    float* __restrict__ output,
    int total_elems, int hidden, int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    int row = idx / hidden, col = idx % hidden;
    int blk = col / block_size, within = col % block_size;
    uint8_t packed = fp4_packed[row * (hidden / 2) + blk * (block_size / 2) + within / 2];
    uint8_t nibble = (within % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    float bs = e4m3_to_float(block_scales[row * (hidden / block_size) + blk]);
    output[idx] = fp4_e2m1_to_float(nibble) * bs * inv_global_scale;
}

torch::Tensor nvfp4_dequant(torch::Tensor fp4_packed, torch::Tensor block_scales,
                            float global_scale, int hidden, int block_size) {
    int total_rows = fp4_packed.size(0);
    auto out = torch::empty({total_rows, hidden}, fp4_packed.options().dtype(torch::kFloat32));
    int n = total_rows * hidden;
    nvfp4_dequant_kernel<<<(n + 255) / 256, 256>>>(
        fp4_packed.data_ptr<uint8_t>(), block_scales.data_ptr<uint8_t>(),
        1.0f / global_scale, out.data_ptr<float>(), n, hidden, block_size);
    return out;
}

__global__ void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ d_global_scale, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden, int block_size
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_blocks = hidden / block_size;
    
    float thread_sum_sq = 0.0f;
    float vals[16];
    
    if (tid < num_blocks) {
        int offset = row * hidden + tid * block_size;
        
        const float4* in_ptr = reinterpret_cast<const float4*>(input + offset);
        float4* res_ptr = reinterpret_cast<float4*>(residual + offset);
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) { // block_size=16, 2 float4s
            float4 in_vec = __ldg(&in_ptr[i]);
            float4 res_vec = res_ptr[i];
            
            const __nv_bfloat162* in_bf162 = reinterpret_cast<const __nv_bfloat162*>(&in_vec);
            const __nv_bfloat162* res_bf162 = reinterpret_cast<const __nv_bfloat162*>(&res_vec);
            __nv_bfloat162 out_bf162[4];
            
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_bfloat162 out_bf162_val = in_bf162[j] + res_bf162[j];
                out_bf162[j] = out_bf162_val;
                
                float2 out_f2 = __bfloat1622float2(out_bf162_val);
                
                vals[i * 8 + j * 2 + 0] = out_f2.x;
                vals[i * 8 + j * 2 + 1] = out_f2.y;
                
                thread_sum_sq += out_f2.x * out_f2.x + out_f2.y * out_f2.y;
            }
            
            res_ptr[i] = *reinterpret_cast<float4*>(out_bf162);
        }
    }
    
    static __shared__ float shared_sum[32];
    float warp_sum = thread_sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) shared_sum[warp_id] = warp_sum;
    __syncthreads();
    
    float block_sum_sq = 0.0f;
    if (tid < 32) {
        block_sum_sq = (tid < (blockDim.x + 31) / 32) ? shared_sum[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum_sq += __shfl_down_sync(0xffffffff, block_sum_sq, offset);
        }
    }
    
    __shared__ float rsigma;
    __shared__ float shared_global_scale;
    if (tid == 0) {
        float variance = block_sum_sq / hidden;
        rsigma = rsqrtf(variance + eps);
        shared_global_scale = d_global_scale ? __ldg(d_global_scale) : 1.0f;
    }
    __syncthreads();
    
    if (tid < num_blocks) {
        int weight_offset = tid * block_size;
        const float4* w_ptr = reinterpret_cast<const float4*>(weight + weight_offset);
        
        float max_abs_y = 0.0f;
        float y_vals[16];
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            float4 w_vec = __ldg(&w_ptr[i]);
            const __nv_bfloat162* w_bf162 = reinterpret_cast<const __nv_bfloat162*>(&w_vec);
            
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 w_f2 = __bfloat1622float2(w_bf162[j]);
                
                float x0 = vals[i * 8 + j * 2 + 0] * rsigma;
                float x1 = vals[i * 8 + j * 2 + 1] * rsigma;
                
                float y0 = x0 * w_f2.x;
                float y1 = x1 * w_f2.y;
                
                y_vals[i * 8 + j * 2 + 0] = y0;
                y_vals[i * 8 + j * 2 + 1] = y1;
                
                max_abs_y = fmaxf(max_abs_y, fabsf(y0));
                max_abs_y = fmaxf(max_abs_y, fabsf(y1));
            }
        }
        
        max_abs_y *= shared_global_scale;
        
        float block_scale = max_abs_y / FP4_MAX;
        uint8_t e4m3_scale = float_to_e4m3(block_scale);
        out_scales[row * num_blocks + tid] = e4m3_scale;
        
        float e4m3_scale_f = e4m3_to_float(e4m3_scale);
        float inv_scale = shared_global_scale / e4m3_scale_f;
        
        int fp4_offset = row * (hidden / 2) + tid * (block_size / 2);
        
        uint32_t packed_out[2];
        uint8_t* packed_bytes = reinterpret_cast<uint8_t*>(packed_out);
        
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            float fp4_val0 = y_vals[i] * inv_scale;
            float fp4_val1 = y_vals[i + 1] * inv_scale;
            
            uint8_t q0 = float_to_fp4_e2m1(fp4_val0);
            uint8_t q1 = float_to_fp4_e2m1(fp4_val1);
            
            packed_bytes[i / 2] = q0 | (q1 << 4);
        }
        
        *reinterpret_cast<uint2*>(out_fp4 + fp4_offset) = *reinterpret_cast<uint2*>(packed_out);
    }
}

std::vector<torch::Tensor> add_rmsnorm_fp4quant(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
    std::optional<torch::Tensor> y_fp4,
    std::optional<torch::Tensor> block_scale,
    std::optional<torch::Tensor> global_scale,
    double eps, int64_t block_size,
    std::optional<std::string> scale_format,
    bool is_sf_swizzled_layout,
    bool output_both_sf_layouts,
    std::optional<torch::Tensor> block_scale_unswizzled
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 2);
    int batch = input.size(0), hidden = input.size(1);
    TORCH_CHECK(hidden % block_size == 0);

    const float* d_gs = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;

    auto fp4_raw = torch::empty({batch, hidden / 2}, input.options().dtype(torch::kUInt8));
    auto scales_raw = torch::empty({batch, hidden / (int)block_size}, input.options().dtype(torch::kUInt8));

    int num_blocks_per_row = hidden / (int)block_size;
    int block_threads = 1;
    while (block_threads < num_blocks_per_row) block_threads <<= 1;
    TORCH_CHECK(block_threads <= 1024);
    fused_add_rmsnorm_nvfp4_kernel<<<batch, block_threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
        batch, hidden, (int)block_size);

    torch::Tensor fp4_out, scale_out;
    if (y_fp4.has_value()) {
        y_fp4->view(torch::kUInt8).copy_(fp4_raw);
        fp4_out = *y_fp4;
    } else {
        fp4_out = fp4_raw.view(c10::ScalarType::Float4_e2m1fn_x2);
    }
    if (block_scale.has_value()) {
        block_scale->view(torch::kUInt8).copy_(scales_raw);
        scale_out = *block_scale;
    } else {
        scale_out = scales_raw.view(c10::ScalarType::Float8_e4m3fn);
    }
    return {fp4_out, scale_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_rmsnorm_fp4quant", &add_rmsnorm_fp4quant,
          py::arg("input"), py::arg("residual"), py::arg("weight"),
          py::arg("y_fp4") = py::none(), py::arg("block_scale") = py::none(),
          py::arg("global_scale") = py::none(), py::arg("eps") = 1e-6,
          py::arg("block_size") = 16, py::arg("scale_format") = py::none(),
          py::arg("is_sf_swizzled_layout") = false,
          py::arg("output_both_sf_layouts") = false,
          py::arg("block_scale_unswizzled") = py::none());
    m.def("nvfp4_dequant", &nvfp4_dequant);
}
