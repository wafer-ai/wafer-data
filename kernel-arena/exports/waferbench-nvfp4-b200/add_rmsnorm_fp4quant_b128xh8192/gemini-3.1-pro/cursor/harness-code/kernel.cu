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
    uint8_t sign = (bits >> 31) & 1;
    uint32_t abs_bits = bits & 0x7FFFFFFF;
    
    if (abs_bits > 0x40C00000) return (sign << 3) | 0b111;
    
    float abs_val = __uint_as_float(abs_bits);
    uint8_t encoded;
    if      (abs_val <= 0.25f) encoded = 0b000;
    else if (abs_val < 0.75f)  encoded = 0b001;
    else if (abs_val <= 1.25f) encoded = 0b010;
    else if (abs_val < 1.75f)  encoded = 0b011;
    else if (abs_val <= 2.5f)  encoded = 0b100;
    else if (abs_val < 3.5f)   encoded = 0b101;
    else if (abs_val <= 5.0f)  encoded = 0b110;
    else                       encoded = 0b111;
    return (sign << 3) | encoded;
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

__global__ void fused_add_rmsnorm_nvfp4_kernel_8192(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ global_scale_ptr,
    float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales
) {
    int row = blockIdx.x;
    int tid = threadIdx.x; // 0 to 1023
    
    const __nv_bfloat16* in_ptr = input + row * 8192;
    __nv_bfloat16* res_ptr = residual + row * 8192;
    uint8_t* fp4_ptr = out_fp4 + row * 4096;
    uint8_t* scale_ptr = out_scales + row * 512;
    
    int col_offset = tid * 8;
    
    float4 in_f4 = __ldg(reinterpret_cast<const float4*>(in_ptr + col_offset));
    float4 res_f4 = *reinterpret_cast<const float4*>(res_ptr + col_offset);
    
    __nv_bfloat162* in_bf2 = reinterpret_cast<__nv_bfloat162*>(&in_f4);
    __nv_bfloat162* res_bf2 = reinterpret_cast<__nv_bfloat162*>(&res_f4);
    
    float4 new_res_f4;
    __nv_bfloat162* new_res_bf2 = reinterpret_cast<__nv_bfloat162*>(&new_res_f4);
    
    float sum_sq = 0.0f;
    
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 in_f2 = __bfloat1622float2(in_bf2[j]);
        float2 res_f2 = __bfloat1622float2(res_bf2[j]);
        
        float2 added;
        added.x = in_f2.x + res_f2.x;
        added.y = in_f2.y + res_f2.y;
        
        new_res_bf2[j] = __float22bfloat162_rn(added);
        
        sum_sq += added.x * added.x + added.y * added.y;
    }
    
    *reinterpret_cast<float4*>(res_ptr + col_offset) = new_res_f4;
    
    typedef cub::BlockReduce<float, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float total_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);
    
    __shared__ float inv_rms_val;
    __shared__ float shared_global_scale;
    if (tid == 0) {
        inv_rms_val = rsqrtf(total_sum_sq / 8192.0f + eps);
        shared_global_scale = global_scale_ptr ? *global_scale_ptr : 1.0f;
    }
    __syncthreads();
    
    float inv_rms = inv_rms_val;
    float global_scale = shared_global_scale;
    
    float4 w_f4 = __ldg(reinterpret_cast<const float4*>(weight + col_offset));
    __nv_bfloat162* w_bf2 = reinterpret_cast<__nv_bfloat162*>(&w_f4);
    
    float max_abs = 0.0f;
    float y[8];
    
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 res_f2 = __bfloat1622float2(new_res_bf2[j]);
        float2 w_f2 = __bfloat1622float2(w_bf2[j]);
        
        y[j * 2 + 0] = res_f2.x * inv_rms * w_f2.x;
        y[j * 2 + 1] = res_f2.y * inv_rms * w_f2.y;
        
        max_abs = fmaxf(max_abs, fabsf(y[j * 2 + 0]));
        max_abs = fmaxf(max_abs, fabsf(y[j * 2 + 1]));
    }
    
    float block_max = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, 1));
    
    float bs_float = block_max * global_scale * (1.0f / 6.0f);
    uint8_t bs_e4m3 = float_to_e4m3(bs_float);
    float bs_actual = e4m3_to_float(bs_e4m3);
    
    float scale_factor = (bs_actual == 0.0f) ? 0.0f : (global_scale / bs_actual);
    
    uint32_t packed_fp4 = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float fp4_val = y[j] * scale_factor;
        uint8_t nibble = float_to_fp4_e2m1(fp4_val);
        packed_fp4 |= (uint32_t)(nibble & 0xF) << (j * 4);
    }
    
    *reinterpret_cast<uint32_t*>(fp4_ptr + col_offset / 2) = packed_fp4;
    
    uint8_t bs_e4m3_other = __shfl_down_sync(0xffffffff, bs_e4m3, 2);
    if ((tid % 4) == 0) {
        uint16_t both_scales = bs_e4m3 | ((uint16_t)bs_e4m3_other << 8);
        reinterpret_cast<uint16_t*>(scale_ptr)[col_offset / 32] = both_scales;
    }
}

constexpr int BLOCK_GEN = 256;

__global__ void fused_add_rmsnorm_nvfp4_kernel_generic(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ global_scale_ptr,
    float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden, int block_size
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch) return;
    
    const __nv_bfloat16* in_ptr = input + row * hidden;
    __nv_bfloat16* res_ptr = residual + row * hidden;
    uint8_t* fp4_ptr = out_fp4 + row * (hidden / 2);
    uint8_t* scale_ptr = out_scales + row * (hidden / block_size);
    
    float sum_sq = 0.0f;
    
    int num_iters = (hidden + blockDim.x * 8 - 1) / (blockDim.x * 8);
    
    float4 my_res_f4[4];
    
    for (int i = 0; i < num_iters; ++i) {
        int col_offset = i * blockDim.x * 8 + tid * 8;
        
        if (col_offset < hidden) {
            float4 in_f4 = __ldg(reinterpret_cast<const float4*>(in_ptr + col_offset));
            float4 res_f4 = *reinterpret_cast<const float4*>(res_ptr + col_offset);
            
            __nv_bfloat162* in_bf2 = reinterpret_cast<__nv_bfloat162*>(&in_f4);
            __nv_bfloat162* res_bf2 = reinterpret_cast<__nv_bfloat162*>(&res_f4);
            
            float4 new_res_f4;
            __nv_bfloat162* new_res_bf2 = reinterpret_cast<__nv_bfloat162*>(&new_res_f4);
            
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 in_f2 = __bfloat1622float2(in_bf2[j]);
                float2 res_f2 = __bfloat1622float2(res_bf2[j]);
                
                float2 added;
                added.x = in_f2.x + res_f2.x;
                added.y = in_f2.y + res_f2.y;
                
                new_res_bf2[j] = __float22bfloat162_rn(added);
                
                sum_sq += added.x * added.x + added.y * added.y;
            }
            
            *reinterpret_cast<float4*>(res_ptr + col_offset) = new_res_f4;
            my_res_f4[i] = new_res_f4;
        }
    }
    
    typedef cub::BlockReduce<float, BLOCK_GEN> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float total_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);
    
    __shared__ float inv_rms_val;
    __shared__ float shared_global_scale;
    if (tid == 0) {
        inv_rms_val = rsqrtf(total_sum_sq / hidden + eps);
        shared_global_scale = global_scale_ptr ? *global_scale_ptr : 1.0f;
    }
    __syncthreads();
    
    float inv_rms = inv_rms_val;
    float global_scale = shared_global_scale;
    
    int threads_per_block = block_size / 8;
    
    for (int i = 0; i < num_iters; ++i) {
        int col_offset = i * blockDim.x * 8 + tid * 8;
        
        float max_abs = 0.0f;
        float y[8] = {0.0f};
        
        if (col_offset < hidden) {
            float4 res_f4 = my_res_f4[i];
            float4 w_f4 = __ldg(reinterpret_cast<const float4*>(weight + col_offset));
            
            __nv_bfloat162* res_bf2 = reinterpret_cast<__nv_bfloat162*>(&res_f4);
            __nv_bfloat162* w_bf2 = reinterpret_cast<__nv_bfloat162*>(&w_f4);
            
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 res_f2 = __bfloat1622float2(res_bf2[j]);
                float2 w_f2 = __bfloat1622float2(w_bf2[j]);
                
                y[j * 2 + 0] = res_f2.x * inv_rms * w_f2.x;
                y[j * 2 + 1] = res_f2.y * inv_rms * w_f2.y;
                
                max_abs = fmaxf(max_abs, fabsf(y[j * 2 + 0]));
                max_abs = fmaxf(max_abs, fabsf(y[j * 2 + 1]));
            }
        }
        
        float block_max = max_abs;
        for (int offset = 1; offset < threads_per_block; offset *= 2) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, offset));
        }
        
        if (col_offset < hidden) {
            float bs_float = block_max * global_scale * (1.0f / 6.0f);
            uint8_t bs_e4m3 = float_to_e4m3(bs_float);
            float bs_actual = e4m3_to_float(bs_e4m3);
            
            float scale_factor = (bs_actual == 0.0f) ? 0.0f : (global_scale / bs_actual);
            
            uint32_t packed_fp4 = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float fp4_val = y[j] * scale_factor;
                uint8_t nibble = float_to_fp4_e2m1(fp4_val);
                packed_fp4 |= (uint32_t)(nibble & 0xF) << (j * 4);
            }
            
            *reinterpret_cast<uint32_t*>(fp4_ptr + col_offset / 2) = packed_fp4;
            
            if ((tid % threads_per_block) == 0) {
                scale_ptr[col_offset / block_size] = bs_e4m3;
            }
        }
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

    const float* gs_ptr = nullptr;
    if (global_scale.has_value()) {
        gs_ptr = global_scale->data_ptr<float>();
    }

    auto fp4_raw = torch::empty({batch, hidden / 2}, input.options().dtype(torch::kUInt8));
    auto scales_raw = torch::empty({batch, hidden / (int)block_size}, input.options().dtype(torch::kUInt8));

    if (hidden == 8192 && block_size == 16) {
        fused_add_rmsnorm_nvfp4_kernel_8192<<<batch, 1024>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            gs_ptr, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>());
    } else {
        fused_add_rmsnorm_nvfp4_kernel_generic<<<batch, BLOCK_GEN>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            gs_ptr, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
            batch, hidden, (int)block_size);
    }

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
