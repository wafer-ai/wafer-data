#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    unsigned short bits = *reinterpret_cast<const unsigned short*>(&x);
    return __uint_as_float((unsigned int)bits << 16);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned short bits = (unsigned short)(u >> 16);
    __nv_bfloat16 r;
    *reinterpret_cast<unsigned short*>(&r) = bits;
    return r;
}

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    uint8_t sign = 0;
    if (val < 0.0f) { sign = 1; val = -val; }
    if (val > FP4_MAX) val = FP4_MAX;
    uint8_t encoded;
    if      (val <= 0.25f) encoded = 0b000;
    else if (val < 0.75f)  encoded = 0b001;
    else if (val <= 1.25f) encoded = 0b010;
    else if (val < 1.75f)  encoded = 0b011;
    else if (val <= 2.5f)  encoded = 0b100;
    else if (val < 3.5f)   encoded = 0b101;
    else if (val <= 5.0f)  encoded = 0b110;
    else                   encoded = 0b111;
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

constexpr int BLOCK = 256;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float smem[BLOCK / WARP_SIZE];
    int wid = threadIdx.x / WARP_SIZE;
    v = warp_reduce_sum(v);
    if (threadIdx.x % WARP_SIZE == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < BLOCK / WARP_SIZE) ? smem[threadIdx.x] : 0.0f;
    return warp_reduce_sum(v);
}

__device__ __forceinline__ void load_bf16x2(const __nv_bfloat16* p, float& a, float& b) {
    unsigned int u = *reinterpret_cast<const unsigned int*>(p);
    a = __uint_as_float((unsigned int)(u & 0xFFFF) << 16);
    b = __uint_as_float((unsigned int)(u >> 16) << 16);
}

__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* p, float& a, float& b, float& c, float& d) {
    unsigned long long u = *reinterpret_cast<const unsigned long long*>(p);
    a = __uint_as_float((unsigned int)(u & 0xFFFF) << 16);
    b = __uint_as_float((unsigned int)((u >> 16) & 0xFFFF) << 16);
    c = __uint_as_float((unsigned int)((u >> 32) & 0xFFFF) << 16);
    d = __uint_as_float((unsigned int)(u >> 48) << 16);
}

__global__ __launch_bounds__(BLOCK, 4) void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    float global_scale, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden, int block_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_blocks = hidden / block_size;
    const int fp4_bytes_per_block = block_size / 2;

    const __nv_bfloat16* in_row = input + row * hidden;
    __nv_bfloat16* res_row = residual + row * hidden;

    float sum_sq = 0.0f;
    for (int i = tid * 4; i + 3 < hidden; i += BLOCK * 4) {
        float in0, in1, in2, in3, res0, res1, res2, res3;
        load_bf16x4(in_row + i, in0, in1, in2, in3);
        load_bf16x4(res_row + i, res0, res1, res2, res3);
        float h0 = in0 + res0, h1 = in1 + res1, h2 = in2 + res2, h3 = in3 + res3;
        res_row[i] = float_to_bf16(h0);
        res_row[i + 1] = float_to_bf16(h1);
        res_row[i + 2] = float_to_bf16(h2);
        res_row[i + 3] = float_to_bf16(h3);
        sum_sq += h0 * h0 + h1 * h1 + h2 * h2 + h3 * h3;
    }
    for (int i = tid * 4; i < hidden; i += BLOCK * 4) {
        if (i + 3 >= hidden) {
            for (int j = i; j < hidden; j++) {
                float h = bf16_to_float(in_row[j]) + bf16_to_float(res_row[j]);
                res_row[j] = float_to_bf16(h);
                sum_sq += h * h;
            }
            break;
        }
    }
    sum_sq = block_reduce_sum(sum_sq);
    float rsigma = rsqrtf(sum_sq / (float)hidden + eps);
    __syncthreads();

    for (int b = tid; b < num_blocks; b += BLOCK) {
        float block_max = 0.0f;
        float normed[16];
#pragma unroll
        for (int k = 0; k < block_size; k++) {
            int idx = b * block_size + k;
            float v = bf16_to_float(res_row[idx]) * rsigma * bf16_to_float(weight[idx]);
            normed[k] = v;
            block_max = fmaxf(block_max, fabsf(v));
        }
        block_max = fmaxf(block_max, 1e-8f);

        float scale_f = global_scale * (block_max / FP4_MAX);
        out_scales[row * num_blocks + b] = float_to_e4m3(scale_f);

        float inv_scale = FP4_MAX / block_max;
#pragma unroll
        for (int k = 0; k < block_size; k += 2) {
            uint8_t nib0 = float_to_fp4_e2m1(normed[k] * inv_scale);
            uint8_t nib1 = float_to_fp4_e2m1(normed[k + 1] * inv_scale);
            out_fp4[row * (hidden / 2) + b * fp4_bytes_per_block + k / 2] = nib0 | (nib1 << 4);
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

    float gs = global_scale.has_value() ? global_scale->item<float>() : 1.0f;

    auto fp4_raw = torch::empty({batch, hidden / 2}, input.options().dtype(torch::kUInt8));
    auto scales_raw = torch::empty({batch, hidden / (int)block_size}, input.options().dtype(torch::kUInt8));

    fused_add_rmsnorm_nvfp4_kernel<<<batch, BLOCK>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
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
