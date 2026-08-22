#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

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
    int biased_exp = (bits >> 3) & 0xF;
    int man = bits & 0x7;
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
    int total_elems, int K, int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    int row = idx / K, col = idx % K;
    int blk = col / block_size, within = col % block_size;
    uint8_t packed = fp4_packed[row * (K / 2) + blk * (block_size / 2) + within / 2];
    uint8_t nibble = (within % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    float bs = e4m3_to_float(block_scales[row * (K / block_size) + blk]);
    output[idx] = fp4_e2m1_to_float(nibble) * bs * inv_global_scale;
}

torch::Tensor nvfp4_dequant(torch::Tensor fp4_packed, torch::Tensor block_scales,
                            float global_scale, int K, int block_size) {
    int M = fp4_packed.size(0);
    auto out = torch::empty({M, K}, fp4_packed.options().dtype(torch::kFloat32));
    int n = M * K;
    nvfp4_dequant_kernel<<<(n + 255) / 256, 256>>>(
        fp4_packed.data_ptr<uint8_t>(), block_scales.data_ptr<uint8_t>(),
        1.0f / global_scale, out.data_ptr<float>(), n, K, block_size);
    return out;
}

__global__ void nvfp4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, const float* __restrict__ d_global_scale,
    uint8_t* __restrict__ output, uint8_t* __restrict__ block_scales,
    int M, int K, int block_size
) {
    const int row = blockIdx.x;
    const int blk = threadIdx.x;
    if (blk >= K / block_size) return;

    const float gs = d_global_scale ? __ldg(d_global_scale) : 1.0f;
    const __nv_bfloat16* row_in = input + row * K + blk * block_size;
    const int half_block = block_size / 2;

    uint4 chunk0 = __ldg(reinterpret_cast<const uint4*>(row_in));
    uint4 chunk1 = __ldg(reinterpret_cast<const uint4*>(row_in + 8));
    float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&chunk0.x));
    float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&chunk0.y));
    float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&chunk0.z));
    float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&chunk0.w));
    float2 f4 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&chunk1.x));
    float2 f5 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&chunk1.y));
    float2 f6 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&chunk1.z));
    float2 f7 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&chunk1.w));

    float vec_max = fmaxf(fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))),
                  fmaxf(fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))),
                  fmaxf(fmaxf(fmaxf(fabsf(f4.x), fabsf(f4.y)), fmaxf(fabsf(f5.x), fabsf(f5.y))),
                  fmaxf(fmaxf(fabsf(f6.x), fabsf(f6.y)), fmaxf(fabsf(f7.x), fabsf(f7.y))))));

    const float scale_raw = (vec_max > 1e-12f) ? (gs * vec_max / FP4_MAX) : 0.0f;
    const uint8_t scale_u8 = float_to_e4m3(scale_raw);
    block_scales[row * (K / block_size) + blk] = scale_u8;

    const float scale_dequant = e4m3_to_float(scale_u8);
    const float out_scale = (scale_dequant > 1e-12f) ? (gs / scale_dequant) : 0.0f;

    #define S(x) (fmaxf(-FP4_MAX, fminf(FP4_MAX, (x) * out_scale)))
    uint64_t packed64 =
        (uint64_t)((float_to_fp4_e2m1(S(f0.y)) << 4) | float_to_fp4_e2m1(S(f0.x))) |
        (uint64_t)((float_to_fp4_e2m1(S(f1.y)) << 4) | float_to_fp4_e2m1(S(f1.x))) << 8 |
        (uint64_t)((float_to_fp4_e2m1(S(f2.y)) << 4) | float_to_fp4_e2m1(S(f2.x))) << 16 |
        (uint64_t)((float_to_fp4_e2m1(S(f3.y)) << 4) | float_to_fp4_e2m1(S(f3.x))) << 24 |
        (uint64_t)((float_to_fp4_e2m1(S(f4.y)) << 4) | float_to_fp4_e2m1(S(f4.x))) << 32 |
        (uint64_t)((float_to_fp4_e2m1(S(f5.y)) << 4) | float_to_fp4_e2m1(S(f5.x))) << 40 |
        (uint64_t)((float_to_fp4_e2m1(S(f6.y)) << 4) | float_to_fp4_e2m1(S(f6.x))) << 48 |
        (uint64_t)((float_to_fp4_e2m1(S(f7.y)) << 4) | float_to_fp4_e2m1(S(f7.x))) << 56;
    #undef S
    *reinterpret_cast<uint64_t*>(output + row * (K / 2) + blk * half_block) = packed64;
}

std::vector<torch::Tensor> fp4_quantize(
    torch::Tensor input,
    std::optional<torch::Tensor> global_scale,
    int64_t sf_vec_size, bool sf_use_ue8m0,
    bool is_sf_swizzled_layout,
    bool is_sf_8x4_layout,
    std::optional<bool> enable_pdl)
{
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 2);
    int M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % sf_vec_size == 0);

    const float* d_gs = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;

    auto packed = torch::empty({M, K / 2}, input.options().dtype(torch::kUInt8));
    auto scales = torch::empty({M, K / (int)sf_vec_size}, input.options().dtype(torch::kUInt8));

    nvfp4_quantize_kernel<<<M, K / (int)sf_vec_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()), d_gs,
        packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), M, K, (int)sf_vec_size);

    return {packed, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp4_quantize", &fp4_quantize,
          py::arg("input"),
          py::arg("global_scale") = py::none(),
          py::arg("sf_vec_size") = 16, py::arg("sf_use_ue8m0") = false,
          py::arg("is_sf_swizzled_layout") = true,
          py::arg("is_sf_8x4_layout") = false,
          py::arg("enable_pdl") = py::none());
    m.def("nvfp4_dequant", &nvfp4_dequant);
}
