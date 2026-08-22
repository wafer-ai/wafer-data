#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    __nv_fp8_e4m3 val;
    val.__x = bits;
    return static_cast<float>(val);
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

template <int VecSize>
__global__ __launch_bounds__(512) void nvfp4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ d_global_scale,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int total_groups
) {
    static_assert(VecSize == 16, "This kernel is specialized for 16-value NVFP4 blocks.");

    const int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= total_groups) {
        return;
    }

    const float global_scale = __ldg(d_global_scale);
    const __nv_bfloat162* group_ptr =
        reinterpret_cast<const __nv_bfloat162*>(input) + group_idx * (VecSize / 2);

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < VecSize / 2; ++i) {
        const float2 vals = __bfloat1622float2(group_ptr[i]);
        amax = fmaxf(amax, fabsf(vals.x));
        amax = fmaxf(amax, fabsf(vals.y));
    }

    const uint8_t scale_byte =
        float_to_e4m3(amax * (global_scale * (1.0f / FP4_MAX)));
    block_scales[group_idx] = scale_byte;

    const float block_scale = e4m3_to_float(scale_byte);
    const float quant_scale = global_scale / block_scale;
    uint64_t packed_word = 0;

#pragma unroll
    for (int i = 0; i < VecSize / 2; ++i) {
        const float2 vals = __bfloat1622float2(group_ptr[i]);
        const float q0 = fminf(fmaxf(vals.x * quant_scale, -FP4_MAX), FP4_MAX);
        const float q1 = fminf(fmaxf(vals.y * quant_scale, -FP4_MAX), FP4_MAX);
        const uint8_t lo = float_to_fp4_e2m1(q0);
        const uint8_t hi = float_to_fp4_e2m1(q1);
        packed_word |= static_cast<uint64_t>(lo | (hi << 4)) << (i * 8);
    }

    reinterpret_cast<uint64_t*>(output)[group_idx] = packed_word;
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
    TORCH_CHECK(global_scale.has_value(), "global_scale is required");
    TORCH_CHECK(!sf_use_ue8m0, "ue8m0 scale factors are not supported");
    TORCH_CHECK(!is_sf_swizzled_layout, "swizzled scale layout is not supported");
    TORCH_CHECK(!is_sf_8x4_layout, "8x4 scale layout is not supported");
    int M = input.size(0), K = input.size(1);
    TORCH_CHECK(sf_vec_size == 16, "Only sf_vec_size=16 is supported");
    TORCH_CHECK(K % sf_vec_size == 0);

    const float* d_gs = global_scale->data_ptr<float>();

    auto packed = torch::empty({M, K / 2}, input.options().dtype(torch::kUInt8));
    auto scales = torch::empty({M, K / (int)sf_vec_size}, input.options().dtype(torch::kUInt8));

    constexpr int kThreads = 512;
    const int total_groups = M * (K / (int)sf_vec_size);
    nvfp4_quantize_kernel<16><<<(total_groups + kThreads - 1) / kThreads, kThreads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        d_gs,
        packed.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(),
        total_groups);

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
