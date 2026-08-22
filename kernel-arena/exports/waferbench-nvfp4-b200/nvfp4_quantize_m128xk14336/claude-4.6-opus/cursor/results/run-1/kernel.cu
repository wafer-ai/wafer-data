#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

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

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ uint64_t cvt_16xf32_to_e2m1(float* a) {
    uint64_t val;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
        ".reg .b32 lo, hi;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0,  %2,  %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1,  %4,  %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2,  %6,  %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3,  %8,  %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b4, %10,  %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b5, %12, %11;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b6, %14, %13;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b7, %16, %15;\n"
        "mov.b32 lo, {b0, b1, b2, b3};\n"
        "mov.b32 hi, {b4, b5, b6, b7};\n"
        "mov.b64 %0, {lo, hi};\n"
        "}"
        : "=l"(val)
        : "f"(a[0]),  "f"(a[1]),  "f"(a[2]),  "f"(a[3]),
          "f"(a[4]),  "f"(a[5]),  "f"(a[6]),  "f"(a[7]),
          "f"(a[8]),  "f"(a[9]),  "f"(a[10]), "f"(a[11]),
          "f"(a[12]), "f"(a[13]), "f"(a[14]), "f"(a[15]));
    return val;
}

constexpr int TPB = 512;

__global__ __launch_bounds__(TPB, 4)
void nvfp4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, const float* __restrict__ d_global_scale,
    uint8_t* __restrict__ output, uint8_t* __restrict__ block_scales,
    int M, int K, int block_size
) {
    const int num_sf_blocks = K / block_size;
    const float gs = __ldg(d_global_scale);
    const int row = blockIdx.y;
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_sf_blocks) return;

    const __nv_bfloat16* blk_in = input + row * K + blk * block_size;

    uint4 raw0 = reinterpret_cast<const uint4*>(blk_in)[0];
    uint4 raw1 = reinterpret_cast<const uint4*>(blk_in)[1];

    __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&raw0);
    __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(&raw1);

    // Convert to float and find amax simultaneously
    float vals[16];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 f2 = __bfloat1622float2(p0[i]);
        vals[2*i]   = f2.x;
        vals[2*i+1] = f2.y;
        amax = fmaxf(amax, fmaxf(fabsf(f2.x), fabsf(f2.y)));
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 f2 = __bfloat1622float2(p1[i]);
        vals[8+2*i]   = f2.x;
        vals[8+2*i+1] = f2.y;
        amax = fmaxf(amax, fmaxf(fabsf(f2.x), fabsf(f2.y)));
    }

    float sf_val = gs * amax * rcp_approx_ftz(6.0f);
    __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_val);
    sf_val = float(sf_fp8);
    float out_scale = (amax != 0.0f) ? rcp_approx_ftz(sf_val * rcp_approx_ftz(gs)) : 0.0f;

    block_scales[row * num_sf_blocks + blk] = sf_fp8.__x;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        vals[i] *= out_scale;
    }

    uint64_t e2m1 = cvt_16xf32_to_e2m1(vals);
    *reinterpret_cast<uint64_t*>(output + row * (K / 2) + blk * (block_size / 2)) = e2m1;
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

    int num_blks = K / (int)sf_vec_size;
    dim3 grid((num_blks + TPB - 1) / TPB, M);
    nvfp4_quantize_kernel<<<grid, TPB>>>(
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
