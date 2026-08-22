#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    uint8_t sign = (val < 0.0f);
    if (sign) val = -val;
    val = fminf(val, FP4_MAX);
    uint8_t encoded = (val > 0.25f) + (val >= 0.75f) + (val > 1.25f) + (val >= 1.75f)
                    + (val > 2.5f) + (val >= 3.5f) + (val > 5.0f);
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

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float smem[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
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
    int num_blocks = hidden / block_size;
    int tid = threadIdx.x;
    if (tid >= num_blocks) return;

    float gs = (d_global_scale != nullptr) ? __ldg(d_global_scale) : 1.0f;
    const __nv_bfloat16* inp_row = input + row * hidden;
    __nv_bfloat16* res_row = residual + row * hidden;
    const __nv_bfloat16* w = weight;
    int base = tid * block_size;

    float s[16];
    const __nv_bfloat16* inp = inp_row + base;
    __nv_bfloat16* res = res_row + base;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        __nv_bfloat162 inv2 = __ldg((const __nv_bfloat162*)(inp + i*2));
        __nv_bfloat162 res2 = *(__nv_bfloat162*)(res + i*2);
        float2 invf = __bfloat1622float2(inv2);
        float2 resf = __bfloat1622float2(res2);
        float v0 = invf.x + resf.x, v1 = invf.y + resf.y;
        s[i*2] = v0;
        s[i*2+1] = v1;
        *(__nv_bfloat162*)(res + i*2) = __float22bfloat162_rn(make_float2(v0, v1));
    }

    float sum_sq = s[0]*s[0]+s[1]*s[1]+s[2]*s[2]+s[3]*s[3]+s[4]*s[4]+s[5]*s[5]+s[6]*s[6]+s[7]*s[7]+
                   s[8]*s[8]+s[9]*s[9]+s[10]*s[10]+s[11]*s[11]+s[12]*s[12]+s[13]*s[13]+s[14]*s[14]+s[15]*s[15];
    float rnorm = rsqrtf(block_reduce_sum(sum_sq) / (float)hidden + eps);

    float n[16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        __nv_bfloat162 w2 = __ldg((const __nv_bfloat162*)(w + base + i*2));
        float2 wf = __bfloat1622float2(w2);
        n[i*2] = s[i*2] * rnorm * wf.x;
        n[i*2+1] = s[i*2+1] * rnorm * wf.y;
    }

    float block_max = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        float a = fabsf(n[i]);
        if (a > block_max) block_max = a;
    }

    float scale_val = (block_max > 1e-9f) ? (gs * block_max * (1.0f/FP4_MAX)) : 0.0f;
    uint8_t scale_u8 = float_to_e4m3(scale_val);
    float scale_dec = e4m3_to_float(scale_u8);
    float scale_inv = (scale_dec > 1e-9f) ? (gs * (1.0f/scale_dec)) : 0.0f;

    uint8_t* fp4_row = out_fp4 + row * (hidden / 2) + base / 2;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        float v0 = fmaxf(-FP4_MAX, fminf(FP4_MAX, n[i*2] * scale_inv));
        float v1 = fmaxf(-FP4_MAX, fminf(FP4_MAX, n[i*2+1] * scale_inv));
        fp4_row[i] = (float_to_fp4_e2m1(v1) << 4) | float_to_fp4_e2m1(v0);
    }
    out_scales[row * num_blocks + tid] = scale_u8;
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
