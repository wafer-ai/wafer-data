#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define WARP_SIZE 32

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    uint8_t sign = (val < 0.0f) ? 1 : 0;
    val = fabsf(fminf(val, FP4_MAX));
    uint8_t encoded = (val > 5.0f) ? 7 : (val > 3.5f) ? 6 : (val > 2.5f) ? 5 : (val > 1.75f) ? 4
        : (val > 1.25f) ? 3 : (val > 0.75f) ? 2 : (val > 0.25f) ? 1 : 0;
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
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
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
    const int row = blockIdx.x;
    const int blk = threadIdx.x;
    const int num_blocks = blockDim.x;
    const float inv_hidden = 1.0f / (float)hidden;
    const float gs = d_global_scale ? __ldg(d_global_scale) : 1.0f;

    const __nv_bfloat16* row_in = input + row * hidden;
    __nv_bfloat16* row_res = residual + row * hidden;
    const __nv_bfloat16* row_w = weight;
    const int base = blk * block_size;

    float vals[16];
    float block_sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < block_size; i += 2) {
        const int col = base + i;
        __nv_bfloat162 in2 = __ldg(reinterpret_cast<const __nv_bfloat162*>(&row_in[col]));
        __nv_bfloat162 res2 = *reinterpret_cast<const __nv_bfloat162*>(&row_res[col]);
        float v0 = bf16_to_float(in2.x) + bf16_to_float(res2.x);
        float v1 = bf16_to_float(in2.y) + bf16_to_float(res2.y);
        vals[i] = v0;
        vals[i + 1] = v1;
        *reinterpret_cast<__nv_bfloat162*>(&row_res[col]) = {float_to_bf16(v0), float_to_bf16(v1)};
        block_sum_sq += v0 * v0 + v1 * v1;
    }

    __shared__ float smem_sum[32];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    float warp_sum = warp_reduce_sum(block_sum_sq);
    if (lane == 0) smem_sum[warp_id] = warp_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = lane < 4 ? smem_sum[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem_sum[0] = v;
    }
    __syncthreads();
    const float rnorm = rsqrtf(smem_sum[0] * inv_hidden + eps);

    float block_max = 0.0f;
#pragma unroll
    for (int i = 0; i < block_size; i += 2) {
        __nv_bfloat162 w2 = __ldg(reinterpret_cast<const __nv_bfloat162*>(&row_w[base + i]));
        float v0 = vals[i] * rnorm * bf16_to_float(w2.x);
        float v1 = vals[i + 1] * rnorm * bf16_to_float(w2.y);
        vals[i] = v0;
        vals[i + 1] = v1;
        block_max = fmaxf(block_max, fmaxf(fabsf(v0), fabsf(v1)));
    }

    const float bs = fmaxf(block_max / FP4_MAX, 1.0f / 512.0f);
    out_scales[row * num_blocks + blk] = float_to_e4m3(gs * bs);

    const float inv_scale = (block_max > 1e-10f) ? (FP4_MAX / block_max) : 0.0f;
    uint8_t* fp4_out = out_fp4 + row * (hidden / 2) + blk * (block_size / 2);
#pragma unroll
    for (int i = 0; i < block_size; i += 2) {
        uint8_t n0 = float_to_fp4_e2m1(vals[i] * inv_scale);
        uint8_t n1 = float_to_fp4_e2m1(vals[i + 1] * inv_scale);
        fp4_out[i / 2] = (n1 << 4) | n0;
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
