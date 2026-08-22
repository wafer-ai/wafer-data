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
    if (val <= 0.0f) return 0;
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    return __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3)));
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

constexpr int BLOCK = 1024;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

template<int WIDTH>
__device__ __forceinline__ float subgroup_reduce_max(float val) {
    static_assert(WIDTH == 4 || WIDTH == 8 || WIDTH == 16 || WIDTH == 32);
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset, WIDTH));
    }
    return val;
}

__global__ __launch_bounds__(BLOCK) void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ global_scale_ptr,
    float global_scale_fallback,
    float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden, int block_size
) {
    constexpr int PAIRS_PER_FP4_BLOCK = 8; // block_size=16
    constexpr int SUBGROUP = 4;
    constexpr int PAIRS_PER_THREAD = 2;
    constexpr float INV_FP4_MAX = 1.0f / FP4_MAX;

    const int row = blockIdx.x;
    if (row >= batch) return;
    if (block_size != 16) return;

    const int tid = threadIdx.x;
    const int pairs = hidden >> 1;
    const int fp4_blocks = hidden >> 4;

    const __nv_bfloat162* input2 = reinterpret_cast<const __nv_bfloat162*>(input + row * hidden);
    __nv_bfloat162* residual2 = reinterpret_cast<__nv_bfloat162*>(residual + row * hidden);
    const __nv_bfloat162* weight2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    uint8_t* out_fp4_row = out_fp4 + row * pairs;
    uint8_t* out_scales_row = out_scales + row * fp4_blocks;

    float sumsq = 0.0f;
    for (int pair_idx = tid; pair_idx < pairs; pair_idx += BLOCK) {
        const __nv_bfloat162 h2 = __hadd2(input2[pair_idx], residual2[pair_idx]);
        residual2[pair_idx] = h2;
        const float2 hf = __bfloat1622float2(h2);
        sumsq += hf.x * hf.x + hf.y * hf.y;
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;

    sumsq = warp_reduce_sum(sumsq);
    __shared__ float warp_sums[BLOCK / 32];
    __shared__ float inv_rms_shared;
    if (lane == 0) {
        warp_sums[warp] = sumsq;
    }
    __syncthreads();

    if (warp == 0) {
        float total = (lane < (BLOCK / 32)) ? warp_sums[lane] : 0.0f;
        total = warp_reduce_sum(total);
        if (lane == 0) {
            inv_rms_shared = rsqrtf(total / static_cast<float>(hidden) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_shared;
    const float global_scale = global_scale_ptr != nullptr ? global_scale_ptr[0] : global_scale_fallback;

    const int group = tid / SUBGROUP;
    const int subgroup_lane = tid & (SUBGROUP - 1);
    const int groups_per_block = BLOCK / SUBGROUP;

    for (int fp4_block = group; fp4_block < fp4_blocks; fp4_block += groups_per_block) {
        const int pair_idx0 = fp4_block * PAIRS_PER_FP4_BLOCK + subgroup_lane * PAIRS_PER_THREAD;
        const int pair_idx1 = pair_idx0 + 1;

        const float2 h0 = __bfloat1622float2(residual2[pair_idx0]);
        const float2 h1 = __bfloat1622float2(residual2[pair_idx1]);
        const float2 w0 = __bfloat1622float2(weight2[pair_idx0]);
        const float2 w1 = __bfloat1622float2(weight2[pair_idx1]);

        const float y0 = h0.x * inv_rms * w0.x;
        const float y1 = h0.y * inv_rms * w0.y;
        const float y2 = h1.x * inv_rms * w1.x;
        const float y3 = h1.y * inv_rms * w1.y;
        float local_max = fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
        float block_max = subgroup_reduce_max<4>(local_max);

        float qmul = 0.0f;
        if (subgroup_lane == 0) {
            const float scale = global_scale * block_max * INV_FP4_MAX;
            const uint8_t scale_bits = float_to_e4m3(scale);
            out_scales_row[fp4_block] = scale_bits;
            const float scale_q = e4m3_to_float(scale_bits);
            qmul = scale_q > 0.0f ? (global_scale / scale_q) : 0.0f;
        }
        qmul = __shfl_sync(0xffffffffu, qmul, 0, 4);

        const uint8_t q0 = float_to_fp4_e2m1(y0 * qmul);
        const uint8_t q1 = float_to_fp4_e2m1(y1 * qmul);
        const uint8_t q2 = float_to_fp4_e2m1(y2 * qmul);
        const uint8_t q3 = float_to_fp4_e2m1(y3 * qmul);
        out_fp4_row[pair_idx0] = static_cast<uint8_t>(q0 | (q1 << 4));
        out_fp4_row[pair_idx1] = static_cast<uint8_t>(q2 | (q3 << 4));
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
    float gs_fallback = 1.0f;
    if (global_scale.has_value()) {
        TORCH_CHECK(global_scale->numel() == 1);
        TORCH_CHECK(global_scale->dtype() == torch::kFloat32);
        if (global_scale->is_cuda()) {
            gs_ptr = global_scale->data_ptr<float>();
        } else {
            gs_fallback = global_scale->item<float>();
        }
    }

    auto fp4_raw = torch::empty({batch, hidden / 2}, input.options().dtype(torch::kUInt8));
    auto scales_raw = torch::empty({batch, hidden / (int)block_size}, input.options().dtype(torch::kUInt8));

    fused_add_rmsnorm_nvfp4_kernel<<<batch, BLOCK>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        gs_ptr, gs_fallback, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
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
