#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    return __nv_cvt_float_to_fp4(val, __NV_E2M1, cudaRoundNearest);
}

__device__ __forceinline__ uint8_t float_to_e4m3(float val) {
    return __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(bits, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&hr));
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

template <int HIDDEN, int BLOCK_SZ, int ROWS_PER_BLOCK>
__global__
void fused_add_rmsnorm_nvfp4_kernel_impl(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ global_scale_ptr, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch
) {
    const float global_scale = global_scale_ptr ? *global_scale_ptr : 1.0f;
    constexpr int THREADS_PER_ROW = BLOCK / ROWS_PER_BLOCK;
    constexpr int VEC = 8;
    constexpr int TOTAL_VECS = HIDDEN / VEC;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / THREADS_PER_ROW;
    constexpr int ELEMS_PER_THREAD = VECS_PER_THREAD * VEC;
    constexpr int HALF_HIDDEN = HIDDEN / 2;
    constexpr int NUM_SCALE_BLOCKS = HIDDEN / BLOCK_SZ;
    constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;

    const int tid = threadIdx.x;
    const int row_in_block = tid / THREADS_PER_ROW;
    const int ltid = tid % THREADS_PER_ROW;
    const int lane = ltid & 31;
    const int warp_id = ltid >> 5;
    const int pair_lane = ltid & 1;

    const int row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    if (row >= batch) return;

    const int4* wt_base = reinterpret_cast<const int4*>(weight);
    const float gs_over_fp4max = global_scale * (1.0f / FP4_MAX);

    __shared__ float smem[BLOCK / 32];
    float* my_smem = smem + row_in_block * WARPS_PER_ROW;

    static_assert(BLOCK_SZ == 16 && VEC == 8, "Assumes 2 adjacent threads per quant block");

    const int row_off = row * HIDDEN;
    const int4* inp_base = reinterpret_cast<const int4*>(input + row_off);
    int4* res_base = reinterpret_cast<int4*>(residual + row_off);

    float h[ELEMS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
        int vi = v * THREADS_PER_ROW + ltid;
        int4 in_v = inp_base[vi];
        int4 rs_v = res_base[vi];
        __nv_bfloat16* iv = reinterpret_cast<__nv_bfloat16*>(&in_v);
        __nv_bfloat16* rv = reinterpret_cast<__nv_bfloat16*>(&rs_v);
        #pragma unroll
        for (int k = 0; k < VEC; k++) {
            float val = __bfloat162float(iv[k]) + __bfloat162float(rv[k]);
            h[v * VEC + k] = val;
            rv[k] = __float2bfloat16(val);
            sum_sq += val * val;
        }
        res_base[vi] = rs_v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    if (lane == 0) my_smem[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < WARPS_PER_ROW) ? my_smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, offset);
        if (lane == 0) my_smem[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(my_smem[0] * (1.0f / float(HIDDEN)) + eps);

    uint8_t* fp4_row = out_fp4 + row * HALF_HIDDEN;
    uint8_t* sc_row = out_scales + row * NUM_SCALE_BLOCKS;

    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
        int vi = v * THREADS_PER_ROW + ltid;
        int4 w_v = wt_base[vi];
        __nv_bfloat16* wv = reinterpret_cast<__nv_bfloat16*>(&w_v);

        #pragma unroll
        for (int k = 0; k < VEC; k++)
            h[v * VEC + k] *= rms_inv * __bfloat162float(wv[k]);

        float amax0 = fmaxf(fabsf(h[v * VEC + 0]), fabsf(h[v * VEC + 1]));
        float amax1 = fmaxf(fabsf(h[v * VEC + 2]), fabsf(h[v * VEC + 3]));
        float amax2 = fmaxf(fabsf(h[v * VEC + 4]), fabsf(h[v * VEC + 5]));
        float amax3 = fmaxf(fabsf(h[v * VEC + 6]), fabsf(h[v * VEC + 7]));
        float my_amax = fmaxf(fmaxf(amax0, amax1), fmaxf(amax2, amax3));

        float partner_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
        float amax = fmaxf(my_amax, partner_amax);

        uint8_t scale_bits = float_to_e4m3(gs_over_fp4max * amax);
        float scale_f = e4m3_to_float(scale_bits);
        float inv_scale = __fdividef(global_scale, fmaxf(scale_f, 1e-30f));

        int blk_idx = v * THREADS_PER_ROW + ltid;
        int vec_col = blk_idx * VEC;

        if (pair_lane == 0)
            sc_row[vec_col / BLOCK_SZ] = scale_bits;

        uint8_t packed[4];
        #pragma unroll
        for (int k = 0; k < VEC; k += 2) {
            uint8_t nib0 = float_to_fp4_e2m1(h[v * VEC + k] * inv_scale);
            uint8_t nib1 = float_to_fp4_e2m1(h[v * VEC + k + 1] * inv_scale);
            packed[k / 2] = (nib1 << 4) | nib0;
        }
        *reinterpret_cast<uint32_t*>(fp4_row + vec_col / 2) = *reinterpret_cast<uint32_t*>(packed);
    }
}

// Wrapper removed - we call the template directly from host code

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
    torch::Tensor gs_tensor;
    if (global_scale.has_value()) {
        gs_tensor = global_scale->to(torch::kFloat32).contiguous();
        gs_ptr = gs_tensor.data_ptr<float>();
    }

    torch::Tensor fp4_raw, scales_raw;

    if (y_fp4.has_value()) {
        fp4_raw = y_fp4->view(torch::kUInt8);
    } else {
        fp4_raw = torch::empty({batch, hidden / 2}, input.options().dtype(torch::kUInt8));
    }

    if (block_scale.has_value()) {
        scales_raw = block_scale->view(torch::kUInt8);
    } else {
        scales_raw = torch::empty({batch, hidden / (int)block_size}, input.options().dtype(torch::kUInt8));
    }

    constexpr int ROWS_PER_BLOCK = 2;
    int grid = (batch + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    fused_add_rmsnorm_nvfp4_kernel_impl<8192, 16, ROWS_PER_BLOCK><<<grid, BLOCK>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        gs_ptr, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
        batch);

    torch::Tensor fp4_out = y_fp4.has_value() ? *y_fp4 : fp4_raw.view(c10::ScalarType::Float4_e2m1fn_x2);
    torch::Tensor scale_out = block_scale.has_value() ? *block_scale : scales_raw.view(c10::ScalarType::Float8_e4m3fn);
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
