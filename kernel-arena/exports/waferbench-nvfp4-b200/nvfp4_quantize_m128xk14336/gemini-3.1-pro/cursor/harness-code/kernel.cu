#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    uint32_t bits = __float_as_uint(val);
    uint8_t sign = (val < 0.0f) ? 8 : 0;
    bits &= 0x7FFFFFFF;
    
    uint8_t encoded;
    if (bits < 0x3fe00000) {
        if (bits < 0x3f400000) {
            encoded = (bits <= 0x3e800000) ? 0 : 1;
        } else {
            encoded = (bits <= 0x3fa00000) ? 2 : 3;
        }
    } else {
        if (bits < 0x40600000) {
            encoded = (bits <= 0x40200000) ? 4 : 5;
        } else {
            encoded = (bits <= 0x40a00000) ? 6 : 7;
        }
    }
    return sign | encoded;
}

__device__ __forceinline__ uint8_t float_to_e4m3(float val) {
    if (val <= 0.0f) return 0;
    if (val > 448.0f) return 0x7E;
    
    uint32_t bits = __float_as_uint(val);
    int exp = ((bits >> 23) & 0xFF) - 127;
    
    if (exp < -6) {
        int man = __float2int_rn(val * 512.0f);
        if (man > 7) man = 7;
        return man;
    }
    
    int biased_exp = exp + 7;
    uint32_t man_bits = bits & 0x7FFFFF;
    
    int man = man_bits >> 20;
    uint32_t rem = man_bits & 0xFFFFF;
    if (rem > 0x80000 || (rem == 0x80000 && (man & 1))) {
        man++;
    }
    
    if (man > 7) {
        man = 0;
        biased_exp++;
    }
    if (biased_exp > 15) {
        biased_exp = 15;
        man = 6;
    }
    return (biased_exp << 3) | man;
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    int biased_exp = (bits >> 3) & 0xF;
    int man = bits & 0x7;
    if (biased_exp == 0) {
        return float(man) * (1.0f / 512.0f);
    }
    uint32_t f_bits = ((biased_exp + 127 - 7) << 23) | (man << 20);
    return __uint_as_float(f_bits);
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t nibble) {
    uint8_t sign = (nibble >> 3) & 1;
    uint8_t exp  = (nibble >> 1) & 3;
    uint8_t man  = nibble & 1;
    if (exp == 0) {
        float val = man * 0.5f;
        return sign ? -val : val;
    }
    uint32_t f_bits = (sign << 31) | ((exp + 127 - 1) << 23) | (man << 22);
    return __uint_as_float(f_bits);
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

template <int BLOCK_SIZE>
__global__ void nvfp4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, const float* __restrict__ d_global_scale,
    uint8_t* __restrict__ output, uint8_t* __restrict__ block_scales,
    int M, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = M * (K / BLOCK_SIZE);
    if (idx >= total_blocks) return;

    int row = idx / (K / BLOCK_SIZE);
    int blk = idx % (K / BLOCK_SIZE);

    float global_scale = d_global_scale ? __ldg(d_global_scale) : 1.0f;
    
    int start_col = blk * BLOCK_SIZE;
    const float4* in_ptr = reinterpret_cast<const float4*>(&input[row * K + start_col]);
    
    float4 vec0 = __ldg(&in_ptr[0]);
    float4 vec1 = __ldg(&in_ptr[1]);
    
    __nv_bfloat162 h[8];
    h[0] = *reinterpret_cast<const __nv_bfloat162*>(&vec0.x);
    h[1] = *reinterpret_cast<const __nv_bfloat162*>(&vec0.y);
    h[2] = *reinterpret_cast<const __nv_bfloat162*>(&vec0.z);
    h[3] = *reinterpret_cast<const __nv_bfloat162*>(&vec0.w);
    h[4] = *reinterpret_cast<const __nv_bfloat162*>(&vec1.x);
    h[5] = *reinterpret_cast<const __nv_bfloat162*>(&vec1.y);
    h[6] = *reinterpret_cast<const __nv_bfloat162*>(&vec1.z);
    h[7] = *reinterpret_cast<const __nv_bfloat162*>(&vec1.w);

    uint32_t max_abs_int = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t bits = *reinterpret_cast<const uint32_t*>(&h[i]);
        bits &= 0x7FFF7FFF;
        uint32_t abs0 = bits & 0xFFFF;
        uint32_t abs1 = bits >> 16;
        max_abs_int = max(max_abs_int, abs0);
        max_abs_int = max(max_abs_int, abs1);
    }
    float max_abs = __uint_as_float(max_abs_int << 16) * global_scale;
    
    uint8_t bs_encoded = float_to_e4m3(max_abs / 6.0f);
    block_scales[row * (K / BLOCK_SIZE) + blk] = bs_encoded;
    
    float bs_val = e4m3_to_float(bs_encoded);
    float inv_bs = (bs_val > 0.0f) ? (1.0f / bs_val) : 0.0f;
    float combined_scale = global_scale * inv_bs;
    
    uint32_t out_packed[2];
    uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out_packed);
    
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float2 f2 = __bfloat1622float2(h[i]);
        float v0 = f2.x * combined_scale;
        float v1 = f2.y * combined_scale;
        out_bytes[i] = float_to_fp4_e2m1(v0) | (float_to_fp4_e2m1(v1) << 4);
    }
    
    uint2* out_ptr = reinterpret_cast<uint2*>(&output[row * (K / 2) + blk * 8]);
    *out_ptr = make_uint2(out_packed[0], out_packed[1]);
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

    int threads = 256;
    if (sf_vec_size == 16) {
        int blocks = (M * K + threads * 16 - 1) / (threads * 16);
        nvfp4_quantize_kernel<16><<<blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()), d_gs,
            packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), M, K);
    } else {
        // Fallback for other block sizes if needed
    }

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
