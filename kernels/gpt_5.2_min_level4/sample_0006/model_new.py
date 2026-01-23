import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

hip_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

__global__ void gated_deltanet_recurrence_fwd(
    const float* __restrict__ q,     // [B,T,H,Dk]
    const float* __restrict__ k,     // [B,T,H,Dk]
    const float* __restrict__ v,     // [B,T,H,Dv]
    const float* __restrict__ alpha, // [B,T,H]
    const float* __restrict__ beta,  // [B,T,H]
    float* __restrict__ state,       // [B,H,Dv,Dk]
    float* __restrict__ out,         // [B,T,H,Dv]
    int B, int T, int H, int Dk, int Dv)
{
    int bh = (int)blockIdx.x;
    int b = bh / H;
    int h = bh - b * H;
    if (b >= B) return;

    // shared: sk(Dv)+err(Dv)+k(Dk)+q(Dk)+v(Dv)
    extern __shared__ float smem[];
    float* sk   = smem;
    float* err  = sk + Dv;
    float* sh_k = err + Dv;
    float* sh_q = sh_k + Dk;
    float* sh_v = sh_q + Dk;

    int tid = (int)threadIdx.x;
    int nthreads = (int)blockDim.x;

    // state pointer for this (b,h)
    float* S = state + ((b * H + h) * Dv * Dk);

    for (int t = 0; t < T; ++t) {
        const int base_qk = (((b * T + t) * H + h) * Dk);
        const int base_v  = (((b * T + t) * H + h) * Dv);

        for (int j = tid; j < Dk; j += nthreads) {
            sh_k[j] = k[base_qk + j];
            sh_q[j] = q[base_qk + j];
        }
        for (int i = tid; i < Dv; i += nthreads) {
            sh_v[i] = v[base_v + i];
        }
        __syncthreads();

        float a = alpha[(b * T + t) * H + h];
        float bt = beta[(b * T + t) * H + h];

        // sk = S@k
        if (tid < Dv) {
            float acc = 0.0f;
            const float* row = S + tid * Dk;
            #pragma unroll
            for (int j = 0; j < 128; ++j) {
                acc += row[j] * sh_k[j];
            }
            sk[tid] = acc;
        }
        __syncthreads();

        if (tid < Dv) err[tid] = sk[tid] - sh_v[tid];
        __syncthreads();

        // update S in global
        int Sd = Dv * Dk;
        for (int idx = tid; idx < Sd; idx += nthreads) {
            int i = idx / Dk;
            int j = idx - i * Dk;
            float s = S[idx];
            s = a * s - bt * err[i] * sh_k[j];
            S[idx] = s;
        }
        __syncthreads();

        // out = S@q
        if (tid < Dv) {
            float acc = 0.0f;
            const float* row = S + tid * Dk;
            #pragma unroll
            for (int j = 0; j < 128; ++j) {
                acc += row[j] * sh_q[j];
            }
            out[base_v + tid] = acc;
        }
        __syncthreads();
    }
}

torch::Tensor gated_deltanet_recurrence_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta)
{
    CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v); CHECK_CUDA(alpha); CHECK_CUDA(beta);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v); CHECK_CONTIGUOUS(alpha); CHECK_CONTIGUOUS(beta);
    CHECK_FLOAT(q); CHECK_FLOAT(k); CHECK_FLOAT(v); CHECK_FLOAT(alpha); CHECK_FLOAT(beta);

    int B = (int)q.size(0);
    int T = (int)q.size(1);
    int H = (int)q.size(2);
    int Dk = (int)q.size(3);
    int Dv = (int)v.size(3);

    TORCH_CHECK(Dk == 128, "This optimized kernel expects head_dim_qk=128");
    TORCH_CHECK(Dv == 128, "This optimized kernel expects head_dim_v=128");

    auto out = torch::empty_like(v);
    auto state = torch::zeros({B, H, Dv, Dk}, q.options());

    dim3 block(256);
    dim3 grid(B * H);

    size_t shmem = (size_t)(Dv + Dv + Dk + Dk + Dv) * sizeof(float);

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();

    hipLaunchKernelGGL(
        gated_deltanet_recurrence_fwd,
        grid, block,
        shmem, stream,
        (const float*)q.data_ptr<float>(),
        (const float*)k.data_ptr<float>(),
        (const float*)v.data_ptr<float>(),
        (const float*)alpha.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)state.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, T, H, Dk, Dv
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gated_deltanet_recurrence_hip", &gated_deltanet_recurrence_hip, "GatedDeltaNet recurrence (HIP)");
}
'''

_gdn_ext = load_inline(
    name='gdn_recurrence_ext2',
    cpp_sources='',
    cuda_sources=hip_src,
    functions=None,
    extra_cuda_cflags=['-O3'],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_short_conv = use_short_conv

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        if use_short_conv:
            self.q_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.k_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.v_conv = nn.Conv1d(
                num_heads * head_dim_v, num_heads * head_dim_v,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_v,
                padding=conv_kernel_size - 1
            )

        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        alpha = torch.sigmoid(self.a_proj(x))
        beta = torch.sigmoid(self.b_proj(x))

        k = k * self.scale

        o = _gdn_ext.gated_deltanet_recurrence_hip(
            q.contiguous(), k.contiguous(), v.contiguous(), alpha.contiguous(), beta.contiguous()
        )

        o = self.o_norm(o)
        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        return self.o_proj(o)


def get_inputs():
    batch_size = 4
    seq_len = 2048
    hidden_size = 2048
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)]


def get_init_inputs():
    return [2048, 16, 128, 128]
