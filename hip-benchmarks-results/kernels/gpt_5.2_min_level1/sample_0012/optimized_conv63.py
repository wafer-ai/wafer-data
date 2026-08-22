import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

miopen_cpp_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

static void check_miopen(miopenStatus_t s, const char* msg) {
    TORCH_CHECK(s == miopenStatusSuccess, msg, " miopenStatus=", (int)s);
}

struct ConvCache {
    bool inited = false;
    miopenHandle_t handle = nullptr;
    miopenTensorDescriptor_t xDesc = nullptr;
    miopenTensorDescriptor_t wDesc = nullptr;
    miopenTensorDescriptor_t yDesc = nullptr;
    miopenConvolutionDescriptor_t convDesc = nullptr;
    miopenConvFwdAlgorithm_t algo;
    size_t ws_size = 0;

    // Shape guards
    int N=0,C=0,H=0,W=0,K=0,Ho=0,Wo=0;
};

static ConvCache g;

static void init_if_needed(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& y) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int K = (int)w.size(0);
    int Ho = (int)y.size(2);
    int Wo = (int)y.size(3);

    if(g.inited) {
        // KernelBench fixed shape; if shape changes, re-init.
        if(g.N==N && g.C==C && g.H==H && g.W==W && g.K==K && g.Ho==Ho && g.Wo==Wo) return;
    }

    // (Re)initialize persistent objects
    if(g.handle) {
        // best-effort cleanup
        miopenDestroyConvolutionDescriptor(g.convDesc);
        miopenDestroyTensorDescriptor(g.xDesc);
        miopenDestroyTensorDescriptor(g.wDesc);
        miopenDestroyTensorDescriptor(g.yDesc);
        miopenDestroy(g.handle);
    }

    check_miopen(miopenCreate(&g.handle), "miopenCreate failed");
    check_miopen(miopenCreateTensorDescriptor(&g.xDesc), "create xDesc");
    check_miopen(miopenCreateTensorDescriptor(&g.wDesc), "create wDesc");
    check_miopen(miopenCreateTensorDescriptor(&g.yDesc), "create yDesc");
    check_miopen(miopenCreateConvolutionDescriptor(&g.convDesc), "create convDesc");

    check_miopen(miopenSet4dTensorDescriptor(g.xDesc, miopenFloat, N, C, H, W), "set xDesc");
    check_miopen(miopenSet4dTensorDescriptor(g.wDesc, miopenFloat, K, C, 3, 3), "set wDesc");
    check_miopen(miopenSet4dTensorDescriptor(g.yDesc, miopenFloat, N, K, Ho, Wo), "set yDesc");
    check_miopen(miopenInitConvolutionDescriptor(g.convDesc, miopenConvolution, 0, 0, 1, 1, 1, 1), "init convDesc");

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    check_miopen(miopenSetStream(g.handle, stream), "miopenSetStream failed");

    // Find best algorithm for this configuration.
    size_t max_ws = 0;
    check_miopen(miopenConvolutionForwardGetWorkSpaceSize(g.handle, g.wDesc, g.xDesc, g.convDesc, g.yDesc, &max_ws), "get ws size");
    auto ws = torch::empty({(long long)max_ws}, x.options().dtype(torch::kUInt8));

    int returnedAlgoCount = 0;
    miopenConvAlgoPerf_t perf[16];
    check_miopen(
        miopenFindConvolutionForwardAlgorithm(g.handle,
            g.xDesc, x.data_ptr(),
            g.wDesc, w.data_ptr(),
            g.convDesc,
            g.yDesc, y.data_ptr(),
            16,
            &returnedAlgoCount,
            perf,
            ws.data_ptr(),
            max_ws,
            false),
        "miopenFindConvolutionForwardAlgorithm failed");

    TORCH_CHECK(returnedAlgoCount > 0, "no miopen fwd algo returned");
    int best = 0;
    for(int i=1;i<returnedAlgoCount;i++) {
        if(perf[i].time < perf[best].time) best = i;
    }
    g.algo = perf[best].fwd_algo;
    g.ws_size = perf[best].memory;

    g.N=N; g.C=C; g.H=H; g.W=W; g.K=K; g.Ho=Ho; g.Wo=Wo;
    g.inited = true;
}

torch::Tensor conv3x3_miopen_forward(torch::Tensor x, torch::Tensor w) {
    CHECK_CUDA(x);
    CHECK_CUDA(w);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(w);
    CHECK_FLOAT(x);
    CHECK_FLOAT(w);
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "x/w must be 4D");
    TORCH_CHECK(w.size(2) == 3 && w.size(3) == 3, "only 3x3 supported");

    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int Ho = H - 3 + 1;
    int Wo = W - 3 + 1;

    auto y = torch::empty({x.size(0), w.size(0), Ho, Wo}, x.options());

    init_if_needed(x, w, y);

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    check_miopen(miopenSetStream(g.handle, stream), "miopenSetStream failed");

    torch::Tensor ws;
    void* ws_ptr = nullptr;
    if(g.ws_size > 0) {
        ws = torch::empty({(long long)g.ws_size}, x.options().dtype(torch::kUInt8));
        ws_ptr = ws.data_ptr();
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    check_miopen(
        miopenConvolutionForward(g.handle,
            &alpha,
            g.xDesc, x.data_ptr(),
            g.wDesc, w.data_ptr(),
            g.convDesc,
            g.algo,
            &beta,
            g.yDesc, y.data_ptr(),
            ws_ptr,
            g.ws_size),
        "miopenConvolutionForward failed");

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3x3_miopen_forward", &conv3x3_miopen_forward, "conv3x3 miopen forward");
}
"""

conv_ext = load_inline(
    name="conv3x3_miopen_ext_kb63",
    cpp_sources=miopen_cpp_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lMIOpen"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._ext = conv_ext
        assert kernel_size == 3 and stride == 1 and padding == 0 and dilation == 1 and groups == 1 and (not bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        w = self.conv2d.weight
        if not w.is_contiguous():
            w = w.contiguous()
        return self._ext.conv3x3_miopen_forward(x, w)


def get_inputs():
    batch_size = 16
    in_channels = 16
    width = 1024
    height = 1024
    x = torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    return [x]


def get_init_inputs():
    return [16, 128, 3]
