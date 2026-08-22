import torch
import torch.nn as nn


class Model(nn.Module):
    """
    FP8-simulated Matrix Multiplication that works on MI300X.
    
    This implements the same math as the FP8 tensor core path but uses
    standard FP16 operations with simulated FP8 quantization.
    """

    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3

        # FP8 format specifications
        if use_e4m3:
            self.fp8_dtype = torch.float8_e4m3fn
            self.fp8_max = 448.0  # Max representable value in E4M3
        else:
            self.fp8_dtype = torch.float8_e5m2
            self.fp8_max = 57344.0  # Max representable value in E5M2

        # Weight matrix stored in FP16
        self.weight = nn.Parameter(torch.randn(K, N) * 0.02)

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-tensor scale for FP8 quantization."""
        amax = x.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        return scale

    def quantize_to_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Quantize FP16/BF16 tensor to FP8."""
        x_scaled = x * scale
        x_clamped = x_scaled.clamp(-self.fp8_max, self.fp8_max)
        return x_clamped.to(self.fp8_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FP8-simulated matmul: x @ weight
        
        This simulates FP8 quantization then does matmul in FP16.
        """
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Reshape for matmul: (batch, seq, K) -> (batch*seq, K)
        x_2d = x.view(-1, self.K)

        # Compute scales for dynamic quantization
        x_scale = self.compute_scale(x_2d)
        w_scale = self.compute_scale(self.weight)

        # Quantize to FP8 then back to FP16 (simulating quantization noise)
        x_fp8 = self.quantize_to_fp8(x_2d, x_scale)
        x_dequant = x_fp8.to(input_dtype) / x_scale
        
        w_fp8 = self.quantize_to_fp8(self.weight, w_scale)
        w_dequant = w_fp8.to(input_dtype) / w_scale

        # Standard matmul on dequantized values
        out = torch.mm(x_dequant, w_dequant)

        return out.view(batch_size, seq_len, self.N)


# Configuration sized for H100/B200 tensor cores
batch_size = 8
seq_len = 2048
M = batch_size * seq_len  # Total rows
K = 4096  # Hidden dimension
N = 4096  # Output dimension
use_e4m3 = True  # E4M3 is more common for weights/activations


def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]


def get_init_inputs():
    return [M, K, N, use_e4m3]
