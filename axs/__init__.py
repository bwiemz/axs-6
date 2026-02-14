"""
AXS-6: Adaptive eXponent Sharing — 6-Bit Training Format
=========================================================

A novel 6-bit block-scaled numerical format for efficient deep learning training.

Key features:
  - 21% memory reduction vs FP8 with 4× better intra-block precision
  - Shared 8-bit exponent per block of 32 values
  - Stochastic rounding for unbiased gradient quantization
  - Error feedback mechanism for iterative training
  - Drop-in PyTorch module replacements
  - Optional Triton GPU kernels for hardware acceleration

Quick start::

    import torch
    from axs import quantize, dequantize
    from axs.nn import AXSLinear

    # Quantize a tensor
    x = torch.randn(128, 256)
    x_axs = quantize(x, block_size=32)
    x_restored = dequantize(x_axs)

    # Drop-in layer replacement
    layer = AXSLinear(256, 128)
    output = layer(x)

"""

from axs.core import (
    AXSBlock,
    AXSTensor,
    BlockConfig,
    axs_decode_block,
    axs_encode_block,
    dequantize,
    quantize,
)
from axs.quantize import (
    RoundingMode,
    quantize_with_error_feedback,
    quantize_stochastic,
    quantize_nearest,
)

__version__ = "0.1.0"
__all__ = [
    # Core types
    "AXSBlock",
    "AXSTensor",
    "BlockConfig",
    # Core functions
    "quantize",
    "dequantize",
    "axs_encode_block",
    "axs_decode_block",
    # Quantization
    "RoundingMode",
    "quantize_nearest",
    "quantize_stochastic",
    "quantize_with_error_feedback",
]
