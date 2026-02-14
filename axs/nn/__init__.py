"""
AXS-6 Neural Network Modules
=============================

Drop-in replacements for standard PyTorch layers that internally quantize
weights and activations to AXS-6 format during the forward pass.

These modules follow the mixed-precision training paradigm:
  - Master weights stored in FP32 (for optimizer updates)
  - Forward pass uses AXS-6 quantized weights and activations
  - Gradients flow through a straight-through estimator (STE)
  - Gradient quantization uses stochastic rounding for unbiasedness
"""

from __future__ import annotations

from axs.nn.modules import (
    AXSConv2d,
    AXSEmbedding,
    AXSLayerNorm,
    AXSLinear,
    AXSMultiheadAttention,
    convert_to_axs,
)
from axs.nn.functional import (
    axs_linear,
    axs_matmul,
    axs_conv2d,
)
from axs.nn.optim import AXSAdamW

__all__ = [
    "AXSLinear",
    "AXSConv2d",
    "AXSEmbedding",
    "AXSLayerNorm",
    "AXSMultiheadAttention",
    "convert_to_axs",
    "axs_linear",
    "axs_matmul",
    "axs_conv2d",
    "AXSAdamW",
]
