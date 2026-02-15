"""
AXS-6 Neural Network Modules (Legacy V1)
==========================================

.. deprecated:: 0.3.0
    Use :mod:`axs.unified` instead.  V1 modules lack the fused NF5 LUT,
    Triton kernel support, mixed-precision training, and distributed
    gradient compression available in the unified API.

    Migration::

        # Old (V1)
        from axs.nn import AXSLinear, convert_to_axs

        # New (unified)
        from axs.unified import AXSLinearUnified, convert_to_axs_unified

Drop-in replacements for standard PyTorch layers that internally quantize
weights and activations to AXS-6 format during the forward pass.

These modules follow the mixed-precision training paradigm:
  - Master weights stored in FP32 (for optimizer updates)
  - Forward pass uses AXS-6 quantized weights and activations
  - Gradients flow through a straight-through estimator (STE)
  - Gradient quantization uses stochastic rounding for unbiasedness
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "axs.nn is deprecated since v0.3.0. Use axs.unified instead. "
    "V1 modules lack fused NF5 LUT, Triton kernels, mixed-precision, "
    "and distributed support. See README for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

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
