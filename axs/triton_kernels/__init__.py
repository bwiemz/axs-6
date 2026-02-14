"""
AXS-6 Triton GPU Kernels
=========================

High-performance GPU kernels for AXS-6 quantization and dequantization
written in Triton. These compile to optimized CUDA PTX code and provide
significant speedups over the pure-PyTorch reference implementation.

Requires: ``pip install triton>=2.1.0``
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

__all__ = [
    "TRITON_AVAILABLE",
    "triton_quantize_kernel",
    "triton_dequantize_kernel",
    "triton_quantize",
    "triton_dequantize",
]
