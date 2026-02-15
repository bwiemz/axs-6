"""
AXS-6 Unified: Fused NF5 Warp Table Quantiser
================================================

This package merges V1's speed with V2's quality using a novel fused
NF5 warp table — a precomputed 1024-entry LUT that replaces the entire
encode → AXSTensor → decode pipeline with a single O(1) gather.

Performance (RTX 5070 Ti, 4096×4096):
  - V1 fake-quantise: 5.54 ms
  - V2 fake-quantise: 8.21 ms
  - **Unified fake-quantise: 3.96 ms** (1.40× faster than V1)

Quality (MSE, same tensor):
  - V1: 0.00116
  - V2: 0.00077
  - **Unified: 0.00077** (matches V2 within 0.1%)
"""

from axs.unified.backend import (
    BackendType,
    accelerated_fake_quantize,
    accelerated_linear,
    backend_info,
    clear_int8_weight_cache,
    detect_best_backend,
    get_backend,
    set_backend,
)
from axs.unified.functional_unified import (
    axs_linear_unified,
    axs_matmul_unified,
    fake_quantize_unified,
)
from axs.unified.modules_unified import (
    AXSEmbeddingUnified,
    AXSLayerNormUnified,
    AXSLinearUnified,
    AXSMultiheadAttentionUnified,
    convert_to_axs_unified,
)
from axs.unified.quantize_unified import (
    FUSED_NF5_LUT,
    NF5_CODEBOOK,
    REVERSE_NF5_LUT,
    dequantize_unified,
    fused_fake_quantize,
    quantization_error_unified,
    quantize_unified,
)
from axs.unified.training_unified import (
    AmaxEMA,
    AXSTrainingPipelineUnified,
)
from axs.unified.mixed_precision import (
    AXSLinearMixedPrecision,
    axs_linear_mixed_precision,
    convert_to_axs_mixed_precision,
    estimate_memory_savings,
)
from axs.unified.triton_kernels import (
    has_triton,
    triton_fused_fake_quantize,
    triton_fused_linear,
)

# Distributed support (lazy import — torch.distributed may not be available)
try:
    from axs.unified.distributed import (
        AXS6GradCompressor,
        axs6_gradient_hook,
        axs6_gradient_hook_packed,
    )

    _HAS_DISTRIBUTED = True
except ImportError:
    _HAS_DISTRIBUTED = False

__all__ = [
    # Quantiser core
    "fused_fake_quantize",
    "quantize_unified",
    "dequantize_unified",
    "quantization_error_unified",
    "FUSED_NF5_LUT",
    "NF5_CODEBOOK",
    "REVERSE_NF5_LUT",
    # Functional
    "fake_quantize_unified",
    "axs_linear_unified",
    "axs_matmul_unified",
    # Modules
    "AXSLinearUnified",
    "AXSLayerNormUnified",
    "AXSEmbeddingUnified",
    "AXSMultiheadAttentionUnified",
    "convert_to_axs_unified",
    # Training
    "AXSTrainingPipelineUnified",
    "AmaxEMA",
    # Backend / acceleration
    "BackendType",
    "accelerated_fake_quantize",
    "accelerated_linear",
    "backend_info",
    "clear_int8_weight_cache",
    "detect_best_backend",
    "get_backend",
    "set_backend",
    # Triton kernels
    "has_triton",
    "triton_fused_fake_quantize",
    "triton_fused_linear",
    # Mixed-precision
    "AXSLinearMixedPrecision",
    "axs_linear_mixed_precision",
    "convert_to_axs_mixed_precision",
    "estimate_memory_savings",
    # Distributed (conditional)
    *((
        "AXS6GradCompressor",
        "axs6_gradient_hook",
        "axs6_gradient_hook_packed",
    ) if _HAS_DISTRIBUTED else ()),
]
