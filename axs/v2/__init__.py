"""
AXS-6 v2: Advanced Quantization Techniques
===========================================

This module contains the next-generation optimizations that make AXS-6
a compelling alternative to FP8/BF16 for real-world pretraining.

Key improvements over v1:
  1. NormalFloat quantization grid — non-uniform codes matched to Gaussian weights
  2. Percentile clipping — outlier-resilient scale computation
  3. Hadamard rotation — spread outliers across block dimensions
  4. SmoothQuant — migrate quantization difficulty from activations to weights
  5. Precision annealing — start FP32, gradually increase quantization
  6. Amax history — delayed scaling with exponential moving average
  7. Per-layer adaptive block sizing
"""

from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
from axs.v2.hadamard import apply_hadamard_rotation, invert_hadamard_rotation
from axs.v2.smooth_quant import SmoothQuantCalibrator, SmoothQuantWrapper
from axs.v2.annealing import (
    PrecisionAnnealingSchedule,
    AmaxHistory,
    FakeQuantizeV2,
    annealed_fake_quantize,
)
from axs.v2.functional_v2 import fake_quantize_v2, axs_linear_v2, axs_matmul_v2
from axs.v2.modules_v2 import (
    AXSLinearV2,
    AXSLayerNormV2,
    AXSEmbeddingV2,
    AXSMultiheadAttentionV2,
    convert_to_axs_v2,
)
from axs.v2.training import AXSTrainingPipelineV2

__all__ = [
    "quantize_v2", "dequantize_v2", "fake_quantize_v2",
    "apply_hadamard_rotation", "invert_hadamard_rotation",
    "SmoothQuantCalibrator", "SmoothQuantWrapper",
    "PrecisionAnnealingSchedule", "AmaxHistory", "FakeQuantizeV2",
    "annealed_fake_quantize",
    "axs_linear_v2", "axs_matmul_v2",
    "AXSLinearV2", "AXSLayerNormV2", "AXSEmbeddingV2",
    "AXSMultiheadAttentionV2", "convert_to_axs_v2",
    "AXSTrainingPipelineV2",
]
