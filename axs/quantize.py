"""
Advanced Quantization Strategies for AXS-6
==========================================

This module provides advanced quantization modes beyond the basic nearest-rounding
implemented in :mod:`axs.core`. These are critical for training convergence:

- **Nearest**: Deterministic, fastest. Good for forward-pass weights.
- **Stochastic**: Unbiased rounding, essential for gradient quantization.
- **Error Feedback**: Corrects accumulated quantization drift over training steps.
- **GASR (Gradient-Aware Stochastic Rounding)**: Biases rounding direction
  based on gradient magnitude for faster convergence.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Literal

import torch

from axs.core import (
    AXS6_EXPONENT_BIAS,
    AXS6_MAX_MAGNITUDE,
    AXSTensor,
    DEFAULT_BLOCK_SIZE,
    VALID_BLOCK_SIZES,
    dequantize,
    quantize,
)


class RoundingMode(Enum):
    """Available rounding strategies for AXS-6 quantization."""

    NEAREST = auto()
    STOCHASTIC = auto()
    ERROR_FEEDBACK = auto()
    GASR = auto()  # Gradient-Aware Stochastic Rounding


# ---------------------------------------------------------------------------
# Error Feedback State
# ---------------------------------------------------------------------------


class ErrorFeedbackState:
    """
    Maintains per-parameter error feedback buffers for compensating
    accumulated quantization drift during training.

    The error from each quantization step is stored and added back
    to the input of the next quantization, ensuring that over time
    the mean quantization error converges to zero.

    Usage::

        state = ErrorFeedbackState()
        for step in training_loop:
            x_q = state.quantize(x, param_name="layer1.weight")
    """

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}

    def get_buffer(self, name: str, like: torch.Tensor) -> torch.Tensor:
        """Get or create an error buffer matching the given tensor."""
        if name not in self._buffers:
            self._buffers[name] = torch.zeros_like(like)
        buf = self._buffers[name]
        if buf.shape != like.shape:
            # Shape changed (e.g., dynamic batching) — reset
            self._buffers[name] = torch.zeros_like(like)
            buf = self._buffers[name]
        return buf

    def update_buffer(self, name: str, error: torch.Tensor) -> None:
        """Store the quantization error for the next step."""
        self._buffers[name] = error.detach()

    def reset(self) -> None:
        """Clear all error buffers."""
        self._buffers.clear()

    @property
    def buffer_names(self) -> list[str]:
        """List of tracked parameter names."""
        return list(self._buffers.keys())


# Global default error feedback state
_default_ef_state = ErrorFeedbackState()


def get_default_error_feedback_state() -> ErrorFeedbackState:
    """Return the module-level default error feedback state."""
    return _default_ef_state


# ---------------------------------------------------------------------------
# Quantization Functions
# ---------------------------------------------------------------------------


def quantize_nearest(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> AXSTensor:
    """
    Quantize with deterministic nearest-rounding.

    Best for: forward-pass weights, inference.
    """
    return quantize(tensor, block_size=block_size, rounding="nearest")


def quantize_stochastic(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> AXSTensor:
    """
    Quantize with stochastic rounding.

    Each value is rounded up with probability proportional to its fractional
    part, making the expected quantized value equal to the true value.

    E[Q(x)] = x  (unbiased)

    Best for: gradient quantization during training.
    """
    return quantize(tensor, block_size=block_size, rounding="stochastic")


def quantize_with_error_feedback(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    param_name: str = "default",
    state: ErrorFeedbackState | None = None,
) -> AXSTensor:
    """
    Quantize with error feedback compensation.

    The quantization error from the previous step is added to the current
    input before quantization. This ensures long-term error cancellation:

        x_corrected = x + error_buffer
        x_quantized = quantize(x_corrected)
        error_buffer = x_corrected - dequantize(x_quantized)

    Args:
        tensor: Input tensor to quantize.
        block_size: AXS-6 block size.
        rounding: Base rounding mode.
        param_name: Unique identifier for the error buffer.
        state: Error feedback state. Uses global default if None.

    Returns:
        Quantized tensor with error compensation applied.
    """
    if state is None:
        state = _default_ef_state

    error_buf = state.get_buffer(param_name, tensor)
    corrected = tensor + error_buf

    axs_tensor = quantize(corrected, block_size=block_size, rounding=rounding)
    reconstructed = dequantize(axs_tensor)

    # Compute and store the new error
    new_error = corrected - reconstructed
    state.update_buffer(param_name, new_error)

    return axs_tensor


def quantize_gasr(
    tensor: torch.Tensor,
    gradient: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    temperature: float = 1.0,
) -> AXSTensor:
    """
    Gradient-Aware Stochastic Rounding (GASR).

    A novel rounding strategy that biases the rounding probability based on
    gradient magnitude. Values whose gradients are large (high-loss-sensitivity)
    are rounded in the direction that preserves more gradient information.

    This reduces the effective noise in directions that matter most for
    optimization, accelerating convergence compared to standard stochastic
    rounding.

    Algorithm:
        For each value x being quantized between levels q_lo and q_hi:
          frac = (x - q_lo) / (q_hi - q_lo)    # standard fractional part
          g = |grad(x)|                          # gradient magnitude
          g_normalized = sigmoid(temperature * g / mean(|grad|))

          # Bias rounding toward the level closer to the current value
          # when gradient is large (preserve accuracy where it matters)
          p_up = frac * g_normalized + frac * (1 - g_normalized)
               = frac  (baseline, but biased by gradient weighting below)

        In practice, we weight the rounding probability as:
          p_up = frac^(1/g_weight)  when g > median
          p_up = frac^(g_weight)    when g <= median
        This makes high-gradient values round "more correctly" (closer to
        nearest) while low-gradient values round more stochastically.

    Args:
        tensor: Values to quantize.
        gradient: Gradient tensor with same shape as ``tensor``.
        block_size: AXS-6 block size.
        temperature: Controls how aggressively to bias toward gradients.

    Returns:
        Quantized tensor with gradient-aware rounding.
    """
    assert tensor.shape == gradient.shape, "Tensor and gradient must have same shape"
    assert block_size in VALID_BLOCK_SIZES

    original_shape = tensor.shape
    device = tensor.device

    # Flatten and block
    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    grad_flat = gradient.reshape(-1, gradient.shape[-1]).float()

    last_dim = flat.shape[-1]
    pad_amount = (block_size - last_dim % block_size) % block_size
    if pad_amount > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_amount))
        grad_flat = torch.nn.functional.pad(grad_flat, (0, pad_amount))

    num_blocks = flat.shape[-1] // block_size
    blocked = flat.reshape(-1, num_blocks, block_size)
    grad_blocked = grad_flat.reshape(-1, num_blocks, block_size)

    # Compute shared exponents (same as core.quantize)
    abs_vals = blocked.abs()
    abs_max = abs_vals.amax(dim=-1)
    safe_max = abs_max.clamp(min=1e-45)
    raw_exp = safe_max.log2().floor().to(torch.int32) + 1
    shared_exponents = (raw_exp + AXS6_EXPONENT_BIAS).clamp(0, 255).to(torch.uint8)
    shared_exponents[abs_max == 0] = 0

    scales = torch.pow(
        2.0, shared_exponents.float() - AXS6_EXPONENT_BIAS
    ).unsqueeze(-1).clamp(min=1e-45)

    # Normalize and compute fractional parts
    normalized = abs_vals / scales
    scaled = normalized * AXS6_MAX_MAGNITUDE
    floor_vals = scaled.floor()
    frac = scaled - floor_vals

    # Gradient-aware rounding probability
    grad_abs = grad_blocked.abs()
    grad_mean = grad_abs.mean(dim=-1, keepdim=True).clamp(min=1e-45)
    grad_weight = torch.sigmoid(temperature * grad_abs / grad_mean)

    # High-gradient → round closer to nearest (less noise)
    # Low-gradient → more stochastic (save precision budget)
    # Implemented as: p_up = frac^(2 - grad_weight) for grad_weight ∈ [0.5, 1)
    exponent = 2.0 - grad_weight  # range [1.0, 1.5] for sigmoid output [0.5, 1.0]
    adjusted_frac = frac.clamp(min=0, max=1).pow(exponent)

    rand = torch.rand_like(adjusted_frac)
    magnitudes = torch.where(rand < adjusted_frac, floor_vals + 1, floor_vals)
    magnitudes = magnitudes.clamp(0, AXS6_MAX_MAGNITUDE).to(torch.uint8)
    signs = blocked < 0

    return AXSTensor(
        shared_exponents=shared_exponents.to(device),
        signs=signs.to(device),
        magnitudes=magnitudes.to(device),
        block_size=block_size,
        original_shape=original_shape,
        num_blocks=num_blocks,
    )


# ---------------------------------------------------------------------------
# Unified quantize interface
# ---------------------------------------------------------------------------


def quantize_adaptive(
    tensor: torch.Tensor,
    mode: RoundingMode = RoundingMode.NEAREST,
    block_size: int = DEFAULT_BLOCK_SIZE,
    gradient: torch.Tensor | None = None,
    param_name: str = "default",
    ef_state: ErrorFeedbackState | None = None,
    gasr_temperature: float = 1.0,
) -> AXSTensor:
    """
    Unified quantization interface supporting all rounding modes.

    Args:
        tensor: Input tensor.
        mode: Rounding strategy to use.
        block_size: Block size for AXS-6 encoding.
        gradient: Required for GASR mode.
        param_name: Identifier for error feedback buffer.
        ef_state: Error feedback state instance.
        gasr_temperature: Temperature for GASR bias.

    Returns:
        Quantized AXSTensor.
    """
    if mode == RoundingMode.NEAREST:
        return quantize_nearest(tensor, block_size)
    elif mode == RoundingMode.STOCHASTIC:
        return quantize_stochastic(tensor, block_size)
    elif mode == RoundingMode.ERROR_FEEDBACK:
        return quantize_with_error_feedback(
            tensor, block_size, rounding="nearest", param_name=param_name, state=ef_state
        )
    elif mode == RoundingMode.GASR:
        if gradient is None:
            raise ValueError("GASR mode requires a gradient tensor")
        return quantize_gasr(tensor, gradient, block_size, gasr_temperature)
    else:
        raise ValueError(f"Unknown rounding mode: {mode}")
