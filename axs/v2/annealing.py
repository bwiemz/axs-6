"""
Precision Annealing & Amax History — Training Stability
========================================================

Two techniques for stable, high-quality low-precision training:

1. **Precision Annealing**: Start training in higher precision (FP32/FP16)
   and gradually introduce quantization noise. This allows the model to
   first find a good region of the loss landscape, then adapt to the
   quantization grid. Think of it as curriculum learning for quantization.

   Schedule: quantization_strength(t) = min(1.0, t / warmup_steps)
   During warmup, interpolate: x_q = (1-s)*x + s*quantize(x)

2. **Amax History (Delayed Scaling)**: Instead of computing per-block
   scales from the current tensor, maintain an exponential moving average
   of per-block amax values. This:
   - Prevents scale oscillation from batch-to-batch variance
   - Enables pre-computing scales for the next iteration
   - Follows NVIDIA Transformer Engine's approach

Both techniques compose well with NF5 + percentile clipping.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from axs.core import (
    AXS6_EXPONENT_BIAS,
    AXS6_MAX_MAGNITUDE,
    AXSTensor,
    DEFAULT_BLOCK_SIZE,
    VALID_BLOCK_SIZES,
)
from axs.v2.quantize_v2 import quantize_v2, dequantize_v2


# ---------------------------------------------------------------------------
# Precision Annealing
# ---------------------------------------------------------------------------

class PrecisionAnnealingSchedule:
    """
    Controls the quantization strength over training.

    During warmup, the effective output is a linear interpolation:
        output = (1 - strength) * fp32_value + strength * quantized_value

    This allows the model to gradually adapt to quantization noise.

    Args:
        warmup_steps: Number of steps to linearly ramp up quantization.
        start_strength: Initial quantization strength (0 = no quant).
        end_strength: Final quantization strength (1 = full quant).
        cooldown_steps: Optional cooldown period at the end of training
            where quantization strength is reduced for fine-tuning.
    """

    def __init__(
        self,
        warmup_steps: int = 1000,
        start_strength: float = 0.0,
        end_strength: float = 1.0,
        cooldown_steps: int = 0,
        total_steps: int | None = None,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.start_strength = start_strength
        self.end_strength = end_strength
        self.cooldown_steps = cooldown_steps
        self.total_steps = total_steps
        self._current_step = 0

    def step(self) -> None:
        """Advance the schedule by one step."""
        self._current_step += 1

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def strength(self) -> float:
        """Current quantization strength in [0, 1]."""
        step = self._current_step

        # Warmup phase
        if step < self.warmup_steps:
            t = step / max(1, self.warmup_steps)
            return self.start_strength + t * (self.end_strength - self.start_strength)

        # Cooldown phase (if configured)
        if self.cooldown_steps > 0 and self.total_steps is not None:
            cooldown_start = self.total_steps - self.cooldown_steps
            if step >= cooldown_start:
                t = (step - cooldown_start) / max(1, self.cooldown_steps)
                # Linearly decrease from end_strength to 0.5 * end_strength
                return self.end_strength * (1.0 - 0.5 * t)

        return self.end_strength

    @property
    def is_warmup(self) -> bool:
        return self._current_step < self.warmup_steps


def annealed_fake_quantize(
    tensor: torch.Tensor,
    strength: float,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    use_nf5: bool = True,
    clip_percentile: float | None = 99.9,
) -> torch.Tensor:
    """
    Fake-quantize with precision annealing.

    When strength < 1.0, interpolates between the original tensor and
    the quantized version:

        output = (1 - strength) * tensor + strength * quantized_tensor

    Args:
        tensor: Input tensor.
        strength: Quantization strength in [0, 1].
        block_size: AXS-6 block size.
        rounding: Rounding mode.
        use_nf5: Whether to use NormalFloat grid.
        clip_percentile: Percentile for scale computation.

    Returns:
        Annealed fake-quantized tensor.
    """
    if strength <= 0.0:
        return tensor
    if strength >= 1.0:
        axs = quantize_v2(tensor, block_size, rounding, use_nf5, clip_percentile)
        return dequantize_v2(axs, use_nf5)

    # Interpolate
    axs = quantize_v2(tensor, block_size, rounding, use_nf5, clip_percentile)
    quantized = dequantize_v2(axs, use_nf5)
    return (1.0 - strength) * tensor + strength * quantized


# ---------------------------------------------------------------------------
# Amax History (Delayed Scaling)
# ---------------------------------------------------------------------------

class AmaxHistory:
    """
    Maintains exponential moving average of per-tensor amax values.

    Following NVIDIA Transformer Engine's approach, this provides stable
    scaling by using historical amax values instead of per-batch values.

    Key benefits:
      - Reduces scale oscillation
      - Enables pre-computing scales for the next step
      - Smooths out anomalous batches

    Usage::

        history = AmaxHistory()
        for step in training_loop:
            scale = history.get_scale(tensor_name)
            # ... use scale for quantization ...
            history.update(tensor_name, tensor.abs().max())
    """

    def __init__(
        self,
        window_size: int = 16,
        ema_decay: float = 0.999,
        mode: Literal["ema", "max_window"] = "ema",
    ) -> None:
        self.window_size = window_size
        self.ema_decay = ema_decay
        self.mode = mode
        self._ema: dict[str, float] = {}
        self._history: dict[str, list[float]] = {}

    def update(self, name: str, amax: float) -> None:
        """Record a new amax observation."""
        if self.mode == "ema":
            if name not in self._ema:
                self._ema[name] = amax
            else:
                self._ema[name] = self.ema_decay * self._ema[name] + (1 - self.ema_decay) * amax
        else:
            if name not in self._history:
                self._history[name] = []
            self._history[name].append(amax)
            if len(self._history[name]) > self.window_size:
                self._history[name].pop(0)

    def get_amax(self, name: str, default: float = 1.0) -> float:
        """Get the smoothed amax value for a named tensor."""
        if self.mode == "ema":
            return self._ema.get(name, default)
        else:
            if name not in self._history or len(self._history[name]) == 0:
                return default
            return max(self._history[name])

    def get_scale(self, name: str, default: float = 1.0) -> float:
        """Get the delayed scale factor (power-of-2 ceiling of smoothed amax)."""
        import math
        amax = self.get_amax(name, default)
        if amax <= 0:
            return 1.0
        raw_exp = math.floor(math.log2(amax)) + 1
        return 2.0 ** raw_exp


# ---------------------------------------------------------------------------
# Combined V2 Fake Quantize with all improvements
# ---------------------------------------------------------------------------

class FakeQuantizeV2(nn.Module):
    """
    Module-level V2 fake quantizer combining all improvements:
      - NormalFloat grid
      - Percentile clipping
      - Precision annealing
      - Amax history (optional)

    This is the recommended replacement for functional.fake_quantize
    in v2 training pipelines.
    """

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        use_nf5: bool = True,
        clip_percentile: float | None = 99.9,
        annealing_schedule: PrecisionAnnealingSchedule | None = None,
        use_amax_history: bool = False,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.use_nf5 = use_nf5
        self.clip_percentile = clip_percentile
        self.schedule = annealing_schedule
        self.amax_history = AmaxHistory() if use_amax_history else None

    def forward(
        self,
        tensor: torch.Tensor,
        rounding: Literal["nearest", "stochastic"] = "nearest",
    ) -> torch.Tensor:
        strength = self.schedule.strength if self.schedule is not None else 1.0

        return annealed_fake_quantize(
            tensor,
            strength=strength,
            block_size=self.block_size,
            rounding=rounding,
            use_nf5=self.use_nf5,
            clip_percentile=self.clip_percentile,
        )
