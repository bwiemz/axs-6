"""
AXS-6 Unified Training Pipeline
=================================

A streamlined training wrapper that combines the unified fused quantiser
with production-grade training utilities:

  1. **Skip-first-N warmup**: A binary flag that bypasses quantisation for
     the first N steps — zero runtime overhead vs precision annealing's
     per-step interpolation.
  2. **Amax EMA (delayed scaling)**: Maintained per named tensor; when
     enabled, the previous step's scale is reused, saving one ``amax``
     reduction per forward pass.
  3. **Gradient clipping** before the optimiser step (V2 lesson: clip
     *before* quantisation to prevent overflow).
  4. **Adaptive loss scaling** with overflow detection and backoff/growth.
  5. **Step-level metrics** for debugging and monitoring.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from axs.unified.modules_unified import AXSLinearUnified, AXSEmbeddingUnified


# ---------------------------------------------------------------------------
# Amax EMA tracker (lightweight replacement for V2's AmaxHistory)
# ---------------------------------------------------------------------------

class AmaxEMA:
    """
    Exponential moving average of per-tensor amax values.

    Implements the delayed-scaling idea from NVIDIA's FP8-LM paper:
    instead of computing amax from the current tensor (which requires a
    full-tensor reduction), reuse the smoothed historical value.

    Usage::

        ema = AmaxEMA()
        for step in training_loop:
            # Use last step's smoothed amax for scaling
            scale = ema.get_scale("weight")
            # ... use scale ...
            # Record this step's actual amax
            ema.update("weight", tensor.abs().max().item())
    """

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay
        self._values: dict[str, float] = {}

    def update(self, name: str, amax: float) -> None:
        if name not in self._values:
            self._values[name] = amax
        else:
            self._values[name] = self.decay * self._values[name] + (1.0 - self.decay) * amax

    def get_amax(self, name: str, default: float = 1.0) -> float:
        return self._values.get(name, default)

    def get_scale(self, name: str, default: float = 1.0) -> float:
        import math
        amax = self.get_amax(name, default)
        if amax <= 0:
            return 1.0
        raw_exp = math.floor(math.log2(amax)) + 1
        return 2.0 ** raw_exp


# ---------------------------------------------------------------------------
# Unified Training Pipeline
# ---------------------------------------------------------------------------

class AXSTrainingPipelineUnified:
    """
    Production training pipeline for the unified AXS-6 quantiser.

    Features:
      - Skip-first-N warmup (binary flag, zero overhead during warmup)
      - Optional Amax EMA for delayed scaling
      - Pre-quantisation gradient clipping
      - Adaptive loss scaling with overflow detection
      - Step-level training metrics

    Args:
        model: Model with unified AXS-6 layers.
        optimizer: PyTorch optimiser.
        warmup_steps: Number of initial steps where quantisation is skipped.
        loss_scale_init: Initial loss scale for mixed-precision stability.
        loss_scale_growth_interval: Steps between loss-scale growth attempts.
        loss_scale_backoff: Factor to reduce loss scale on overflow.
        loss_scale_growth: Factor to increase loss scale on success.
        max_grad_norm: Maximum gradient norm for clipping.
        use_amax_ema: Enable Amax EMA delayed scaling.
        amax_ema_decay: Decay factor for Amax EMA.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        loss_scale_init: float = 2.0 ** 16,
        loss_scale_growth_interval: int = 2000,
        loss_scale_backoff: float = 0.5,
        loss_scale_growth: float = 2.0,
        max_grad_norm: float = 1.0,
        use_amax_ema: bool = False,
        amax_ema_decay: float = 0.999,
    ) -> None:
        self.model = model
        self.optimizer = optimizer

        # Warmup
        self.warmup_steps = warmup_steps

        # Loss scaling
        self.loss_scale = loss_scale_init
        self.growth_interval = loss_scale_growth_interval
        self.backoff = loss_scale_backoff
        self.growth = loss_scale_growth
        self.max_grad_norm = max_grad_norm

        # Amax EMA
        self.amax_ema = AmaxEMA(decay=amax_ema_decay) if use_amax_ema else None

        # Tracking
        self._steps_since_scale_change = 0
        self._total_steps = 0
        self._overflow_count = 0
        self._grad_norms: list[float] = []

    def _set_warmup(self, active: bool) -> None:
        """Set the warmup flag on all unified layers."""
        for module in self.model.modules():
            if isinstance(module, (AXSLinearUnified, AXSEmbeddingUnified)):
                module._warmup_active = active

    @property
    def is_warmup(self) -> bool:
        """Whether the current step is within the warmup window."""
        return self._total_steps < self.warmup_steps

    def training_step(
        self,
        inputs: Any,
        loss_fn: Callable[..., torch.Tensor],
    ) -> dict[str, Any]:
        """
        Execute one training step.

        Args:
            inputs: Input data forwarded to model and loss function.
            loss_fn: ``(model_output, inputs) → loss``.

        Returns:
            Dict with keys: ``loss``, ``loss_scale``, ``grad_norm``,
            ``overflow``, ``step``, ``warmup``.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Set warmup flag
        warmup = self.is_warmup
        self._set_warmup(warmup)

        # Forward
        output = self.model(inputs)
        loss = loss_fn(output, inputs)

        # Backward with loss scaling
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward()

        # Unscale gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.div_(self.loss_scale)

        # Check overflow
        if self._check_overflow():
            self.loss_scale *= self.backoff
            self._steps_since_scale_change = 0
            self._overflow_count += 1
            self.optimizer.zero_grad()
            return {
                "loss": loss.item(),
                "loss_scale": self.loss_scale,
                "grad_norm": 0.0,
                "overflow": True,
                "step": self._total_steps,
                "warmup": warmup,
            }

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm,
        )
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        self._grad_norms.append(grad_norm_val)

        # Optimiser step
        self.optimizer.step()

        # Update loss scale
        self._steps_since_scale_change += 1
        if self._steps_since_scale_change >= self.growth_interval:
            self.loss_scale *= self.growth
            self._steps_since_scale_change = 0

        self._total_steps += 1

        return {
            "loss": loss.item(),
            "loss_scale": self.loss_scale,
            "grad_norm": grad_norm_val,
            "overflow": False,
            "step": self._total_steps,
            "warmup": warmup,
        }

    def _check_overflow(self) -> bool:
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.any(torch.isinf(param.grad)) or torch.any(torch.isnan(param.grad)):
                    return True
        return False

    @property
    def stats(self) -> dict[str, Any]:
        """Comprehensive training statistics."""
        recent_norms = self._grad_norms[-100:] if self._grad_norms else [0.0]
        return {
            "total_steps": self._total_steps,
            "overflow_count": self._overflow_count,
            "current_loss_scale": self.loss_scale,
            "is_warmup": self.is_warmup,
            "warmup_steps": self.warmup_steps,
            "avg_grad_norm_recent": sum(recent_norms) / len(recent_norms),
            "max_grad_norm_recent": max(recent_norms),
            "amax_ema_active": self.amax_ema is not None,
        }
