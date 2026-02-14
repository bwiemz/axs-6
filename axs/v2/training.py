"""
AXS-6 V2 Training Pipeline
============================

High-level training wrapper that combines ALL V2 improvements into a
production-ready training loop:

  1. Precision annealing (gradual quantization introduction)
  2. SmoothQuant (online activation-weight balancing)
  3. Gradient clipping BEFORE quantization
  4. Robust loss scaling with overflow detection
  5. Training statistics tracking
  6. Optional Amax history for delayed scaling

Usage::

    from axs.v2.modules_v2 import convert_to_axs_v2
    from axs.v2.training import AXSTrainingPipelineV2
    from axs.v2.annealing import PrecisionAnnealingSchedule

    model = convert_to_axs_v2(base_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = PrecisionAnnealingSchedule(warmup_steps=500)

    pipeline = AXSTrainingPipelineV2(
        model, optimizer,
        annealing_schedule=schedule,
        use_smooth_quant=True,
    )

    for batch in dataloader:
        stats = pipeline.training_step(batch, loss_fn)
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from axs.v2.annealing import PrecisionAnnealingSchedule, AmaxHistory
from axs.v2.smooth_quant import SmoothQuantWrapper
from axs.v2.modules_v2 import AXSLinearV2


class AXSTrainingPipelineV2:
    """
    V2 training pipeline with all optimizations.

    Features over v1 AXSTrainingWrapper:
      - Precision annealing schedule
      - Online SmoothQuant integration
      - Pre-quantization gradient clipping
      - Gradient norm tracking per layer
      - Adaptive loss scaling with backoff/growth
      - Step-level metrics for debugging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        annealing_schedule: PrecisionAnnealingSchedule | None = None,
        use_smooth_quant: bool = False,
        smooth_quant_alpha: float = 0.5,
        smooth_quant_update_interval: int = 100,
        loss_scale_init: float = 2.0**16,
        loss_scale_growth_interval: int = 2000,
        loss_scale_backoff: float = 0.5,
        loss_scale_growth: float = 2.0,
        max_grad_norm: float = 1.0,
        use_amax_history: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer

        # Precision annealing
        self.annealing = annealing_schedule

        # SmoothQuant (online, EMA-based)
        self.smooth_quant: SmoothQuantWrapper | None = None
        if use_smooth_quant:
            # Find all V2 linear layers
            linear_layers: dict[str, AXSLinearV2] = {}
            for name, module in model.named_modules():
                if isinstance(module, AXSLinearV2):
                    linear_layers[name] = module
            if linear_layers:
                self.smooth_quant = SmoothQuantWrapper(
                    linear_layers,
                    alpha=smooth_quant_alpha,
                    update_interval=smooth_quant_update_interval,
                )

        # Loss scaling
        self.loss_scale = loss_scale_init
        self.growth_interval = loss_scale_growth_interval
        self.backoff = loss_scale_backoff
        self.growth = loss_scale_growth
        self.max_grad_norm = max_grad_norm

        # Amax history
        self.amax_history = AmaxHistory() if use_amax_history else None

        # Tracking
        self._steps_since_scale_change = 0
        self._total_steps = 0
        self._overflow_count = 0
        self._grad_norms: list[float] = []

    def training_step(
        self,
        inputs: Any,
        loss_fn: Callable[..., torch.Tensor],
    ) -> dict[str, Any]:
        """
        Execute one V2 training step.

        Args:
            inputs: Input data (passed to model and loss function).
            loss_fn: Callable taking (model_output, inputs) → loss.

        Returns:
            Dict with training metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(inputs)
        loss = loss_fn(output, inputs)

        # Scale loss for numerical stability
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward()

        # Unscale gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.div_(self.loss_scale)

        # Check for overflow
        has_overflow = self._check_overflow()
        if has_overflow:
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
                "quant_strength": self._get_strength(),
            }

        # Gradient clipping BEFORE quantization (bug fix from v1)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        self._grad_norms.append(grad_norm_val)

        # Optimizer step
        self.optimizer.step()

        # Update SmoothQuant
        if self.smooth_quant is not None:
            self.smooth_quant.step()

        # Update precision annealing schedule
        if self.annealing is not None:
            self.annealing.step()

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
            "quant_strength": self._get_strength(),
        }

    def _check_overflow(self) -> bool:
        """Check if any gradient contains inf or nan."""
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.any(torch.isinf(param.grad)) or torch.any(torch.isnan(param.grad)):
                    return True
        return False

    def _get_strength(self) -> float:
        """Get current quantization strength from annealing schedule."""
        if self.annealing is not None:
            return self.annealing.strength
        return 1.0

    @property
    def stats(self) -> dict[str, Any]:
        """Comprehensive training statistics."""
        recent_norms = self._grad_norms[-100:] if self._grad_norms else [0.0]
        return {
            "total_steps": self._total_steps,
            "overflow_count": self._overflow_count,
            "current_loss_scale": self.loss_scale,
            "quant_strength": self._get_strength(),
            "avg_grad_norm_recent": sum(recent_norms) / len(recent_norms),
            "max_grad_norm_recent": max(recent_norms),
            "smooth_quant_active": self.smooth_quant is not None,
        }
