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
  6. **Checkpoint save/load**: Full training state (model, optimiser,
     scheduler, step counter, loss scale, RNG) for crash recovery.
  7. **LR scheduler integration**: Optional scheduler stepped per-batch
     or per-epoch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from axs.unified.modules_unified import AXSLinearUnified, AXSEmbeddingUnified

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency — resolved at runtime
_MIXED_PRECISION_TYPES: tuple[type, ...] | None = None


def _get_warmup_types() -> tuple[type, ...]:
    """Return all module types that have a ``_warmup_active`` flag."""
    global _MIXED_PRECISION_TYPES
    if _MIXED_PRECISION_TYPES is None:
        try:
            from axs.unified.mixed_precision import AXSLinearMixedPrecision
            _MIXED_PRECISION_TYPES = (
                AXSLinearUnified,
                AXSEmbeddingUnified,
                AXSLinearMixedPrecision,
            )
        except ImportError:
            _MIXED_PRECISION_TYPES = (AXSLinearUnified, AXSEmbeddingUnified)
    return _MIXED_PRECISION_TYPES


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

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "values": dict(self._values)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = state["decay"]
        self._values = dict(state["values"])


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
      - Checkpoint save/load for crash recovery
      - LR scheduler integration

    Args:
        model: Model with unified AXS-6 layers (or BF16 mixed-precision).
        optimizer: PyTorch optimiser.
        scheduler: Optional LR scheduler (stepped every training step).
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
        scheduler: LRScheduler | None = None,
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
        self.scheduler = scheduler

        # Warmup
        self.warmup_steps = warmup_steps

        # Loss scaling
        self.loss_scale = loss_scale_init
        self._loss_scale_init = loss_scale_init
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
        """Set the warmup flag on all AXS-6 layers (unified + BF16)."""
        warmup_types = _get_warmup_types()
        for module in self.model.modules():
            if isinstance(module, warmup_types):
                module._warmup_active = active  # type: ignore[union-attr]

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
            ``overflow``, ``step``, ``warmup``, ``lr``.
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
                "lr": self._current_lr(),
            }

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm,
        )
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        self._grad_norms.append(grad_norm_val)

        # Optimiser step
        self.optimizer.step()

        # Scheduler step (per-batch)
        if self.scheduler is not None:
            self.scheduler.step()

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
            "lr": self._current_lr(),
        }

    def _current_lr(self) -> float:
        """Return the current learning rate from the first param group."""
        return self.optimizer.param_groups[0]["lr"]

    def _check_overflow(self) -> bool:
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.any(torch.isinf(param.grad)) or torch.any(torch.isnan(param.grad)):
                    return True
        return False

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save full training state to a checkpoint file.

        Saves: model state_dict, optimiser state_dict, scheduler state_dict,
        loss scale, step counters, overflow history, and CUDA RNG state.

        Args:
            path: File path for the checkpoint (``.pt`` extension recommended).
            extra: Optional dict of user data (e.g. epoch, best_loss).

        Returns:
            The resolved Path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self._total_steps,
            "loss_scale": self.loss_scale,
            "steps_since_scale_change": self._steps_since_scale_change,
            "overflow_count": self._overflow_count,
            "warmup_steps": self.warmup_steps,
        }

        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        if self.amax_ema is not None:
            state["amax_ema"] = self.amax_ema.state_dict()

        # Save RNG state for exact reproducibility
        state["rng"] = {
            "python": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }

        if extra is not None:
            state["extra"] = extra

        torch.save(state, path)
        logger.info("Saved checkpoint to %s (step %d)", path, self._total_steps)
        return path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """
        Restore full training state from a checkpoint file.

        Args:
            path: Checkpoint file path.
            map_location: Device mapping (e.g. ``'cuda:0'``).

        Returns:
            The ``extra`` dict that was saved, or ``{}`` if none.
        """
        path = Path(path)
        state = torch.load(path, map_location=map_location, weights_only=False)

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

        self._total_steps = state["total_steps"]
        self.loss_scale = state["loss_scale"]
        self._steps_since_scale_change = state.get("steps_since_scale_change", 0)
        self._overflow_count = state.get("overflow_count", 0)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)

        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

        if self.amax_ema is not None and "amax_ema" in state:
            self.amax_ema.load_state_dict(state["amax_ema"])

        # Restore RNG state
        rng = state.get("rng", {})
        if "python" in rng:
            torch.random.set_rng_state(rng["python"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])

        logger.info(
            "Loaded checkpoint from %s (step %d)",
            path, self._total_steps,
        )
        return state.get("extra", {})

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
            "lr": self._current_lr(),
        }
