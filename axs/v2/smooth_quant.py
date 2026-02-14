"""
SmoothQuant: Activation-Weight Quantization Balancing
=====================================================

From SmoothQuant (Xiao et al., 2023): Transformer activations have severe
per-channel outliers (some channels are 10-100× larger), making them hard
to quantize. Weights are relatively uniform across channels.

SmoothQuant applies a per-channel scaling transform:

    Y = (X · diag(s)⁻¹) · (diag(s) · W)

This "migrates" quantization difficulty from activations to weights:
  - Activations become smoother (less outlier variance)
  - Weights absorb the channel-wise scaling (easy since they're static)

The scaling factors s are computed from calibration data:

    s_j = max(|X_:,j|)^α / max(|W_:,j|)^(1-α)

where α ∈ [0, 1] controls the balance (typically 0.5).

For AXS-6, this directly addresses the biggest quality issue: the shared
exponent is set by abs_max, so reducing activation outliers means better
utilization of the quantization grid across the entire block.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SmoothQuant Scaling
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_smooth_scales(
    activation_stats: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    min_scale: float = 1e-5,
) -> torch.Tensor:
    """
    Compute per-channel SmoothQuant scaling factors.

    Args:
        activation_stats: Per-channel max absolute activation values.
            Shape: (in_features,) — typically computed from calibration data.
        weight: Weight matrix of shape (out_features, in_features).
        alpha: Migration strength. 0 = all on weights, 1 = all on activations.
        min_scale: Minimum scale to prevent division by zero.

    Returns:
        Scaling vector of shape (in_features,).
    """
    # Per-channel max of weight
    weight_max = weight.abs().amax(dim=0).clamp(min=min_scale)
    act_max = activation_stats.clamp(min=min_scale)

    # SmoothQuant formula: s = act_max^α / weight_max^(1-α)
    scales = act_max.pow(alpha) / weight_max.pow(1.0 - alpha)

    # Clamp for safety
    scales = scales.clamp(min=min_scale)

    return scales


@torch.no_grad()
def apply_smooth_scales_to_layer(
    layer: nn.Linear,
    scales: torch.Tensor,
) -> None:
    """
    Apply SmoothQuant scaling to a linear layer in-place.

    This modifies the weight matrix: W_new = W_old * diag(scales)
    The caller is responsible for dividing the input by scales.

    Args:
        layer: Linear layer to modify.
        scales: Per-channel scaling factors of shape (in_features,).
    """
    layer.weight.data.mul_(scales.unsqueeze(0))  # (out, in) * (1, in)
    # Note: bias is unchanged since it's added after matmul


class SmoothQuantCalibrator:
    """
    Collects activation statistics during calibration for SmoothQuant.

    Usage::

        calibrator = SmoothQuantCalibrator(model)
        with calibrator.calibrate():
            for batch in calibration_data:
                model(batch)
        calibrator.apply(alpha=0.5)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._act_stats: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def calibrate(self):
        """Context manager that registers hooks to collect activation stats."""
        return _CalibrationContext(self)

    def _register_hooks(self) -> None:
        """Register forward hooks on all Linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a forward hook that records activation statistics."""
        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            # input[0] is the activation tensor going into this layer
            act = input[0].detach()
            # Flatten batch dimensions, compute per-channel max
            flat = act.reshape(-1, act.shape[-1])
            channel_max = flat.abs().amax(dim=0)

            if name not in self._act_stats:
                self._act_stats[name] = channel_max
                self._counts[name] = 1
            else:
                # Running max (across calibration batches)
                self._act_stats[name] = torch.max(self._act_stats[name], channel_max)
                self._counts[name] += 1

        return hook_fn

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def apply(
        self,
        alpha: float = 0.5,
        min_scale: float = 1e-5,
    ) -> dict[str, torch.Tensor]:
        """
        Apply SmoothQuant scaling to all Linear layers using collected stats.

        This modifies the model's weights in-place and returns the scaling
        factors that must be applied to activations during inference/training.

        Args:
            alpha: Migration strength.
            min_scale: Minimum scale value.

        Returns:
            Dict mapping layer names to their scaling factors.
        """
        scaling_factors: dict[str, torch.Tensor] = {}

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name not in self._act_stats:
                continue

            act_stats = self._act_stats[name]
            scales = compute_smooth_scales(
                act_stats, module.weight, alpha=alpha, min_scale=min_scale
            )

            apply_smooth_scales_to_layer(module, scales)
            scaling_factors[name] = scales

        return scaling_factors

    def clear(self) -> None:
        """Clear collected statistics."""
        self._act_stats.clear()
        self._counts.clear()


class _CalibrationContext:
    """Context manager for SmoothQuant calibration."""

    def __init__(self, calibrator: SmoothQuantCalibrator) -> None:
        self.calibrator = calibrator

    def __enter__(self):
        self.calibrator._register_hooks()
        return self.calibrator

    def __exit__(self, *args):
        self.calibrator._remove_hooks()


# ---------------------------------------------------------------------------
# Inline SmoothQuant for Training (no calibration needed)
# ---------------------------------------------------------------------------

class SmoothQuantWrapper(nn.Module):
    """
    Wraps a Linear layer with online SmoothQuant scaling.

    Unlike the calibration-based approach, this computes and applies
    smooth scaling on-the-fly using an exponential moving average of
    activation statistics. This is suitable for use during training
    where calibration data isn't available upfront.

    The wrapper maintains:
      - EMA of per-channel activation max values
      - Current smooth scaling factors (updated every `update_interval` steps)
    """

    def __init__(
        self,
        linear: nn.Linear,
        alpha: float = 0.5,
        ema_decay: float = 0.99,
        update_interval: int = 100,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.update_interval = update_interval

        in_features = linear.in_features
        self.register_buffer(
            "act_ema", torch.ones(in_features)
        )
        self.register_buffer(
            "smooth_scales", torch.ones(in_features)
        )
        self._step_count = 0
        self._scales_applied = False

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        """Update activation EMA statistics."""
        flat = x.detach().reshape(-1, x.shape[-1])
        channel_max = flat.abs().amax(dim=0)
        self.act_ema.mul_(self.ema_decay).add_(channel_max, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def _recompute_scales(self) -> None:
        """Recompute smooth scaling factors from current statistics."""
        if self._scales_applied:
            # Undo previous scaling on weights
            self.linear.weight.data.div_(self.smooth_scales.unsqueeze(0))

        new_scales = compute_smooth_scales(
            self.act_ema,
            self.linear.weight.data,
            alpha=self.alpha,
        )

        # Apply new scaling to weights
        self.linear.weight.data.mul_(new_scales.unsqueeze(0))
        self.smooth_scales.copy_(new_scales)
        self._scales_applied = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_stats(x)
            self._step_count += 1
            if self._step_count % self.update_interval == 0:
                self._recompute_scales()

        # Apply inverse scaling to activations
        x_smooth = x / self.smooth_scales

        return self.linear(x_smooth)
