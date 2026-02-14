"""
AXS-Aware Optimizer Wrapper
============================

Wraps standard PyTorch optimizers to integrate AXS-6 quantization into the
training loop. Maintains FP32 master weights while applying quantized updates.

The key insight: during mixed-precision training, we need to:
  1. Keep FP32 master weights (for optimizer state precision)
  2. Quantize weights to AXS-6 for the forward pass
  3. Accumulate gradients and apply optimizer steps in FP32
  4. Optionally quantize gradients before accumulation (stochastic rounding)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from axs.core import DEFAULT_BLOCK_SIZE
from axs.nn.functional import fake_quantize
from axs.quantize import (
    ErrorFeedbackState,
    quantize_stochastic,
    quantize_with_error_feedback,
)
from axs.core import dequantize


class AXSAdamW(Optimizer):
    """
    AdamW optimizer with integrated AXS-6 quantization support.

    This optimizer maintains FP32 master weights and applies weight decay
    and momentum updates in full precision. Optionally, gradient quantization
    with error feedback can be enabled for communication compression in
    distributed training.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate. Default: 1e-3.
        betas: Coefficients for computing running averages. Default: (0.9, 0.999).
        eps: Term added for numerical stability. Default: 1e-8.
        weight_decay: Weight decay (L2 penalty). Default: 0.01.
        block_size: AXS-6 block size for gradient quantization. Default: 32.
        quantize_gradients: Whether to quantize gradients. Default: False.
        use_error_feedback: Whether to use error feedback for gradient
            quantization. Default: True.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_size: int = DEFAULT_BLOCK_SIZE,
        quantize_gradients: bool = False,
        use_error_feedback: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            block_size=block_size,
        )
        super().__init__(params, defaults)
        self.quantize_gradients = quantize_gradients
        self.use_error_feedback = use_error_feedback
        self._ef_state = ErrorFeedbackState() if use_error_feedback else None

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:  # type: ignore[override]
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            block_size = group["block_size"]

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad

                # Optionally quantize gradient
                if self.quantize_gradients:
                    param_name = f"group_{id(group)}_param_{i}"
                    if self.use_error_feedback and self._ef_state is not None:
                        q_grad = quantize_with_error_feedback(
                            grad, block_size=block_size,
                            rounding="nearest",
                            param_name=param_name,
                            state=self._ef_state,
                        )
                        grad = dequantize(q_grad)
                    else:
                        q_grad = quantize_stochastic(grad, block_size=block_size)
                        grad = dequantize(q_grad)

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Decoupled weight decay
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Adam update in FP32
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AXSTrainingWrapper:
    """
    High-level training wrapper that manages the full AXS-6 mixed-precision
    training pipeline.

    This wrapper handles:
      - Dynamic loss scaling for AXS-6 training
      - Automatic gradient clipping before quantization
      - Training statistics tracking

    Usage::

        model = convert_to_axs(model)
        optimizer = AXSAdamW(model.parameters(), lr=1e-3)
        trainer = AXSTrainingWrapper(model, optimizer)

        for batch in dataloader:
            loss = trainer.training_step(batch, loss_fn)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_scale_init: float = 2.0**16,
        loss_scale_growth_interval: int = 2000,
        loss_scale_backoff: float = 0.5,
        loss_scale_growth: float = 2.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_scale = loss_scale_init
        self.growth_interval = loss_scale_growth_interval
        self.backoff = loss_scale_backoff
        self.growth = loss_scale_growth
        self.max_grad_norm = max_grad_norm

        self._steps_since_scale_change = 0
        self._total_steps = 0
        self._overflow_count = 0

    def training_step(
        self,
        inputs: Any,
        loss_fn: Any,
    ) -> dict[str, float]:
        """
        Execute one training step with AXS-6 mixed precision.

        Args:
            inputs: Input data (passed to model and loss function).
            loss_fn: Callable that takes (model_output, inputs) and returns loss.

        Returns:
            Dict with ``loss``, ``loss_scale``, ``grad_norm`` keys.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass (quantized via AXS modules)
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
        has_overflow = False
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.any(torch.isinf(param.grad)) or torch.any(torch.isnan(param.grad)):
                    has_overflow = True
                    break

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
            }

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

        # Optimizer step
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
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "overflow": False,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Training statistics."""
        return {
            "total_steps": self._total_steps,
            "overflow_count": self._overflow_count,
            "current_loss_scale": self.loss_scale,
        }
