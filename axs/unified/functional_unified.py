"""
AXS-6 Unified Functional Operations
=====================================

Autograd-aware operations using the fused NF5 warp table.  These are the
building blocks for the unified drop-in modules.

Key design choices:
  - Straight-Through Estimator (STE) passes gradients through unchanged —
    the same strategy used by V1 and V2.
  - The forward path calls :func:`fused_fake_quantize` which never
    materialises an AXSTensor, achieving V1-class latency.
  - Hadamard rotation is intentionally **not** included here.  Profiling
    showed it adds ~2.5× overhead and is not justified during training.
    Users who want it for post-training can use the V2 Hadamard utilities.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from axs.core import DEFAULT_BLOCK_SIZE
from axs.unified.quantize_unified import fused_fake_quantize


# ---------------------------------------------------------------------------
# STE: Straight-Through Estimator with Fused NF5
# ---------------------------------------------------------------------------

class _FusedQuantizeSTE(torch.autograd.Function):
    """
    Fused NF5 fake-quantize with straight-through gradient.

    Forward: ``fused_fake_quantize(tensor)``
    Backward: identity (gradients pass through unchanged)
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        block_size: int,
        rounding: str,
    ) -> torch.Tensor:
        return fused_fake_quantize(tensor, block_size=block_size, rounding=rounding)  # type: ignore[arg-type]

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        return grad_output, None, None


def fake_quantize_unified(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
) -> torch.Tensor:
    """
    Unified fake-quantize with autograd support.

    Uses the fused NF5 warp table for the forward pass and straight-through
    estimation for the backward pass.

    Args:
        tensor: Input tensor.
        block_size: AXS-6 block size (8, 16, or 32).
        rounding: ``"nearest"`` or ``"stochastic"`` (dithered).

    Returns:
        Tensor with NF5 quantisation noise applied.
    """
    return _FusedQuantizeSTE.apply(tensor, block_size, rounding)


# ---------------------------------------------------------------------------
# Quantised Linear
# ---------------------------------------------------------------------------

class _AXSLinearUnifiedFunction(torch.autograd.Function):
    """
    Unified AXS-6 quantised linear with fused NF5.

    Forward:
      - Fake-quantise weights (always, nearest)
      - Optionally fake-quantise input activations (nearest)
      - Standard ``F.linear``

    Backward:
      - Optionally fake-quantise ``grad_output`` (stochastic)
      - Standard linear backward through quantised inputs/weights
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        block_size: int,
        quantize_input: bool,
        quantize_grad: bool,
    ) -> torch.Tensor:
        weight_q = fused_fake_quantize(weight, block_size, "nearest")

        if quantize_input:
            input_q = fused_fake_quantize(input, block_size, "nearest")
        else:
            input_q = input

        output = F.linear(input_q, weight_q, bias)

        ctx.save_for_backward(input_q, weight_q, bias)
        ctx.block_size = block_size  # type: ignore[attr-defined]
        ctx.quantize_grad = quantize_grad  # type: ignore[attr-defined]

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
        None, None, None,
    ]:
        input_q, weight_q, bias = ctx.saved_tensors  # type: ignore[attr-defined]
        block_size: int = ctx.block_size  # type: ignore[attr-defined]
        quantize_grad: bool = ctx.quantize_grad  # type: ignore[attr-defined]

        if quantize_grad:
            grad_output = fused_fake_quantize(grad_output, block_size, "stochastic")

        grad_input = grad_output @ weight_q if ctx.needs_input_grad[0] else None  # type: ignore[index]
        grad_weight = (
            grad_output.reshape(-1, grad_output.shape[-1]).T
            @ input_q.reshape(-1, input_q.shape[-1])
            if ctx.needs_input_grad[1]  # type: ignore[index]
            else None
        )
        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
            if (bias is not None and ctx.needs_input_grad[2])  # type: ignore[index]
            else None
        )

        return grad_input, grad_weight, grad_bias, None, None, None


def axs_linear_unified(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
    quantize_grad: bool = True,
) -> torch.Tensor:
    """
    Unified AXS-6 quantised linear function.

    Args:
        input: Input tensor ``(*, in_features)``.
        weight: Weight matrix ``(out_features, in_features)``.
        bias: Optional bias ``(out_features,)``.
        block_size: AXS-6 block size.
        quantize_input: Whether to quantise input activations.
        quantize_grad: Whether to quantise gradients (stochastic rounding).

    Returns:
        Output tensor ``(*, out_features)``.
    """
    return _AXSLinearUnifiedFunction.apply(
        input, weight, bias, block_size, quantize_input, quantize_grad,
    )


# ---------------------------------------------------------------------------
# Quantised MatMul
# ---------------------------------------------------------------------------

def axs_matmul_unified(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_inputs: bool = True,
) -> torch.Tensor:
    """
    Unified AXS-6 quantised matrix multiplication.

    Args:
        a: Left matrix ``(*, M, K)``.
        b: Right matrix ``(*, K, N)``.
        block_size: AXS-6 block size.
        quantize_inputs: Whether to quantise inputs before matmul.

    Returns:
        Result tensor ``(*, M, N)``.
    """
    if quantize_inputs:
        a_q = fused_fake_quantize(a, block_size, "nearest")
        b_q = fused_fake_quantize(b, block_size, "nearest")
    else:
        a_q, b_q = a, b
    return torch.matmul(a_q, b_q)
