"""
AXS-6 Functional Operations
============================

Low-level functional implementations of AXS-6 quantized operations.
These are the building blocks used by the module-level layers.

All functions use the Straight-Through Estimator (STE) to allow gradient
flow through quantization operations during backpropagation.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from axs.core import (
    AXS6_EXPONENT_BIAS,
    AXS6_MAX_MAGNITUDE,
    DEFAULT_BLOCK_SIZE,
    VALID_BLOCK_SIZES,
    AXSTensor,
    dequantize,
    quantize,
)


# ---------------------------------------------------------------------------
# Straight-Through Estimator for quantization
# ---------------------------------------------------------------------------


class _QuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for AXS-6 quantization.

    Forward: quantize → dequantize (introduces quantization noise)
    Backward: pass gradients through unchanged (identity)

    This is the standard approach used in quantization-aware training (QAT)
    and is critical for making the non-differentiable quantization operation
    compatible with gradient-based optimization.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        block_size: int,
        rounding: str,
    ) -> torch.Tensor:
        axs = quantize(tensor, block_size=block_size, rounding=rounding)  # type: ignore[arg-type]
        return dequantize(axs)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        # STE: pass gradient through unchanged
        return grad_output, None, None


def fake_quantize(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
) -> torch.Tensor:
    """
    Fake-quantize a tensor: quantize to AXS-6 and immediately dequantize.

    This simulates the effect of quantization during training while maintaining
    gradient flow via the Straight-Through Estimator.

    Args:
        tensor: Input tensor.
        block_size: AXS-6 block size.
        rounding: Rounding mode.

    Returns:
        Tensor with quantization noise applied, same shape as input.
    """
    return _QuantizeSTE.apply(tensor, block_size, rounding)


# ---------------------------------------------------------------------------
# Quantized Linear Operation
# ---------------------------------------------------------------------------


class _AXSLinearFunction(torch.autograd.Function):
    """
    Custom autograd function for AXS-6 quantized linear operation.

    Forward: quantize weights and input, compute output in FP32
    Backward: quantize gradients with stochastic rounding
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
        # Fake-quantize weight (always)
        weight_q = fake_quantize(weight, block_size, "nearest")

        # Optionally fake-quantize input activations
        if quantize_input:
            input_q = fake_quantize(input, block_size, "nearest")
        else:
            input_q = input

        # Standard linear forward
        output = F.linear(input_q, weight_q, bias)

        # Save for backward
        ctx.save_for_backward(input_q, weight_q, bias)
        ctx.block_size = block_size  # type: ignore[attr-defined]
        ctx.quantize_grad = quantize_grad  # type: ignore[attr-defined]

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None, None]:
        input_q, weight_q, bias = ctx.saved_tensors  # type: ignore[attr-defined]
        block_size = ctx.block_size  # type: ignore[attr-defined]
        quantize_grad = ctx.quantize_grad  # type: ignore[attr-defined]

        # Optionally quantize gradient with stochastic rounding
        if quantize_grad:
            grad_output = fake_quantize(grad_output, block_size, "stochastic")

        # Standard linear backward
        grad_input = grad_output @ weight_q if ctx.needs_input_grad[0] else None  # type: ignore[index]
        grad_weight = grad_output.transpose(-2, -1) @ input_q if ctx.needs_input_grad[1] else None  # type: ignore[index]
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if (bias is not None and ctx.needs_input_grad[2]) else None  # type: ignore[index]

        return grad_input, grad_weight, grad_bias, None, None, None


def axs_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
    quantize_grad: bool = True,
) -> torch.Tensor:
    """
    AXS-6 quantized linear transformation.

    Equivalent to ``F.linear(input, weight, bias)`` but with AXS-6 quantization
    applied to weights (always), inputs (optionally), and gradients (optionally).

    Args:
        input: Input tensor of shape ``(*, in_features)``.
        weight: Weight matrix of shape ``(out_features, in_features)``.
        bias: Optional bias of shape ``(out_features,)``.
        block_size: AXS-6 block size.
        quantize_input: Whether to quantize input activations.
        quantize_grad: Whether to quantize gradients with stochastic rounding.

    Returns:
        Output tensor of shape ``(*, out_features)``.
    """
    return _AXSLinearFunction.apply(
        input, weight, bias, block_size, quantize_input, quantize_grad
    )


# ---------------------------------------------------------------------------
# Quantized Matrix Multiplication
# ---------------------------------------------------------------------------


def axs_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_inputs: bool = True,
) -> torch.Tensor:
    """
    AXS-6 quantized matrix multiplication.

    Both inputs are fake-quantized to AXS-6 before multiplication.
    Gradient flow uses STE.

    Args:
        a: Left matrix ``(*, M, K)``.
        b: Right matrix ``(*, K, N)``.
        block_size: AXS-6 block size.
        quantize_inputs: Whether to quantize inputs.

    Returns:
        Result tensor ``(*, M, N)``.
    """
    if quantize_inputs:
        a_q = fake_quantize(a, block_size)
        b_q = fake_quantize(b, block_size)
    else:
        a_q, b_q = a, b
    return torch.matmul(a_q, b_q)


# ---------------------------------------------------------------------------
# Quantized Conv2d Operation
# ---------------------------------------------------------------------------


class _AXSConv2dFunction(torch.autograd.Function):
    """Custom autograd for AXS-6 quantized 2D convolution."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        block_size: int,
    ) -> torch.Tensor:
        # Fake-quantize weight — reshape to 2D for block quantization
        w_shape = weight.shape
        weight_2d = weight.reshape(w_shape[0], -1)
        weight_q_2d = fake_quantize(weight_2d, block_size, "nearest")
        weight_q = weight_q_2d.reshape(w_shape)

        # Fake-quantize input activations (per-channel reshape for block quant)
        in_shape = input.shape
        input_2d = input.reshape(-1, in_shape[-1]) if input.ndim > 2 else input.reshape(-1, in_shape[1] * in_shape[2] * in_shape[3]) if input.ndim == 4 else input
        input_q = fake_quantize(input.reshape(-1, input.shape[-1]), block_size, "nearest").reshape(in_shape) if input.numel() > 0 else input

        output = F.conv2d(input_q, weight_q, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(input_q, weight_q, bias)
        ctx.stride = stride  # type: ignore[attr-defined]
        ctx.padding = padding  # type: ignore[attr-defined]
        ctx.dilation = dilation  # type: ignore[attr-defined]
        ctx.groups = groups  # type: ignore[attr-defined]
        ctx.block_size = block_size  # type: ignore[attr-defined]

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None, None, None, None, None,
    ]:
        input_q, weight_q, bias = ctx.saved_tensors  # type: ignore[attr-defined]
        block_size = ctx.block_size  # type: ignore[attr-defined]

        # Quantize gradient for communication compression (stochastic rounding)
        grad_output_q = fake_quantize(grad_output.reshape(-1, grad_output.shape[-1]), block_size, "stochastic").reshape(grad_output.shape) if grad_output.numel() > 0 else grad_output

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:  # type: ignore[index]
            grad_input = torch.nn.grad.conv2d_input(
                input_q.shape, weight_q, grad_output_q,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,  # type: ignore[attr-defined]
            )
        if ctx.needs_input_grad[1]:  # type: ignore[index]
            grad_weight = torch.nn.grad.conv2d_weight(
                input_q, weight_q.shape, grad_output_q,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,  # type: ignore[attr-defined]
            )
        if bias is not None and ctx.needs_input_grad[2]:  # type: ignore[index]
            grad_bias = grad_output_q.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def axs_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """
    AXS-6 quantized 2D convolution.

    Args:
        input: Input tensor ``(N, C_in, H, W)``.
        weight: Conv weight ``(C_out, C_in/groups, kH, kW)``.
        bias: Optional bias ``(C_out,)``.
        stride, padding, dilation, groups: Standard conv parameters.
        block_size: AXS-6 block size.

    Returns:
        Convolution output tensor.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    return _AXSConv2dFunction.apply(
        input, weight, bias, stride, padding, dilation, groups, block_size
    )
