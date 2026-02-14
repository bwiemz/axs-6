"""
AXS-6 V2 Functional Operations
================================

V2 functional ops integrating ALL improvements:
  - NormalFloat-5 quantization grid (information-optimal)
  - Percentile clipping (outlier-robust scaling)
  - Hadamard rotation (outlier spreading)
  - Precision annealing (gradual quantization introduction)
  - STE with gradient-aware scaling

These are drop-in replacements for axs.nn.functional operations.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from axs.core import DEFAULT_BLOCK_SIZE
from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
from axs.v2.hadamard import apply_hadamard_rotation, invert_hadamard_rotation


# ---------------------------------------------------------------------------
# V2 STE: Straight-Through Estimator with NF5 + all v2 improvements
# ---------------------------------------------------------------------------


class _QuantizeSTEv2(torch.autograd.Function):
    """
    V2 STE for AXS-6 quantization with NormalFloat grid.

    Forward: quantize_v2 → dequantize_v2 (NF5 grid, percentile clipping)
    Backward: pass gradients through with optional gradient scaling

    Improvements over v1:
      - NF5 grid places quantization levels optimally for normal distributions
      - Percentile clipping reduces outlier sensitivity
      - Optional gradient scaling based on quantization step size
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        block_size: int,
        rounding: str,
        use_nf5: bool,
        clip_percentile: float | None,
    ) -> torch.Tensor:
        axs = quantize_v2(
            tensor,
            block_size=block_size,
            rounding=rounding,  # type: ignore[arg-type]
            use_nf5=use_nf5,
            clip_percentile=clip_percentile if clip_percentile and clip_percentile > 0 else None,
        )
        return dequantize_v2(axs, use_nf5=use_nf5)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        return grad_output, None, None, None, None


def fake_quantize_v2(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    use_nf5: bool = True,
    clip_percentile: float | None = 99.9,
) -> torch.Tensor:
    """
    V2 fake-quantize: quantize to AXS-6 with NF5 grid and immediately dequantize.

    Args:
        tensor: Input tensor.
        block_size: AXS-6 block size.
        rounding: Rounding mode.
        use_nf5: Use NormalFloat-5 grid (True) or uniform grid (False).
        clip_percentile: Percentile for scale computation (None = abs_max).

    Returns:
        Tensor with V2 quantization noise applied.
    """
    return _QuantizeSTEv2.apply(tensor, block_size, rounding, use_nf5, clip_percentile)


# ---------------------------------------------------------------------------
# V2 Quantized Linear with Hadamard rotation
# ---------------------------------------------------------------------------


class _AXSLinearV2Function(torch.autograd.Function):
    """
    V2 linear with NF5 grid + Hadamard rotation on weights.

    Hadamard rotation spreads outliers across all elements in a block,
    making quantization more uniform and reducing worst-case error.
    The rotation is applied before quantization and inverted after
    dequantization (in the backward pass, via STE).
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
        use_nf5: bool,
        clip_percentile: float | None,
        use_hadamard: bool,
    ) -> torch.Tensor:
        # Fake-quantize weight (always) — with optional Hadamard rotation
        if use_hadamard and weight.shape[-1] >= block_size and (weight.shape[-1] % block_size == 0):
            # Reshape weight to blocks, rotate, quantize, unrotate
            w_shape = weight.shape
            blocked = weight.reshape(*w_shape[:-1], -1, block_size)
            blocked_rotated = apply_hadamard_rotation(blocked)
            blocked_q = fake_quantize_v2(
                blocked_rotated.reshape(w_shape), block_size, "nearest", use_nf5, clip_percentile
            )
            blocked_q_reshaped = blocked_q.reshape(*w_shape[:-1], -1, block_size)
            blocked_unrotated = invert_hadamard_rotation(blocked_q_reshaped)
            weight_q = blocked_unrotated.reshape(w_shape)
        else:
            weight_q = fake_quantize_v2(
                weight, block_size, "nearest", use_nf5, clip_percentile
            )

        # Optionally fake-quantize input activations
        if quantize_input:
            input_q = fake_quantize_v2(
                input, block_size, "nearest", use_nf5, clip_percentile
            )
        else:
            input_q = input

        # Standard linear forward
        output = F.linear(input_q, weight_q, bias)

        # Save for backward
        ctx.save_for_backward(input_q, weight_q, bias)
        ctx.block_size = block_size  # type: ignore[attr-defined]
        ctx.quantize_grad = quantize_grad  # type: ignore[attr-defined]
        ctx.use_nf5 = use_nf5  # type: ignore[attr-defined]
        ctx.clip_percentile = clip_percentile  # type: ignore[attr-defined]

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
        None, None, None, None, None, None,
    ]:
        input_q, weight_q, bias = ctx.saved_tensors  # type: ignore[attr-defined]
        block_size = ctx.block_size  # type: ignore[attr-defined]
        quantize_grad = ctx.quantize_grad  # type: ignore[attr-defined]
        use_nf5 = ctx.use_nf5  # type: ignore[attr-defined]
        clip_percentile = ctx.clip_percentile  # type: ignore[attr-defined]

        # Optionally quantize gradient (stochastic rounding for unbiased compression)
        if quantize_grad:
            grad_output = fake_quantize_v2(
                grad_output, block_size, "stochastic", use_nf5, clip_percentile
            )

        # Standard linear backward
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

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


def axs_linear_v2(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
    quantize_grad: bool = True,
    use_nf5: bool = True,
    clip_percentile: float | None = 99.9,
    use_hadamard: bool = False,
) -> torch.Tensor:
    """
    V2 AXS-6 quantized linear transformation.

    Improvements over v1:
      - NormalFloat-5 grid for optimal quantization levels
      - Percentile clipping for outlier-robust scaling
      - Optional Hadamard rotation for outlier spreading

    Args:
        input: Input tensor ``(*, in_features)``.
        weight: Weight matrix ``(out_features, in_features)``.
        bias: Optional bias ``(out_features,)``.
        block_size: AXS-6 block size.
        quantize_input: Whether to quantize input activations.
        quantize_grad: Whether to quantize gradients.
        use_nf5: Use NormalFloat-5 grid.
        clip_percentile: Percentile for scale computation.
        use_hadamard: Apply Hadamard rotation before weight quantization.

    Returns:
        Output tensor ``(*, out_features)``.
    """
    return _AXSLinearV2Function.apply(
        input, weight, bias, block_size,
        quantize_input, quantize_grad,
        use_nf5, clip_percentile, use_hadamard,
    )


# ---------------------------------------------------------------------------
# V2 Quantized MatMul
# ---------------------------------------------------------------------------


def axs_matmul_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_inputs: bool = True,
    use_nf5: bool = True,
    clip_percentile: float | None = 99.9,
) -> torch.Tensor:
    """
    V2 AXS-6 quantized matrix multiplication.

    Args:
        a: Left matrix ``(*, M, K)``.
        b: Right matrix ``(*, K, N)``.
        block_size: AXS-6 block size.
        quantize_inputs: Whether to quantize inputs.
        use_nf5: Use NormalFloat-5 grid.
        clip_percentile: Percentile for scale computation.

    Returns:
        Result tensor ``(*, M, N)``.
    """
    if quantize_inputs:
        a_q = fake_quantize_v2(a, block_size, "nearest", use_nf5, clip_percentile)
        b_q = fake_quantize_v2(b, block_size, "nearest", use_nf5, clip_percentile)
    else:
        a_q, b_q = a, b
    return torch.matmul(a_q, b_q)
