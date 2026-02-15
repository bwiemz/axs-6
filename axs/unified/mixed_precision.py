"""
AXS-6 Mixed-Precision (BF16) Training
=======================================

Provides BF16 mixed-precision training for AXS-6, combining:

  - **BF16 tensor-core matmul** for ~2× compute speedup
  - **AXS-6 NF5 fake-quantize** for 6-bit weight/activation quantisation
  - **FP32 master weights** for numerically stable optimiser updates
  - **Activation recomputation** — weight_q is never saved; recomputed in
    backward using the fast Triton FQ kernel (saves ~50% activation memory
    per layer)

Memory savings vs FP32 AXS-6 baseline:

  - **4× less activation memory** — BF16 activations (2×) combined with
    weight recomputation (2×)
  - **Same parameter memory** — master weights remain FP32

The key insight is that AXS-6's NF5 quantisation already introduces ~6-bit
noise, which dominates BF16's inherent ~7.5-bit mantissa precision.  BF16
therefore adds *zero additional quantisation error* while unlocking tensor
cores.

Usage::

    from axs.unified.mixed_precision import (
        AXSLinearMixedPrecision,
        convert_to_axs_mixed_precision,
    )

    model = convert_to_axs_mixed_precision(model)
    model.cuda()

    # Training loop — no manual casting needed
    for batch in dataloader:
        out = model(batch.cuda())
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from axs.core import DEFAULT_BLOCK_SIZE
from axs.unified.backend import (
    accelerated_fake_quantize as _accel_fq,
)
from axs.unified.functional_unified import fake_quantize_unified


# ---------------------------------------------------------------------------
# BF16 autograd function — the core building block
# ---------------------------------------------------------------------------


class _AXSLinearBF16Function(torch.autograd.Function):
    """
    AXS-6 quantised linear with BF16 mixed-precision.

    Forward (BF16 compute path):
      1. Cast weight & input to BF16
      2. Fake-quantise weight (always) and input (optionally) in BF16
         (FQ kernel computes in FP32 internally, returns BF16)
      3. BF16 matmul via F.linear (uses BF16 tensor cores)

    Backward:
      - Recompute weight_q from saved leaf weight (not stored — saves memory)
      - Compute gradients in BF16 for speed, accumulate in FP32 for weight
        gradient (ensures stable optimiser updates)

    Memory layout:
      - Saved: input_q (BF16), weight (leaf ref), bias (leaf ref or None)
      - NOT saved: weight_q (recomputed from weight in backward)
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
        # Cast to BF16 for compute
        input_bf = input.to(torch.bfloat16)
        weight_bf = weight.to(torch.bfloat16)

        # FQ in BF16 (kernel computes FP32 internally, returns BF16)
        weight_q = _accel_fq(weight_bf, block_size, "nearest")

        if quantize_input:
            input_q = _accel_fq(input_bf, block_size, "nearest")
        else:
            input_q = input_bf

        # BF16 matmul — leverages tensor cores
        output = F.linear(input_q, weight_q, bias.to(torch.bfloat16) if bias is not None else None)

        # Save leaf weight (reference only) + BF16 input_q
        ctx.save_for_backward(input_q, weight, bias)
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
        input_q, weight, bias = ctx.saved_tensors  # type: ignore[attr-defined]
        block_size: int = ctx.block_size  # type: ignore[attr-defined]
        quantize_grad: bool = ctx.quantize_grad  # type: ignore[attr-defined]

        # Recompute weight_q in BF16 (essentially free, saves activation memory)
        weight_q = _accel_fq(weight.to(torch.bfloat16), block_size, "nearest")

        # Ensure grad_output is BF16 for tensor-core backward matmuls
        grad_bf = grad_output.to(torch.bfloat16)

        if quantize_grad:
            grad_bf = _accel_fq(grad_bf, block_size, "stochastic")

        # grad_input: BF16 matmul (same dtype as forward)
        grad_input = grad_bf @ weight_q if ctx.needs_input_grad[0] else None  # type: ignore[index]

        # grad_weight: accumulate in FP32 for stable optimiser updates
        if ctx.needs_input_grad[1]:  # type: ignore[index]
            grad_2d = grad_bf.reshape(-1, grad_bf.shape[-1])
            inp_2d = input_q.reshape(-1, input_q.shape[-1])
            grad_weight = (grad_2d.float().T @ inp_2d.float())
        else:
            grad_weight = None

        grad_bias = (
            grad_output.float().sum(dim=tuple(range(grad_output.ndim - 1)))
            if (bias is not None and ctx.needs_input_grad[2])  # type: ignore[index]
            else None
        )

        return grad_input, grad_weight, grad_bias, None, None, None


def axs_linear_mixed_precision(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
    quantize_grad: bool = True,
) -> torch.Tensor:
    """
    AXS-6 quantised linear with BF16 mixed-precision.

    Master weights stay FP32; forward compute is BF16; weight gradient
    is accumulated in FP32 for numerically stable optimiser updates.

    Args:
        input: Input tensor ``(*, in_features)``.
        weight: FP32 weight matrix ``(out_features, in_features)``.
        bias: Optional FP32 bias ``(out_features,)``.
        block_size: AXS-6 block size.
        quantize_input: Quantise input activations.
        quantize_grad: Quantise gradients (stochastic rounding).

    Returns:
        BF16 output tensor ``(*, out_features)``.
    """
    return _AXSLinearBF16Function.apply(
        input, weight, bias, block_size, quantize_input, quantize_grad,
    )


# ---------------------------------------------------------------------------
# Drop-in module
# ---------------------------------------------------------------------------


class AXSLinearMixedPrecision(nn.Module):
    """
    Drop-in ``nn.Linear`` replacement with AXS-6 NF5 + BF16 tensor cores.

    Keeps master weights in FP32 for stable optimiser updates.  Forward
    and backward compute in BF16 via tensor cores, with NF5 fake-quantise
    applied on-the-fly.

    Activation memory per layer: ``input_q`` (BF16) — no ``weight_q`` saved.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Include bias term.
        block_size: AXS-6 block size (8, 16, or 32).
        quantize_input: Quantise input activations.
        quantize_grad: Quantise gradients with stochastic rounding.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = DEFAULT_BLOCK_SIZE,
        quantize_input: bool = True,
        quantize_grad: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.quantize_input = quantize_input
        self.quantize_grad = quantize_grad

        # Master weights always FP32
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Warmup flag — when True, bypasses quantisation entirely
        self._warmup_active: bool = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._warmup_active:
            return F.linear(
                input.to(torch.bfloat16),
                self.weight.to(torch.bfloat16),
                self.bias.to(torch.bfloat16) if self.bias is not None else None,
            )
        return axs_linear_mixed_precision(
            input,
            self.weight,
            self.bias,
            self.block_size,
            self.quantize_input,
            self.quantize_grad,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}, "
            f"quant_input={self.quantize_input}, quant_grad={self.quantize_grad}, "
            f"mixed_precision=bf16"
        )


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------


def convert_to_axs_mixed_precision(
    model: nn.Module,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
    quantize_grad: bool = True,
    skip_layers: set[str] | None = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert a model to use AXS-6 BF16 mixed-precision linear layers.

    Replaces ``nn.Linear`` (and AXS-6 V1/V2/Unified variants) with
    :class:`AXSLinearMixedPrecision` modules.  ``nn.LayerNorm`` and
    ``nn.Embedding`` are handled by the standard unified conversion.

    Master weights remain FP32.  Forward/backward compute in BF16.

    Args:
        model: Model to convert.
        block_size: AXS-6 block size.
        quantize_input: Quantise activations in linear layers.
        quantize_grad: Quantise gradients in linear layers.
        skip_layers: Set of layer names to leave unconverted.
        inplace: Modify model in-place (default: deep copy).

    Returns:
        Model with AXS-6 BF16 mixed-precision linear layers.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    if skip_layers is None:
        skip_layers = set()

    # Import module types for isinstance checks
    from axs.nn.modules import AXSLinear
    from axs.v2.modules_v2 import AXSLinearV2
    from axs.unified.modules_unified import (
        AXSLinearUnified,
        AXSLayerNormUnified,
        AXSEmbeddingUnified,
    )

    linear_types = (nn.Linear, AXSLinear, AXSLinearV2, AXSLinearUnified)

    def _convert(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if full_name in skip_layers:
                continue

            if isinstance(child, linear_types):
                in_f = child.in_features
                out_f = child.out_features
                has_bias = child.bias is not None
                device = child.weight.device
                new_layer = AXSLinearMixedPrecision(
                    in_f, out_f, bias=has_bias,
                    block_size=block_size,
                    quantize_input=quantize_input,
                    quantize_grad=quantize_grad,
                )
                new_layer.weight.data.copy_(child.weight.data.float())
                if has_bias and new_layer.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data.float())
                setattr(module, name, new_layer.to(device))
            else:
                _convert(child, full_name)

    _convert(model)
    return model


# ---------------------------------------------------------------------------
# Memory estimation utility
# ---------------------------------------------------------------------------


def estimate_memory_savings(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 1,
) -> dict[str, float]:
    """
    Estimate activation memory savings from mixed-precision + recomputation.

    Compares:
      - **FP32 baseline**: saves ``input_q`` (FP32) + ``weight_q`` (FP32)
      - **BF16 + recomp**: saves ``input_q`` (BF16) only

    Args:
        model: Model with AXS-6 layers.
        batch_size: Batch size for activation estimation.
        seq_len: Sequence length (for transformer-style models).

    Returns:
        Dict with ``fp32_mb``, ``bf16_recomp_mb``, ``savings_ratio``.
    """
    fp32_bytes = 0
    bf16_recomp_bytes = 0

    for module in model.modules():
        if isinstance(module, (AXSLinearMixedPrecision,)):
            in_f = module.in_features
            out_f = module.out_features
            act_elements = batch_size * seq_len * in_f
            weight_elements = out_f * in_f

            # FP32 baseline: input_q (FP32) + weight_q (FP32)
            fp32_bytes += (act_elements + weight_elements) * 4

            # BF16 + recomp: input_q (BF16) only
            bf16_recomp_bytes += act_elements * 2

    fp32_mb = fp32_bytes / (1024 * 1024)
    bf16_mb = bf16_recomp_bytes / (1024 * 1024)
    ratio = fp32_mb / bf16_mb if bf16_mb > 0 else float("inf")

    return {
        "fp32_mb": round(fp32_mb, 2),
        "bf16_recomp_mb": round(bf16_mb, 2),
        "savings_ratio": round(ratio, 1),
    }
