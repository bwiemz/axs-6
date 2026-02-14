"""
AXS-6 Unified Drop-in Module Replacements
===========================================

Production-ready PyTorch modules that achieve NF5-quality quantisation at
V1-class speed using the fused warp table.

Modules:
  - :class:`AXSLinearUnified` — ``nn.Linear`` replacement
  - :class:`AXSLayerNormUnified` — ``nn.LayerNorm`` (no output quant, same as V2)
  - :class:`AXSEmbeddingUnified` — ``nn.Embedding`` with lazy quant
  - :class:`AXSMultiheadAttentionUnified` — multi-head attention

Conversion:
  - :func:`convert_to_axs_unified` — swap standard / V1 / V2 layers in-place
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from axs.core import DEFAULT_BLOCK_SIZE
from axs.unified.functional_unified import (
    axs_linear_unified,
    fake_quantize_unified,
)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class AXSLinearUnified(nn.Module):
    """
    Drop-in ``nn.Linear`` replacement with fused NF5 quantisation.

    Features:
      - Fused NF5 warp table (V2 quality, faster than V1)
      - Power-of-2 block scaling
      - Optional stochastic-rounding gradient quantisation
      - Zero-overhead skip-first-N warmup integration

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
            return F.linear(input, self.weight, self.bias)
        return axs_linear_unified(
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
            f"quant_input={self.quantize_input}, quant_grad={self.quantize_grad}"
        )


# ---------------------------------------------------------------------------
# LayerNorm — no output quantisation (same design as V2)
# ---------------------------------------------------------------------------

class AXSLayerNormUnified(nn.Module):
    """
    LayerNorm without output quantisation.

    The downstream linear layer quantises its input anyway, so
    double-quantising is wasteful and harms training stability.
    """

    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps,
        )


# ---------------------------------------------------------------------------
# Embedding — lazy quantisation (only quantise accessed rows)
# ---------------------------------------------------------------------------

class AXSEmbeddingUnified(nn.Module):
    """
    Embedding with lazy quantisation.

    Looks up in FP32 first, then applies fused NF5 fake-quantise to
    the selected rows only.  For ``vocab_size=50 000`` this is ~50 000×
    less work than quantising the full table every forward pass.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

        self._warmup_active: bool = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raw = F.embedding(input, self.weight, self.padding_idx)
        if self._warmup_active:
            return raw
        return fake_quantize_unified(raw, self.block_size, "nearest")


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class AXSMultiheadAttentionUnified(nn.Module):
    """
    Multi-head attention with fused NF5 quantisation on all projections.

    Design choices (matching V2):
      - Q/K/V/out projections use AXSLinearUnified
      - Attention scores computed in FP32 (no pre-softmax quantisation)
      - Post-softmax attention weights are fake-quantised (stochastic rounding)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.dropout = dropout

        self.q_proj = AXSLinearUnified(embed_dim, embed_dim, bias=bias, block_size=block_size)
        self.k_proj = AXSLinearUnified(embed_dim, embed_dim, bias=bias, block_size=block_size)
        self.v_proj = AXSLinearUnified(embed_dim, embed_dim, bias=bias, block_size=block_size)
        self.out_proj = AXSLinearUnified(embed_dim, embed_dim, bias=bias, block_size=block_size)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        q = self.q_proj(query).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        k = self.k_proj(key).reshape(
            batch_size, key.shape[1], self.num_heads, self.head_dim,
        ).transpose(1, 2)
        v = self.v_proj(value).reshape(
            batch_size, value.shape[1], self.num_heads, self.head_dim,
        ).transpose(1, 2)

        # Attention in FP32 — do NOT quantise before softmax
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Quantise post-softmax weights (stochastic rounding)
        attn_weights = fake_quantize_unified(
            attn_weights, self.block_size, "stochastic",
        )

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim,
        )

        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# Model Conversion
# ---------------------------------------------------------------------------

def convert_to_axs_unified(
    model: nn.Module,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
    quantize_grad: bool = True,
    skip_layers: set[str] | None = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert a standard PyTorch model to use unified AXS-6 quantised layers.

    Replaces ``nn.Linear``, ``nn.LayerNorm``, and ``nn.Embedding`` (and
    their V1/V2 counterparts) with unified modules.  Preserves weights.

    Args:
        model: Model to convert.
        block_size: AXS-6 block size.
        quantize_input: Quantise activations in linear layers.
        quantize_grad: Quantise gradients in linear layers.
        skip_layers: Set of layer names to leave unconverted.
        inplace: Modify model in-place (default: deep copy).

    Returns:
        Model with unified AXS-6 quantised layers.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    if skip_layers is None:
        skip_layers = set()

    # Import V1 and V2 module types for isinstance checks
    from axs.nn.modules import AXSLinear, AXSLayerNorm, AXSEmbedding
    from axs.v2.modules_v2 import AXSLinearV2, AXSLayerNormV2, AXSEmbeddingV2

    def _convert(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if full_name in skip_layers:
                continue

            if isinstance(child, (nn.Linear, AXSLinear, AXSLinearV2)):
                in_f = child.in_features
                out_f = child.out_features
                has_bias = child.bias is not None
                new_layer = AXSLinearUnified(
                    in_f, out_f, bias=has_bias,
                    block_size=block_size,
                    quantize_input=quantize_input,
                    quantize_grad=quantize_grad,
                )
                new_layer.weight.data.copy_(child.weight.data)
                if has_bias and new_layer.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)

            elif isinstance(child, (nn.LayerNorm, AXSLayerNorm, AXSLayerNormV2)):
                new_layer = AXSLayerNormUnified(
                    list(child.normalized_shape),
                    eps=child.eps,
                )
                new_layer.weight.data.copy_(child.weight.data)
                new_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)

            elif isinstance(child, (nn.Embedding, AXSEmbedding, AXSEmbeddingV2)):
                new_layer = AXSEmbeddingUnified(
                    child.num_embeddings,
                    child.embedding_dim,
                    padding_idx=child.padding_idx,
                    block_size=block_size,
                )
                new_layer.weight.data.copy_(child.weight.data)
                setattr(module, name, new_layer)

            else:
                _convert(child, full_name)

    _convert(model)
    return model
