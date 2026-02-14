"""
AXS-6 V2 Drop-in Module Replacements
======================================

V2 modules integrating ALL optimization techniques into clean, drop-in
replacements for standard PyTorch layers.

Improvements over v1 modules:
  - NormalFloat-5 grid (information-optimal quantization levels)
  - Percentile clipping (outlier-robust block scaling)
  - Hadamard rotation (outlier spreading for uniform quantization error)
  - SmoothQuant integration (activation-weight balancing)
  - Precision annealing support (gradual quantization introduction)
  - Lazy embedding quantization (only quantize accessed rows)

Usage::

    model = convert_to_axs_v2(base_model)
    # or
    layer = AXSLinearV2(768, 256)
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from axs.core import DEFAULT_BLOCK_SIZE
from axs.v2.functional_v2 import axs_linear_v2, fake_quantize_v2
from axs.v2.annealing import PrecisionAnnealingSchedule, annealed_fake_quantize


# ---------------------------------------------------------------------------
# V2 Linear
# ---------------------------------------------------------------------------


class AXSLinearV2(nn.Module):
    """
    V2 drop-in replacement for ``nn.Linear`` with all optimizations.

    Features:
      - NF5 quantization grid
      - Percentile clipping for scaling
      - Optional Hadamard rotation on weights
      - Precision annealing support
      - SmoothQuant-compatible (external wrapper applies channel scaling)

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias.
        block_size: AXS-6 block size.
        quantize_input: Quantize input activations.
        quantize_grad: Quantize gradients (stochastic rounding).
        use_nf5: Use NormalFloat-5 grid.
        clip_percentile: Percentile for block scale computation.
        use_hadamard: Apply Hadamard rotation before weight quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = DEFAULT_BLOCK_SIZE,
        quantize_input: bool = True,
        quantize_grad: bool = True,
        use_nf5: bool = True,
        clip_percentile: float | None = 99.9,
        use_hadamard: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.quantize_input = quantize_input
        self.quantize_grad = quantize_grad
        self.use_nf5 = use_nf5
        self.clip_percentile = clip_percentile
        self.use_hadamard = use_hadamard

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return axs_linear_v2(
            input,
            self.weight,
            self.bias,
            self.block_size,
            self.quantize_input,
            self.quantize_grad,
            self.use_nf5,
            self.clip_percentile,
            self.use_hadamard,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}, "
            f"nf5={self.use_nf5}, hadamard={self.use_hadamard}, "
            f"clip={self.clip_percentile}"
        )


# ---------------------------------------------------------------------------
# V2 LayerNorm — NO output quantization (stability improvement)
# ---------------------------------------------------------------------------


class AXSLayerNormV2(nn.Module):
    """
    LayerNorm without output quantization.

    V1 quantized LayerNorm output, which hurts training stability.
    V2 runs LayerNorm entirely in FP32 — the downstream linear layer
    will quantize its input anyway, so double-quantizing is wasteful
    and harmful.
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
        # Pure FP32 LayerNorm — no quantization on output
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


# ---------------------------------------------------------------------------
# V2 Embedding — lazy quantization
# ---------------------------------------------------------------------------


class AXSEmbeddingV2(nn.Module):
    """
    Embedding with lazy quantization — only quantizes accessed rows.

    V1 quantized the entire embedding table every forward pass.
    V2 does the lookup first (FP32), then quantizes only the result.
    For vocab_size=50k, this is ~50k× less work per forward pass.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        use_nf5: bool = True,
        clip_percentile: float | None = 99.9,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.block_size = block_size
        self.use_nf5 = use_nf5
        self.clip_percentile = clip_percentile

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Lookup first (FP32), then quantize only the selected rows
        raw = F.embedding(input, self.weight, self.padding_idx)
        return fake_quantize_v2(
            raw, self.block_size, "nearest", self.use_nf5, self.clip_percentile
        )


# ---------------------------------------------------------------------------
# V2 Multi-Head Attention
# ---------------------------------------------------------------------------


class AXSMultiheadAttentionV2(nn.Module):
    """
    V2 multi-head attention with all optimizations.

    Improvements over v1:
      - All projections use AXSLinearV2 (NF5 + Hadamard + percentile clip)
      - Attention scores are NOT quantized before softmax (preserves precision)
      - Post-softmax attention weights use v2 fake quantize (stochastic rounding)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = DEFAULT_BLOCK_SIZE,
        use_nf5: bool = True,
        clip_percentile: float | None = 99.9,
        use_hadamard: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.use_nf5 = use_nf5
        self.clip_percentile = clip_percentile

        self.q_proj = AXSLinearV2(
            embed_dim, embed_dim, bias=bias, block_size=block_size,
            use_nf5=use_nf5, clip_percentile=clip_percentile,
            use_hadamard=use_hadamard,
        )
        self.k_proj = AXSLinearV2(
            embed_dim, embed_dim, bias=bias, block_size=block_size,
            use_nf5=use_nf5, clip_percentile=clip_percentile,
            use_hadamard=use_hadamard,
        )
        self.v_proj = AXSLinearV2(
            embed_dim, embed_dim, bias=bias, block_size=block_size,
            use_nf5=use_nf5, clip_percentile=clip_percentile,
            use_hadamard=use_hadamard,
        )
        self.out_proj = AXSLinearV2(
            embed_dim, embed_dim, bias=bias, block_size=block_size,
            use_nf5=use_nf5, clip_percentile=clip_percentile,
            use_hadamard=use_hadamard,
        )

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

        # Project Q, K, V (quantized via AXSLinearV2)
        q = self.q_proj(query).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key).reshape(
            batch_size, key.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(value).reshape(
            batch_size, value.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Attention scores — compute in FP32, do NOT quantize before softmax
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Quantize post-softmax attention weights (stochastic rounding)
        attn_weights = fake_quantize_v2(
            attn_weights, self.block_size, "stochastic",
            self.use_nf5, self.clip_percentile,
        )

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )

        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# Model Conversion: v1 → v2
# ---------------------------------------------------------------------------


def convert_to_axs_v2(
    model: nn.Module,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
    quantize_grad: bool = True,
    use_nf5: bool = True,
    clip_percentile: float | None = 99.9,
    use_hadamard: bool = False,
    skip_layers: set[str] | None = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert a standard PyTorch model to use AXS-6 V2 quantized layers.

    Replaces ``nn.Linear``, ``nn.LayerNorm``, and ``nn.Embedding`` with
    V2 counterparts. Preserves pretrained weights.

    Key differences from v1 convert_to_axs:
      - Uses NF5 grid instead of uniform
      - Applies Hadamard rotation
      - Uses percentile clipping
      - LayerNorm output is NOT quantized
      - Embedding uses lazy quantization

    Args:
        model: Model to convert.
        block_size: AXS-6 block size.
        quantize_input: Quantize activations in linear layers.
        quantize_grad: Quantize gradients in linear layers.
        use_nf5: Use NormalFloat-5 grid.
        clip_percentile: Percentile for block scaling.
        use_hadamard: Apply Hadamard rotation on weights.
        skip_layers: Layer names to skip.
        inplace: Modify model in-place.

    Returns:
        Model with V2 AXS-6 quantized layers.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    if skip_layers is None:
        skip_layers = set()

    # Also import v1 modules for isinstance checks during conversion
    from axs.nn.modules import AXSLinear, AXSLayerNorm, AXSEmbedding

    def _convert_module(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if full_name in skip_layers:
                continue

            if isinstance(child, (nn.Linear, AXSLinear)):
                in_f = child.in_features
                out_f = child.out_features
                has_bias = child.bias is not None
                v2_layer = AXSLinearV2(
                    in_f, out_f, bias=has_bias,
                    block_size=block_size,
                    quantize_input=quantize_input,
                    quantize_grad=quantize_grad,
                    use_nf5=use_nf5,
                    clip_percentile=clip_percentile,
                    use_hadamard=use_hadamard,
                )
                v2_layer.weight.data.copy_(child.weight.data)
                if has_bias and v2_layer.bias is not None:
                    v2_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, v2_layer)

            elif isinstance(child, (nn.LayerNorm, AXSLayerNorm)):
                v2_layer = AXSLayerNormV2(
                    list(child.normalized_shape),
                    eps=child.eps,
                )
                v2_layer.weight.data.copy_(child.weight.data)
                v2_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, v2_layer)

            elif isinstance(child, (nn.Embedding, AXSEmbedding)):
                v2_layer = AXSEmbeddingV2(
                    child.num_embeddings,
                    child.embedding_dim,
                    padding_idx=child.padding_idx,
                    block_size=block_size,
                    use_nf5=use_nf5,
                    clip_percentile=clip_percentile,
                )
                v2_layer.weight.data.copy_(child.weight.data)
                setattr(module, name, v2_layer)

            else:
                _convert_module(child, full_name)

    _convert_module(model)
    return model
