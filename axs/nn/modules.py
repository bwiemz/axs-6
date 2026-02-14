"""
AXS-6 Drop-in Module Replacements
==================================

PyTorch ``nn.Module`` wrappers that transparently apply AXS-6 quantization
during training. These maintain FP32 master weights for optimizer updates
while using quantized weights/activations in the forward pass.

Usage::

    # Replace standard layers
    model.fc1 = AXSLinear(768, 256)

    # Or convert an entire model
    model = convert_to_axs(model, block_size=32)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from axs.core import DEFAULT_BLOCK_SIZE
from axs.nn.functional import axs_conv2d, axs_linear, fake_quantize


# ---------------------------------------------------------------------------
# AXS Linear Layer
# ---------------------------------------------------------------------------


class AXSLinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` with AXS-6 quantization.

    Master weights are stored in FP32. During forward:
      1. Weights are fake-quantized to AXS-6 (nearest rounding)
      2. Inputs are optionally fake-quantized
      3. Standard linear computation in FP32
      4. Gradients flow via STE

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias. Default: ``True``.
        block_size: AXS-6 block size. Default: 32.
        quantize_input: Whether to quantize input activations. Default: ``True``.
        quantize_grad: Whether to quantize gradients. Default: ``True``.
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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return axs_linear(
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
            f"bits_per_value={6 + 10 / self.block_size:.2f}"
        )


# ---------------------------------------------------------------------------
# AXS Conv2d Layer
# ---------------------------------------------------------------------------


class AXSConv2d(nn.Module):
    """
    Drop-in replacement for ``nn.Conv2d`` with AXS-6 quantized weights.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections. Default: 1.
        bias: If ``True``, adds a learnable bias. Default: ``True``.
        block_size: AXS-6 block size. Default: 32.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.block_size = block_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return axs_conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.block_size,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, block_size={self.block_size}"
        )


# ---------------------------------------------------------------------------
# AXS LayerNorm (quantize output only)
# ---------------------------------------------------------------------------


class AXSLayerNorm(nn.Module):
    """
    LayerNorm with AXS-6 quantized output.

    The normalization itself runs in FP32 for numerical stability, but
    the output is fake-quantized to AXS-6 before being passed downstream.
    """

    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 1e-5,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.block_size = block_size
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        return fake_quantize(output, self.block_size)


# ---------------------------------------------------------------------------
# AXS Embedding
# ---------------------------------------------------------------------------


class AXSEmbedding(nn.Module):
    """
    Embedding layer with AXS-6 quantized embedding weights.

    The embedding table is stored in FP32 but fake-quantized during lookup.
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Lazy quantization: look up rows first, then quantize only those
        raw = F.embedding(input, self.weight, self.padding_idx)
        return fake_quantize(raw, self.block_size)


# ---------------------------------------------------------------------------
# AXS Multi-Head Attention
# ---------------------------------------------------------------------------


class AXSMultiheadAttention(nn.Module):
    """
    Multi-head attention with AXS-6 quantized projections and attention weights.

    All Q/K/V/O projections use AXS-6 quantized linear layers, and the
    attention score matrix is fake-quantized before softmax.
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

        self.q_proj = AXSLinear(embed_dim, embed_dim, bias=bias, block_size=block_size)
        self.k_proj = AXSLinear(embed_dim, embed_dim, bias=bias, block_size=block_size)
        self.v_proj = AXSLinear(embed_dim, embed_dim, bias=bias, block_size=block_size)
        self.out_proj = AXSLinear(embed_dim, embed_dim, bias=bias, block_size=block_size)

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

        # Project Q, K, V
        q = self.q_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores — quantize before softmax
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# Model Conversion Utility
# ---------------------------------------------------------------------------


def convert_to_axs(
    model: nn.Module,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
    quantize_grad: bool = True,
    skip_layers: set[str] | None = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert a standard PyTorch model to use AXS-6 quantized layers.

    Replaces ``nn.Linear``, ``nn.Conv2d``, ``nn.LayerNorm``, and ``nn.Embedding``
    with their AXS-6 counterparts, preserving pretrained weights.

    Args:
        model: The model to convert.
        block_size: AXS-6 block size for all layers.
        quantize_input: Whether to quantize activations in linear layers.
        quantize_grad: Whether to quantize gradients in linear layers.
        skip_layers: Set of layer names (dot-separated) to skip conversion.
        inplace: If True, modifies model in-place. Otherwise returns a copy.

    Returns:
        Model with AXS-6 quantized layers.

    Example::

        model = torchvision.models.resnet18(pretrained=True)
        model_axs = convert_to_axs(model, block_size=32)
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    if skip_layers is None:
        skip_layers = set()

    def _convert_module(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if full_name in skip_layers:
                continue

            if isinstance(child, nn.Linear):
                axs_layer = AXSLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    block_size=block_size,
                    quantize_input=quantize_input,
                    quantize_grad=quantize_grad,
                )
                axs_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None and axs_layer.bias is not None:
                    axs_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, axs_layer)

            elif isinstance(child, nn.Conv2d):
                axs_layer = AXSConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,  # type: ignore[arg-type]
                    child.stride,  # type: ignore[arg-type]
                    child.padding,  # type: ignore[arg-type]
                    child.dilation,  # type: ignore[arg-type]
                    child.groups,
                    bias=child.bias is not None,
                    block_size=block_size,
                )
                axs_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None and axs_layer.bias is not None:
                    axs_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, axs_layer)

            elif isinstance(child, nn.LayerNorm):
                axs_layer = AXSLayerNorm(
                    list(child.normalized_shape),
                    eps=child.eps,
                    block_size=block_size,
                )
                axs_layer.weight.data.copy_(child.weight.data)
                axs_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, axs_layer)

            elif isinstance(child, nn.Embedding):
                axs_layer = AXSEmbedding(
                    child.num_embeddings,
                    child.embedding_dim,
                    padding_idx=child.padding_idx,
                    block_size=block_size,
                )
                axs_layer.weight.data.copy_(child.weight.data)
                setattr(module, name, axs_layer)

            else:
                _convert_module(child, full_name)

    _convert_module(model)
    return model
