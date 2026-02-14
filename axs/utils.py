"""
Utility Functions for AXS-6
============================

Helper functions for tensor analysis, block statistics, format comparison,
and diagnostic tools.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from axs.core import (
    AXS6_EXPONENT_BIAS,
    AXS6_HEADER_BITS,
    AXS6_MAX_MAGNITUDE,
    AXS6_VALUE_BITS,
    DEFAULT_BLOCK_SIZE,
    VALID_BLOCK_SIZES,
    AXSTensor,
    dequantize,
    quantize,
)


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    """
    Compute comprehensive statistics for a tensor, useful for understanding
    quantization behavior.

    Returns dict with: mean, std, min, max, abs_mean, abs_max, sparsity,
    kurtosis, dynamic_range_db.
    """
    t = tensor.float()
    abs_t = t.abs()

    # Sparsity: fraction of values below 1e-6 * max
    threshold = abs_t.max().item() * 1e-6
    sparsity = (abs_t < threshold).float().mean().item()

    # Kurtosis: measure of tail heaviness
    mean = t.mean()
    std = t.std()
    if std > 0:
        kurtosis = ((t - mean) / std).pow(4).mean().item()
    else:
        kurtosis = 0.0

    # Dynamic range in dB
    abs_max = abs_t.max().item()
    abs_min = abs_t[abs_t > 0].min().item() if (abs_t > 0).any() else 1e-45
    dynamic_range_db = 20 * math.log10(abs_max / abs_min) if abs_min > 0 else 0.0

    return {
        "mean": t.mean().item(),
        "std": std.item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "abs_mean": abs_t.mean().item(),
        "abs_max": abs_max,
        "sparsity": sparsity,
        "kurtosis": kurtosis,
        "dynamic_range_db": dynamic_range_db,
    }


def block_distribution_analysis(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    """
    Analyze how well-suited a tensor is for AXS-6 block quantization.

    Computes per-block statistics to assess whether the shared-exponent
    assumption holds (i.e., values within each block have similar magnitudes).

    Returns:
        Dict with:
          - intra_block_cv: Mean coefficient of variation within blocks
            (lower = better suited for AXS-6)
          - inter_block_range_db: Dynamic range across block max values
          - block_utilization: Mean fraction of the [0, 31] code range used
          - outlier_block_fraction: Fraction of blocks with high CV (>2.0)
    """
    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    last_dim = flat.shape[-1]
    pad = (block_size - last_dim % block_size) % block_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))

    num_blocks = flat.shape[-1] // block_size
    blocked = flat.reshape(-1, num_blocks, block_size)
    abs_blocked = blocked.abs()

    # Per-block coefficient of variation
    block_means = abs_blocked.mean(dim=-1)
    block_stds = abs_blocked.std(dim=-1)
    safe_means = block_means.clamp(min=1e-45)
    cvs = block_stds / safe_means
    mean_cv = cvs.mean().item()

    # Dynamic range across blocks
    block_maxes = abs_blocked.amax(dim=-1)
    nz_maxes = block_maxes[block_maxes > 0]
    if len(nz_maxes) > 1:
        inter_range_db = 20 * math.log10(
            nz_maxes.max().item() / nz_maxes.min().item()
        )
    else:
        inter_range_db = 0.0

    # Block utilization: how many of the 32 code levels are actually needed
    axs = quantize(tensor, block_size=block_size)
    unique_per_block = []
    mags_flat = axs.magnitudes.reshape(-1, block_size)
    for i in range(min(1000, mags_flat.shape[0])):  # sample up to 1000 blocks
        unique = mags_flat[i].unique().numel()
        unique_per_block.append(unique / (AXS6_MAX_MAGNITUDE + 1))
    utilization = sum(unique_per_block) / len(unique_per_block) if unique_per_block else 0.0

    # Outlier blocks
    outlier_frac = (cvs > 2.0).float().mean().item()

    return {
        "intra_block_cv": mean_cv,
        "inter_block_range_db": inter_range_db,
        "block_utilization": utilization,
        "outlier_block_fraction": outlier_frac,
        "num_blocks_total": int(blocked.shape[0] * blocked.shape[1]),
    }


def memory_comparison(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    """
    Compare memory usage of the tensor in different formats.

    Returns dict mapping format names to byte counts and compression ratios.
    """
    numel = tensor.numel()

    formats = {
        "FP32": numel * 4,
        "FP16": numel * 2,
        "BF16": numel * 2,
        "FP8": numel * 1,
        "AXS-6": math.ceil(
            (numel * AXS6_VALUE_BITS + (numel // block_size) * AXS6_HEADER_BITS) / 8
        ),
        "INT4": math.ceil(numel * 0.5),
    }

    # Compression ratios relative to FP32
    ratios = {k: formats["FP32"] / v for k, v in formats.items()}

    return {
        "bytes": formats,
        "compression_vs_fp32": ratios,
        "savings_vs_fp8_pct": (1 - formats["AXS-6"] / formats["FP8"]) * 100,
    }


def format_comparison_table(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> str:
    """
    Generate a formatted comparison table of quantization error across formats.

    Returns a string table for display.
    """
    from axs.core import quantization_error

    results: list[dict[str, Any]] = []

    # AXS-6
    axs_err = quantization_error(tensor, block_size=block_size)
    axs_mem = memory_comparison(tensor, block_size)
    results.append({
        "Format": f"AXS-6 (B={block_size})",
        "Bits/val": f"{AXS6_VALUE_BITS + AXS6_HEADER_BITS / block_size:.2f}",
        "Bytes": axs_mem["bytes"]["AXS-6"],
        "RMSE": f"{axs_err['rmse']:.6f}",
        "SNR (dB)": f"{axs_err['signal_to_noise_db']:.1f}",
        "Max Error": f"{axs_err['max_abs_error']:.6f}",
    })

    # FP8 simulation (E4M3-like: 4-bit exponent, 3-bit mantissa)
    fp8_q = _simulate_fp8_e4m3(tensor)
    fp8_err_t = (tensor.float() - fp8_q.float())
    fp8_mse = fp8_err_t.pow(2).mean().item()
    fp8_snr = 10 * math.log10(tensor.float().pow(2).mean().item() / max(fp8_mse, 1e-45))
    results.append({
        "Format": "FP8 E4M3",
        "Bits/val": "8.00",
        "Bytes": tensor.numel(),
        "RMSE": f"{math.sqrt(fp8_mse):.6f}",
        "SNR (dB)": f"{fp8_snr:.1f}",
        "Max Error": f"{fp8_err_t.abs().max().item():.6f}",
    })

    # FP16
    fp16_q = tensor.half().float()
    fp16_err_t = tensor.float() - fp16_q
    fp16_mse = fp16_err_t.pow(2).mean().item()
    fp16_snr = 10 * math.log10(tensor.float().pow(2).mean().item() / max(fp16_mse, 1e-45))
    results.append({
        "Format": "FP16",
        "Bits/val": "16.00",
        "Bytes": tensor.numel() * 2,
        "RMSE": f"{math.sqrt(fp16_mse):.6f}",
        "SNR (dB)": f"{fp16_snr:.1f}",
        "Max Error": f"{fp16_err_t.abs().max().item():.6f}",
    })

    # Build table
    headers = ["Format", "Bits/val", "Bytes", "RMSE", "SNR (dB)", "Max Error"]
    col_widths = [max(len(h), max(len(str(r[h])) for r in results)) for h in headers]

    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-+-".join("-" * w for w in col_widths))
    for r in results:
        line = " | ".join(str(r[h]).ljust(w) for h, w in zip(headers, col_widths))
        lines.append(line)

    return "\n".join(lines)


def _simulate_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """
    Simulate FP8 E4M3 quantization (4-bit exponent, 3-bit mantissa).

    This is a software simulation since PyTorch may not have native FP8 support.
    Uses the same encode/decode approach as the IEEE FP8 E4M3 standard.
    """
    # E4M3: bias=7, max_exp=15, mantissa_levels=8
    x = tensor.float()
    sign = x.sign()
    abs_x = x.abs()

    # Clamp to representable range: max value = 1.75 * 2^8 = 448
    abs_x = abs_x.clamp(max=448.0)

    # For each value, compute exponent and mantissa
    safe_abs = abs_x.clamp(min=2**-9)  # min subnormal
    exp = safe_abs.log2().floor().clamp(-6, 8)
    scale = torch.pow(2.0, exp)
    mantissa = (safe_abs / scale).clamp(1.0, 2.0)  # normalized: [1, 2)

    # Quantize mantissa to 3 bits (8 levels in [1, 2))
    mantissa_q = ((mantissa - 1.0) * 8).round() / 8 + 1.0

    result = sign * mantissa_q * scale
    result[abs_x < 2**-9] = 0  # flush subnormals to zero in this simulation
    return result
