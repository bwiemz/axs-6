"""
Benchmark: AXS-6 vs FP8/FP16/BF16 Precision Comparison
=======================================================

Comprehensive comparison of quantization accuracy across formats using
realistic neural network weight distributions.

Run: ``python -m benchmarks.benchmark_precision``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axs.core import dequantize, quantize, quantization_error
from axs.utils import _simulate_fp8_e4m3, format_comparison_table, tensor_stats


def _simulate_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate BF16 quantization."""
    return tensor.bfloat16().float()


def _simulate_int8_symmetric(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate symmetric INT8 quantization."""
    abs_max = tensor.abs().max()
    scale = abs_max / 127.0
    quantized = (tensor / scale).round().clamp(-128, 127)
    return quantized * scale


def _compute_error_stats(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    """Compute error statistics between original and reconstructed tensors."""
    error = (original.float() - reconstructed.float())
    mse = error.pow(2).mean().item()
    signal_power = original.float().pow(2).mean().item()
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "max_abs_error": error.abs().max().item(),
        "mean_abs_error": error.abs().mean().item(),
        "snr_db": 10 * math.log10(signal_power / max(mse, 1e-45)),
        "relative_error": math.sqrt(mse) / (original.float().pow(2).mean().sqrt().item() + 1e-45),
    }


def benchmark_distributions() -> None:
    """
    Benchmark quantization error across different tensor distributions
    that commonly appear in neural networks.
    """
    print("=" * 80)
    print("AXS-6 FORMAT PRECISION BENCHMARK")
    print("=" * 80)

    distributions = {
        "Normal(0, 1)": lambda: torch.randn(1024, 1024),
        "Normal(0, 0.02)": lambda: torch.randn(1024, 1024) * 0.02,
        "Uniform[-1, 1]": lambda: torch.rand(1024, 1024) * 2 - 1,
        "Kaiming Normal": lambda: torch.nn.init.kaiming_normal_(torch.empty(1024, 1024)),
        "Xavier Normal": lambda: torch.nn.init.xavier_normal_(torch.empty(1024, 1024)),
        "Post-LayerNorm": lambda: torch.nn.functional.layer_norm(
            torch.randn(1024, 1024), [1024]
        ),
        "Sparse (50% zeros)": lambda: torch.randn(1024, 1024) * (torch.rand(1024, 1024) > 0.5).float(),
        "Heavy-tailed": lambda: torch.randn(1024, 1024) * torch.randn(1024, 1024).abs().pow(0.5),
        "Outlier-rich": lambda: _create_outlier_tensor(1024, 1024, outlier_fraction=0.01),
    }

    formats = {
        "AXS-6 (B=32)": lambda t: dequantize(quantize(t, block_size=32)),
        "AXS-6 (B=16)": lambda t: dequantize(quantize(t, block_size=16)),
        "AXS-6 (B=8)": lambda t: dequantize(quantize(t, block_size=8)),
        "FP8 E4M3": lambda t: _simulate_fp8_e4m3(t),
        "FP16": lambda t: t.half().float(),
        "BF16": lambda t: _simulate_bf16(t),
        "INT8 Sym": lambda t: _simulate_int8_symmetric(t),
    }

    for dist_name, gen_fn in distributions.items():
        print(f"\n{'─' * 80}")
        print(f"Distribution: {dist_name}")
        tensor = gen_fn()
        stats = tensor_stats(tensor)
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}], "
              f"Std: {stats['std']:.4f}, Kurtosis: {stats['kurtosis']:.2f}, "
              f"Sparsity: {stats['sparsity']:.1%}")
        print()

        print(f"  {'Format':<20} {'Bits/val':>10} {'RMSE':>12} {'SNR (dB)':>10} "
              f"{'Max Error':>12} {'Rel Error':>12}")
        print(f"  {'─' * 20} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 12} {'─' * 12}")

        for fmt_name, fmt_fn in formats.items():
            reconstructed = fmt_fn(tensor)
            err = _compute_error_stats(tensor, reconstructed)

            bits = _format_bits(fmt_name)
            print(f"  {fmt_name:<20} {bits:>10} {err['rmse']:>12.6f} "
                  f"{err['snr_db']:>10.1f} {err['max_abs_error']:>12.6f} "
                  f"{err['relative_error']:>12.6f}")


def _create_outlier_tensor(
    rows: int, cols: int, outlier_fraction: float = 0.01
) -> torch.Tensor:
    """Create a tensor with a small fraction of outlier values."""
    tensor = torch.randn(rows, cols) * 0.1
    mask = torch.rand(rows, cols) < outlier_fraction
    tensor[mask] = torch.randn(mask.sum().item()) * 10
    return tensor


def _format_bits(name: str) -> str:
    """Return bits/value string for a format name."""
    bits_map = {
        "AXS-6 (B=32)": "6.31",
        "AXS-6 (B=16)": "6.63",
        "AXS-6 (B=8)": "7.25",
        "FP8 E4M3": "8.00",
        "FP16": "16.00",
        "BF16": "16.00",
        "INT8 Sym": "8.00",
    }
    return bits_map.get(name, "?")


def benchmark_block_sizes() -> None:
    """Analyze how block size affects quantization quality."""
    print("\n" + "=" * 80)
    print("BLOCK SIZE SENSITIVITY ANALYSIS")
    print("=" * 80)

    tensor = torch.randn(4096, 4096)

    for bs in [8, 16, 32]:
        err = quantization_error(tensor, block_size=bs)
        bits = 6 + 10 / bs
        print(f"\n  Block size={bs:>2}: "
              f"{bits:.2f} bits/val | "
              f"RMSE={err['rmse']:.6f} | "
              f"SNR={err['signal_to_noise_db']:.1f} dB | "
              f"Max Error={err['max_abs_error']:.6f}")


def benchmark_stochastic_rounding() -> None:
    """Demonstrate unbiasedness of stochastic rounding."""
    print("\n" + "=" * 80)
    print("STOCHASTIC ROUNDING BIAS ANALYSIS")
    print("=" * 80)

    tensor = torch.randn(512, 512)
    num_trials = 100

    nearest_errors = []
    stochastic_errors = []

    for _ in range(num_trials):
        # Nearest
        q_n = dequantize(quantize(tensor, rounding="nearest"))
        nearest_errors.append((tensor - q_n).mean().item())

        # Stochastic
        q_s = dequantize(quantize(tensor, rounding="stochastic"))
        stochastic_errors.append((tensor - q_s).mean().item())

    nearest_bias = sum(nearest_errors) / len(nearest_errors)
    stochastic_bias = sum(stochastic_errors) / len(stochastic_errors)
    stochastic_std = (sum((e - stochastic_bias)**2 for e in stochastic_errors) / len(stochastic_errors)) ** 0.5

    print(f"\n  Nearest rounding bias:    {nearest_bias:+.8f}")
    print(f"  Stochastic rounding bias: {stochastic_bias:+.8f}")
    print(f"  Stochastic std of bias:   {stochastic_std:.8f}")
    print(f"  Stochastic is {'UNBIASED' if abs(stochastic_bias) < 3 * stochastic_std / num_trials**0.5 else 'BIASED'}")


if __name__ == "__main__":
    torch.manual_seed(42)
    benchmark_distributions()
    benchmark_block_sizes()
    benchmark_stochastic_rounding()
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
