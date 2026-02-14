"""
Benchmark: Memory Footprint Comparison
=======================================

Compares memory usage of AXS-6 against FP8, FP16, BF16, and FP32
for different model sizes.

Run: ``python -m benchmarks.benchmark_memory``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axs.core import AXS6_HEADER_BITS, AXS6_VALUE_BITS


def compute_model_memory(
    num_params: int,
    bits_per_param: float,
) -> dict[str, float]:
    """Compute memory in different units."""
    total_bits = num_params * bits_per_param
    total_bytes = total_bits / 8
    return {
        "bits": total_bits,
        "bytes": total_bytes,
        "KB": total_bytes / 1024,
        "MB": total_bytes / (1024**2),
        "GB": total_bytes / (1024**3),
    }


def benchmark_model_sizes() -> None:
    """Compare memory for different model sizes and formats."""
    print("=" * 90)
    print("MEMORY FOOTPRINT COMPARISON")
    print("=" * 90)

    models = {
        "GPT-2 Small (124M)": 124_000_000,
        "GPT-2 Medium (355M)": 355_000_000,
        "GPT-2 Large (774M)": 774_000_000,
        "GPT-2 XL (1.5B)": 1_500_000_000,
        "LLaMA-7B": 7_000_000_000,
        "LLaMA-13B": 13_000_000_000,
        "LLaMA-70B": 70_000_000_000,
        "GPT-4 class (1.8T est.)": 1_800_000_000_000,
    }

    format_bits = {
        "FP32": 32.0,
        "FP16": 16.0,
        "BF16": 16.0,
        "FP8": 8.0,
        "AXS-6 (B=32)": AXS6_VALUE_BITS + AXS6_HEADER_BITS / 32,
        "AXS-6 (B=16)": AXS6_VALUE_BITS + AXS6_HEADER_BITS / 16,
        "INT4": 4.0,
    }

    for model_name, num_params in models.items():
        print(f"\n{'─' * 90}")
        print(f"Model: {model_name}")
        print(f"  {'Format':<20} {'Bits/param':>10} {'Size':>12} {'vs FP32':>10} {'vs FP8':>10}")
        print(f"  {'─' * 20} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 10}")

        fp32_mem = compute_model_memory(num_params, 32.0)

        for fmt_name, bits in format_bits.items():
            mem = compute_model_memory(num_params, bits)

            # Choose appropriate unit
            if mem["GB"] >= 1:
                size_str = f"{mem['GB']:.2f} GB"
            elif mem["MB"] >= 1:
                size_str = f"{mem['MB']:.1f} MB"
            else:
                size_str = f"{mem['KB']:.1f} KB"

            ratio_fp32 = f"{bits / 32:.2%}"
            ratio_fp8 = f"{bits / 8:.2%}"

            print(f"  {fmt_name:<20} {bits:>10.2f} {size_str:>12} {ratio_fp32:>10} {ratio_fp8:>10}")


def benchmark_training_memory() -> None:
    """
    Compare total training memory (including optimizer states, activations,
    and gradients) across formats.
    """
    print("\n" + "=" * 90)
    print("TRAINING MEMORY BREAKDOWN (1B parameter model)")
    print("=" * 90)

    num_params = 1_000_000_000
    batch_size = 32
    seq_len = 2048
    hidden_dim = 4096  # typical for 1B model
    num_layers = 24

    # Activation memory per layer (approximate): batch × seq × hidden × 2 (for Q,K,V blocks)
    activation_elements = batch_size * seq_len * hidden_dim * 4  # factor 4 for layernorm, attn, ff

    configs = {
        "FP32 Baseline": {
            "master_weights": num_params * 32,
            "forward_weights": 0,  # same as master
            "gradients": num_params * 32,
            "optimizer_state": num_params * 32 * 2,  # Adam: m and v
            "activations": activation_elements * num_layers * 32,
        },
        "FP16 Mixed Precision": {
            "master_weights": num_params * 32,
            "forward_weights": num_params * 16,
            "gradients": num_params * 16,
            "optimizer_state": num_params * 32 * 2,
            "activations": activation_elements * num_layers * 16,
        },
        "FP8 Mixed Precision": {
            "master_weights": num_params * 32,
            "forward_weights": num_params * 8,
            "gradients": num_params * 8,
            "optimizer_state": num_params * 32 * 2,
            "activations": activation_elements * num_layers * 8,
        },
        "AXS-6 Mixed Precision": {
            "master_weights": num_params * 32,
            "forward_weights": num_params * 6.3125,
            "gradients": num_params * 6.3125,
            "optimizer_state": num_params * 32 * 2,
            "activations": activation_elements * num_layers * 6.3125,
        },
    }

    for config_name, components in configs.items():
        total_bits = sum(components.values())
        total_gb = total_bits / 8 / (1024**3)

        print(f"\n  {config_name}:")
        for comp_name, bits in components.items():
            gb = bits / 8 / (1024**3)
            print(f"    {comp_name:<24}: {gb:>8.2f} GB")
        print(f"    {'─' * 35}")
        print(f"    {'TOTAL':<24}: {total_gb:>8.2f} GB")

    # Summary
    fp32_total = sum(configs["FP32 Baseline"].values()) / 8 / (1024**3)
    fp8_total = sum(configs["FP8 Mixed Precision"].values()) / 8 / (1024**3)
    axs_total = sum(configs["AXS-6 Mixed Precision"].values()) / 8 / (1024**3)

    print(f"\n  {'─' * 50}")
    print(f"  AXS-6 saves {(1 - axs_total/fp8_total)*100:.1f}% vs FP8 training")
    print(f"  AXS-6 saves {(1 - axs_total/fp32_total)*100:.1f}% vs FP32 training")
    print(f"  Absolute saving vs FP8: {fp8_total - axs_total:.2f} GB")


if __name__ == "__main__":
    benchmark_model_sizes()
    benchmark_training_memory()
    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)
