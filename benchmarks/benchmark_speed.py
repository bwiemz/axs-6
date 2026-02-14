"""
Benchmark: Computational Throughput
====================================

Measures the computational overhead of AXS-6 quantization operations
and compares quantized matmul throughput.

Run: ``python -m benchmarks.benchmark_speed``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axs.core import DEFAULT_BLOCK_SIZE, dequantize, quantize
from axs.nn.functional import fake_quantize


def time_function(fn, *args, warmup: int = 5, runs: int = 50, **kwargs) -> dict[str, float]:
    """Time a function with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
    }


def benchmark_quantize_speed() -> None:
    """Benchmark quantization and dequantization throughput."""
    print("=" * 80)
    print("QUANTIZATION THROUGHPUT BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sizes = [
        (256, 256),
        (1024, 1024),
        (4096, 4096),
        (8192, 4096),
    ]

    for H, W in sizes:
        tensor = torch.randn(H, W, device=device)
        num_elements = H * W

        print(f"\nTensor size: {H}×{W} = {num_elements:,} elements")

        # Quantize (nearest)
        t = time_function(quantize, tensor, rounding="nearest")
        throughput = num_elements / (t["mean_ms"] / 1000) / 1e9
        print(f"  Quantize (nearest):    {t['mean_ms']:>8.3f} ms  ({throughput:.2f} Gelements/s)")

        # Quantize (stochastic)
        t = time_function(quantize, tensor, rounding="stochastic")
        throughput = num_elements / (t["mean_ms"] / 1000) / 1e9
        print(f"  Quantize (stochastic): {t['mean_ms']:>8.3f} ms  ({throughput:.2f} Gelements/s)")

        # Dequantize
        axs = quantize(tensor)
        t = time_function(dequantize, axs)
        throughput = num_elements / (t["mean_ms"] / 1000) / 1e9
        print(f"  Dequantize:            {t['mean_ms']:>8.3f} ms  ({throughput:.2f} Gelements/s)")

        # Fake quantize (combined, used in training)
        t = time_function(fake_quantize, tensor)
        throughput = num_elements / (t["mean_ms"] / 1000) / 1e9
        print(f"  Fake quantize:         {t['mean_ms']:>8.3f} ms  ({throughput:.2f} Gelements/s)")


def benchmark_matmul_overhead() -> None:
    """
    Benchmark the overhead of AXS-6 quantization in matrix multiplication.
    """
    print("\n" + "=" * 80)
    print("MATMUL OVERHEAD BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    for M, K, N in sizes:
        A = torch.randn(M, K, device=device)
        B = torch.randn(K, N, device=device)

        print(f"\nMatmul: ({M}×{K}) @ ({K}×{N})")

        # FP32 baseline
        t_fp32 = time_function(torch.matmul, A, B)
        flops = 2 * M * K * N
        gflops_fp32 = flops / (t_fp32["mean_ms"] / 1000) / 1e9

        # FP16
        A_fp16 = A.half()
        B_fp16 = B.half()
        t_fp16 = time_function(torch.matmul, A_fp16, B_fp16)
        gflops_fp16 = flops / (t_fp16["mean_ms"] / 1000) / 1e9

        # AXS-6 (fake quantize + matmul)
        def axs_matmul():
            Aq = fake_quantize(A)
            Bq = fake_quantize(B)
            return torch.matmul(Aq, Bq)

        t_axs = time_function(axs_matmul)
        gflops_axs = flops / (t_axs["mean_ms"] / 1000) / 1e9

        overhead = (t_axs["mean_ms"] / t_fp32["mean_ms"] - 1) * 100

        print(f"  FP32:  {t_fp32['mean_ms']:>8.3f} ms  ({gflops_fp32:>8.1f} GFLOPS)")
        print(f"  FP16:  {t_fp16['mean_ms']:>8.3f} ms  ({gflops_fp16:>8.1f} GFLOPS)")
        print(f"  AXS-6: {t_axs['mean_ms']:>8.3f} ms  ({gflops_axs:>8.1f} GFLOPS, +{overhead:.1f}% overhead)")


def benchmark_triton_kernels() -> None:
    """Benchmark Triton kernels if available."""
    print("\n" + "=" * 80)
    print("TRITON KERNEL BENCHMARK")
    print("=" * 80)

    try:
        from axs.triton_kernels.quantize_kernel import (
            triton_fake_quantize,
            triton_quantize,
            triton_dequantize,
            TRITON_AVAILABLE,
        )
        from axs.triton_kernels.matmul_kernel import triton_axs_matmul
    except ImportError:
        print("  Triton not available — skipping")
        return

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("  Triton or CUDA not available — skipping")
        return

    device = "cuda"

    sizes = [(1024, 1024), (4096, 4096)]

    for H, W in sizes:
        tensor = torch.randn(H, W, device=device)
        num_elements = H * W

        print(f"\nTensor size: {H}×{W}")

        # PyTorch fake quantize
        t_pt = time_function(fake_quantize, tensor)
        tp_pt = num_elements / (t_pt["mean_ms"] / 1000) / 1e9

        # Triton fake quantize
        t_tr = time_function(triton_fake_quantize, tensor)
        tp_tr = num_elements / (t_tr["mean_ms"] / 1000) / 1e9

        speedup = t_pt["mean_ms"] / t_tr["mean_ms"]
        print(f"  PyTorch fake_quantize: {t_pt['mean_ms']:>8.3f} ms  ({tp_pt:.2f} G/s)")
        print(f"  Triton fake_quantize:  {t_tr['mean_ms']:>8.3f} ms  ({tp_tr:.2f} G/s, {speedup:.1f}× speedup)")

    # Triton matmul
    for M in [1024, 4096]:
        A = torch.randn(M, M, device=device)
        B = torch.randn(M, M, device=device)

        print(f"\nMatmul: {M}×{M}")

        t_fp32 = time_function(torch.matmul, A, B)
        t_triton = time_function(triton_axs_matmul, A, B)

        overhead = (t_triton["mean_ms"] / t_fp32["mean_ms"] - 1) * 100
        print(f"  FP32 matmul:    {t_fp32['mean_ms']:>8.3f} ms")
        print(f"  Triton AXS-6:   {t_triton['mean_ms']:>8.3f} ms  ({overhead:+.1f}% vs FP32)")


if __name__ == "__main__":
    torch.manual_seed(42)
    benchmark_quantize_speed()
    benchmark_matmul_overhead()
    benchmark_triton_kernels()
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
