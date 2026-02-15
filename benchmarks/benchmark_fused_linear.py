"""
Benchmark: Fused FQ+Matmul Triton Kernel vs Separate Path
==========================================================

Compares three approaches for AXS-6 quantised linear:

1. **Fused** — ``triton_fused_linear``: single Triton kernel that applies
   NF5 fake-quantise on-the-fly during the matmul tile loop.
2. **Separate** — ``triton_fused_fake_quantize`` × 2 + ``F.linear``:
   three separate kernel launches.
3. **cuBLAS** — ``F.linear`` only (no quantisation baseline).

Usage::

    python benchmarks/benchmark_fused_linear.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from axs.unified.triton_kernels import (
    triton_fused_fake_quantize,
    triton_fused_linear,
)


def _bench(fn: callable, warmup: int = 20, iters: int = 100) -> float:
    """Return mean latency in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main() -> None:
    device = "cuda"
    print("=" * 75)
    print("FUSED FQ+MATMUL TRITON KERNEL BENCHMARK")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print("=" * 75)

    shapes = [
        (64, 64, 128),
        (128, 64, 256),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    print()
    print(
        f"{'Shape':>20s}  {'Separate':>10s}  {'Fused':>10s}  "
        f"{'cuBLAS':>10s}  {'Fused/Sep':>10s}  {'Winner':>8s}"
    )
    print("-" * 75)

    for M, N, K in shapes:
        x = torch.randn(M, K, device=device)
        w = torch.randn(N, K, device=device)
        b = torch.randn(N, device=device)

        def _separate() -> torch.Tensor:
            wq = triton_fused_fake_quantize(w, 32, "nearest")
            xq = triton_fused_fake_quantize(x, 32, "nearest")
            return F.linear(xq, wq, b)

        def _fused() -> torch.Tensor:
            return triton_fused_linear(x, w, b, 32, True)

        def _cublas() -> torch.Tensor:
            return F.linear(x, w, b)

        t_sep = _bench(_separate)
        t_fused = _bench(_fused)
        t_cub = _bench(_cublas)
        ratio = t_fused / t_sep
        winner = "fused" if t_fused < t_sep else "separate"

        shape_str = f"{M}x{K}->{N}"
        print(
            f"{shape_str:>20s}  {t_sep:>8.3f}ms  {t_fused:>8.3f}ms  "
            f"{t_cub:>8.3f}ms  {ratio:>9.2f}x  {winner:>8s}"
        )

    print()
    print("Separate = triton_fq(w) + triton_fq(x) + F.linear  (3 kernels)")
    print("Fused    = triton_fused_linear                      (1 kernel)")
    print("cuBLAS   = F.linear                                 (no quant)")


if __name__ == "__main__":
    main()
