"""
Benchmark: FP32 AXS-6 vs BF16 Mixed-Precision AXS-6
=====================================================

Compares:
  - Forward pass latency (ms)
  - Forward + backward latency (ms)
  - Peak activation memory per layer
  - Training convergence (loss over steps)

Run::

    python benchmarks/benchmark_mixed_precision.py
"""

from __future__ import annotations

import gc
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from axs.unified.modules_unified import AXSLinearUnified
from axs.unified.mixed_precision import (
    AXSLinearMixedPrecision,
    estimate_memory_savings,
)


def _sync() -> None:
    torch.cuda.synchronize()


def _warmup(fn, n: int = 5) -> None:
    for _ in range(n):
        fn()
    _sync()


def _time_fn(fn, n: int = 50) -> float:
    """Time a function in milliseconds (median of n runs)."""
    _sync()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2]


def benchmark_forward(M: int, K: int, N: int) -> dict[str, float]:
    """Benchmark forward pass: FP32 vs BF16 mixed-precision."""
    x = torch.randn(M, K, device="cuda")

    # FP32 AXS-6
    layer_fp32 = AXSLinearUnified(K, N).cuda()
    _warmup(lambda: layer_fp32(x))
    t_fp32 = _time_fn(lambda: layer_fp32(x))

    # BF16 mixed-precision AXS-6
    layer_bf16 = AXSLinearMixedPrecision(K, N).cuda()
    layer_bf16.weight.data.copy_(layer_fp32.weight.data)
    if layer_bf16.bias is not None and layer_fp32.bias is not None:
        layer_bf16.bias.data.copy_(layer_fp32.bias.data)
    _warmup(lambda: layer_bf16(x))
    t_bf16 = _time_fn(lambda: layer_bf16(x))

    return {
        "fp32_fwd_ms": round(t_fp32, 4),
        "bf16_fwd_ms": round(t_bf16, 4),
        "speedup": round(t_fp32 / t_bf16, 2),
    }


def benchmark_forward_backward(M: int, K: int, N: int) -> dict[str, float]:
    """Benchmark forward + backward: FP32 vs BF16 mixed-precision."""
    x_fp32 = torch.randn(M, K, device="cuda", requires_grad=True)

    # FP32 AXS-6
    layer_fp32 = AXSLinearUnified(K, N).cuda()

    def fwd_bwd_fp32() -> None:
        out = layer_fp32(x_fp32)
        loss = out.sum()
        loss.backward()
        layer_fp32.zero_grad()
        if x_fp32.grad is not None:
            x_fp32.grad = None

    _warmup(fwd_bwd_fp32)
    t_fp32 = _time_fn(fwd_bwd_fp32)

    # BF16 mixed-precision
    x_bf16 = torch.randn(M, K, device="cuda", requires_grad=True)
    layer_bf16 = AXSLinearMixedPrecision(K, N).cuda()
    layer_bf16.weight.data.copy_(layer_fp32.weight.data)
    if layer_bf16.bias is not None and layer_fp32.bias is not None:
        layer_bf16.bias.data.copy_(layer_fp32.bias.data)

    def fwd_bwd_bf16() -> None:
        out = layer_bf16(x_bf16)
        loss = out.float().sum()
        loss.backward()
        layer_bf16.zero_grad()
        if x_bf16.grad is not None:
            x_bf16.grad = None

    _warmup(fwd_bwd_bf16)
    t_bf16 = _time_fn(fwd_bwd_bf16)

    return {
        "fp32_fwd_bwd_ms": round(t_fp32, 4),
        "bf16_fwd_bwd_ms": round(t_bf16, 4),
        "speedup": round(t_fp32 / t_bf16, 2),
    }


def benchmark_memory(M: int, K: int, N: int) -> dict[str, str]:
    """Measure activation memory via estimate_memory_savings."""
    model = nn.Sequential(AXSLinearMixedPrecision(K, N))
    est = estimate_memory_savings(model, batch_size=M)
    return {
        "fp32_activation_mb": f"{est['fp32_mb']:.2f}",
        "bf16_recomp_mb": f"{est['bf16_recomp_mb']:.2f}",
        "savings_ratio": f"{est['savings_ratio']:.1f}x",
    }


def benchmark_convergence(
    in_f: int = 128,
    out_f: int = 64,
    steps: int = 200,
    batch: int = 32,
) -> dict[str, list[float]]:
    """Compare training convergence (FP32 vs BF16)."""
    torch.manual_seed(42)
    W_target = torch.randn(out_f, in_f, device="cuda")
    b_target = torch.randn(out_f, device="cuda")

    results: dict[str, list[float]] = {}

    for label, layer_cls in [
        ("fp32", AXSLinearUnified),
        ("bf16", AXSLinearMixedPrecision),
    ]:
        torch.manual_seed(42)
        layer = layer_cls(in_f, out_f).cuda()
        opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
        losses = []

        for _ in range(steps):
            x = torch.randn(batch, in_f, device="cuda")
            y = F.linear(x, W_target, b_target)
            pred = layer(x).float()
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        results[label] = losses

    return results


def main() -> None:
    print("=" * 70)
    print("AXS-6: FP32 vs BF16 Mixed-Precision Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    shapes = [
        (32, 128, 64),
        (128, 512, 256),
        (256, 1024, 512),
        (512, 2048, 1024),
    ]

    # --- Forward latency ---
    print("Forward Pass Latency (ms)")
    print("-" * 55)
    print(f"{'Shape':>20}  {'FP32':>8}  {'BF16':>8}  {'Speedup':>8}")
    print("-" * 55)
    for M, K, N in shapes:
        r = benchmark_forward(M, K, N)
        print(
            f"  {M}×{K}→{N:>4}  "
            f"{r['fp32_fwd_ms']:>8.3f}  "
            f"{r['bf16_fwd_ms']:>8.3f}  "
            f"{r['speedup']:>7.2f}×"
        )
    print()

    # --- Forward + backward latency ---
    print("Forward + Backward Latency (ms)")
    print("-" * 55)
    print(f"{'Shape':>20}  {'FP32':>8}  {'BF16':>8}  {'Speedup':>8}")
    print("-" * 55)
    for M, K, N in shapes:
        r = benchmark_forward_backward(M, K, N)
        print(
            f"  {M}×{K}→{N:>4}  "
            f"{r['fp32_fwd_bwd_ms']:>8.3f}  "
            f"{r['bf16_fwd_bwd_ms']:>8.3f}  "
            f"{r['speedup']:>7.2f}×"
        )
    print()

    # --- Activation memory ---
    print("Activation Memory Savings")
    print("-" * 55)
    print(f"{'Shape':>20}  {'FP32 (MB)':>10} {'BF16+R (MB)':>12} {'Savings':>10}")
    print("-" * 55)
    for M, K, N in shapes:
        r = benchmark_memory(M, K, N)
        print(
            f"  {M}×{K}→{N:>4}  "
            f"{r['fp32_activation_mb']:>10}  "
            f"{r['bf16_recomp_mb']:>10}  "
            f"{r['savings_ratio']:>10}"
        )
    print()

    # --- Convergence comparison ---
    print("Training Convergence (200 steps, 128→64)")
    print("-" * 55)
    conv = benchmark_convergence()
    for label in ["fp32", "bf16"]:
        losses = conv[label]
        print(
            f"  {label.upper():>4}: "
            f"step 0 = {losses[0]:.2f}, "
            f"step 50 = {losses[49]:.2f}, "
            f"step 100 = {losses[99]:.2f}, "
            f"step 200 = {losses[-1]:.2f}"
        )
    print()

    # --- Summary ---
    r = benchmark_forward_backward(256, 1024, 512)
    m = benchmark_memory(256, 1024, 512)
    print("Summary (256×1024→512)")
    print("-" * 55)
    print(f"  Compute speedup:    {r['speedup']:.2f}×")
    print(f"  Memory savings:     {m['savings_ratio']}")
    print(f"  FP32 convergence:   {conv['fp32'][-1]:.4f}")
    print(f"  BF16 convergence:   {conv['bf16'][-1]:.4f}")
    print()


if __name__ == "__main__":
    main()
