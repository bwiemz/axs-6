"""
AXS-6 Backend Acceleration Benchmark
=======================================

Compares eager vs compiled vs INT8 backends on:
  1. Fake-quantize latency (the dominant cost)
  2. Linear layer throughput (forward pass)
  3. Full training step (forward + backward + optimizer)

Run:
    py benchmarks/benchmark_backend.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from axs.unified.backend import (
    BackendType,
    accelerated_fake_quantize,
    accelerated_linear,
    backend_info,
    get_backend,
    int8_linear,
    set_backend,
)
from axs.unified.quantize_unified import fused_fake_quantize
from axs.unified.modules_unified import AXSLinearUnified, convert_to_axs_unified
from axs.core import DEFAULT_BLOCK_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 10
TRIALS = 50


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def bench_fake_quantize() -> None:
    """Benchmark fake-quantize across backends."""
    print("\n" + "=" * 70)
    print("  FAKE-QUANTIZE LATENCY  (4096 × 4096, block_size=32)")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(4096, 4096, device=DEVICE)

    results: dict[str, float] = {}

    # --- Eager ---
    set_backend("eager")
    for _ in range(WARMUP):
        fused_fake_quantize(x, DEFAULT_BLOCK_SIZE, "nearest")
    _sync()

    t0 = time.perf_counter()
    for _ in range(TRIALS):
        fused_fake_quantize(x, DEFAULT_BLOCK_SIZE, "nearest")
    _sync()
    eager_ms = (time.perf_counter() - t0) / TRIALS * 1000
    results["Eager"] = eager_ms

    # --- Compiled ---
    set_backend("compiled")
    # Extra warmup for compilation
    for _ in range(WARMUP + 5):
        accelerated_fake_quantize(x, DEFAULT_BLOCK_SIZE, "nearest")
    _sync()

    t0 = time.perf_counter()
    for _ in range(TRIALS):
        accelerated_fake_quantize(x, DEFAULT_BLOCK_SIZE, "nearest")
    _sync()
    compiled_ms = (time.perf_counter() - t0) / TRIALS * 1000
    results["Compiled"] = compiled_ms

    # Print
    for name, ms in results.items():
        speedup = eager_ms / ms if ms > 0 else 0
        print(f"  {name:12s}: {ms:8.3f} ms  ({speedup:5.2f}× vs eager)")


def bench_linear() -> None:
    """Benchmark linear forward pass across backends."""
    print("\n" + "=" * 70)
    print("  LINEAR FORWARD PASS  (batch=128, in=1024, out=1024)")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(128, 1024, device=DEVICE)
    w = torch.randn(1024, 1024, device=DEVICE)
    b = torch.randn(1024, device=DEVICE)

    results: dict[str, float] = {}

    # --- Eager ---
    set_backend("eager")
    for _ in range(WARMUP):
        accelerated_linear(x, w, b)
    _sync()

    t0 = time.perf_counter()
    for _ in range(TRIALS):
        accelerated_linear(x, w, b)
    _sync()
    eager_ms = (time.perf_counter() - t0) / TRIALS * 1000
    results["Eager"] = eager_ms

    # --- Compiled ---
    set_backend("compiled")
    for _ in range(WARMUP + 5):
        accelerated_linear(x, w, b)
    _sync()

    t0 = time.perf_counter()
    for _ in range(TRIALS):
        accelerated_linear(x, w, b)
    _sync()
    compiled_ms = (time.perf_counter() - t0) / TRIALS * 1000
    results["Compiled"] = compiled_ms

    # --- INT8 ---
    if DEVICE == "cuda":
        for _ in range(WARMUP):
            int8_linear(x, w, b)
        _sync()

        t0 = time.perf_counter()
        for _ in range(TRIALS):
            int8_linear(x, w, b)
        _sync()
        int8_ms = (time.perf_counter() - t0) / TRIALS * 1000
        results["INT8 TC"] = int8_ms

    for name, ms in results.items():
        speedup = eager_ms / ms if ms > 0 else 0
        print(f"  {name:12s}: {ms:8.3f} ms  ({speedup:5.2f}× vs eager)")


def bench_training_step() -> None:
    """Benchmark full training step (fwd + bwd + optim)."""
    print("\n" + "=" * 70)
    print("  FULL TRAINING STEP  (2-layer MLP, 512→512→10, batch=64)")
    print("=" * 70)

    results: dict[str, float] = {}

    for backend_name in ["eager", "compiled"]:
        set_backend(backend_name)

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        ).to(DEVICE)
        model = convert_to_axs_unified(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        x = torch.randn(64, 512, device=DEVICE)
        y = torch.randint(0, 10, (64,), device=DEVICE)

        # Warmup
        for _ in range(WARMUP + (5 if backend_name == "compiled" else 0)):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        _sync()

        t0 = time.perf_counter()
        for _ in range(TRIALS):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        _sync()
        ms = (time.perf_counter() - t0) / TRIALS * 1000
        results[backend_name.capitalize()] = ms

    eager_ms = results.get("Eager", 1.0)
    for name, ms in results.items():
        speedup = eager_ms / ms if ms > 0 else 0
        print(f"  {name:12s}: {ms:8.3f} ms/step  ({speedup:5.2f}× vs eager)")


def main() -> None:
    info = backend_info()
    print("AXS-6 Backend Benchmark")
    print("-" * 40)
    print(f"  GPU: {info['gpu_name']}")
    print(f"  Compute Cap: {info['compute_capability']}")
    print(f"  torch.compile: {info['torch_compile']}")
    print(f"  INT8 TC: {info['int8_tensorcore']}")
    print(f"  Device: {DEVICE}")

    bench_fake_quantize()
    bench_linear()
    bench_training_step()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
