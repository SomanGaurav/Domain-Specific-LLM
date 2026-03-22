"""Latency and memory profiling helpers for inference-time comparison."""

import time
from contextlib import contextmanager

import torch


class LatencyTracker:
    """Collects per-query wall-clock latencies and summarizes them."""

    def __init__(self) -> None:
        self.latencies_s: list[float] = []

    @contextmanager
    def track(self):
        start = time.perf_counter()
        yield
        self.latencies_s.append(time.perf_counter() - start)

    def summary(self) -> dict[str, float]:
        xs = sorted(self.latencies_s)
        n = len(xs)
        return {
            "latency_mean_s": sum(xs) / n,
            "latency_p50_s": xs[n // 2],
            "latency_p95_s": xs[min(n - 1, int(n * 0.95))],
        }


def gpu_memory_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
