"""Token-level latency profiling for LLM inference.

Decomposes inference time into:
- TTFT (Time to First Token) — prefill latency
- TPOT (Time Per Output Token) — decode latency
- ITL (Inter-Token Latency) — per-token variance

Supports statistical analysis: distribution, tail latency, regression.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TokenLatency:
    """Latency measurement for a single token."""

    token_idx: int
    latency_ms: float
    cumulative_ms: float
    is_first: bool = False


@dataclass
class LatencyProfile:
    """Complete latency profile for a generation run."""

    model_path: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0  # Time to first token
    avg_tpot_ms: float = 0.0  # Average time per output token
    p50_tpot_ms: float = 0.0  # Median TPOT
    p90_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    min_tpot_ms: float = 0.0
    max_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    total_ms: float = 0.0
    tps: float = 0.0
    token_latencies: list[TokenLatency] = field(default_factory=list)
    # Tail latency analysis
    tail_latency_tokens: list[int] = field(default_factory=list)  # Token indices with p99+ latency


def profile_generation(
    model_path: str | Path,
    prompt: str = "Explain the concept of recursion in computer science with examples.",
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> LatencyProfile:
    """Profile token-level latency during generation.

    Uses MLX engine directly for fine-grained timing.
    """
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    profile = LatencyProfile(model_path=str(model_path))

    # Load model
    model, tokenizer = mlx_lm.load(str(model_path))
    sampler = make_sampler(temp=temperature)

    # Generate with per-token timing
    start_total = time.perf_counter()
    prev_time = start_total
    token_times: list[float] = []

    for i, response in enumerate(mlx_lm.stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )):
        now = time.perf_counter()
        latency_ms = (now - prev_time) * 1000

        tl = TokenLatency(
            token_idx=i,
            latency_ms=latency_ms,
            cumulative_ms=(now - start_total) * 1000,
            is_first=(i == 0),
        )
        profile.token_latencies.append(tl)

        if i == 0:
            profile.ttft_ms = latency_ms
        else:
            token_times.append(latency_ms)

        prev_time = now

    end_total = time.perf_counter()
    profile.total_ms = (end_total - start_total) * 1000
    profile.generated_tokens = len(profile.token_latencies)

    # Calculate statistics (excluding first token / TTFT)
    if token_times:
        token_times_sorted = sorted(token_times)
        n = len(token_times_sorted)

        profile.avg_tpot_ms = statistics.mean(token_times)
        profile.p50_tpot_ms = token_times_sorted[n // 2]
        profile.p90_tpot_ms = token_times_sorted[int(n * 0.9)]
        profile.p99_tpot_ms = token_times_sorted[int(n * 0.99)]
        profile.min_tpot_ms = token_times_sorted[0]
        profile.max_tpot_ms = token_times_sorted[-1]
        profile.std_tpot_ms = statistics.stdev(token_times) if n > 1 else 0

        # Identify tail latency tokens (> p99)
        p99_threshold = profile.p99_tpot_ms
        profile.tail_latency_tokens = [
            tl.token_idx for tl in profile.token_latencies
            if not tl.is_first and tl.latency_ms > p99_threshold
        ]

    if profile.total_ms > 0:
        profile.tps = profile.generated_tokens / (profile.total_ms / 1000)

    return profile


def format_latency_report(p: LatencyProfile) -> str:
    """Format latency profile for display."""
    lines = [
        "Latency Profile",
        "=" * 50,
        f"  Model:          {p.model_path}",
        f"  Tokens:         {p.generated_tokens}",
        f"  Total Time:     {p.total_ms:.0f}ms ({p.tps:.1f} tok/s)",
        "",
        "  Latency Breakdown:",
        f"    TTFT:         {p.ttft_ms:.1f}ms",
        f"    Avg TPOT:     {p.avg_tpot_ms:.1f}ms",
        f"    P50 TPOT:     {p.p50_tpot_ms:.1f}ms",
        f"    P90 TPOT:     {p.p90_tpot_ms:.1f}ms",
        f"    P99 TPOT:     {p.p99_tpot_ms:.1f}ms",
        f"    Min TPOT:     {p.min_tpot_ms:.1f}ms",
        f"    Max TPOT:     {p.max_tpot_ms:.1f}ms",
        f"    StdDev:       {p.std_tpot_ms:.1f}ms",
    ]

    if p.tail_latency_tokens:
        lines.append("")
        lines.append(
            f"  Tail Latency Tokens (>{p.p99_tpot_ms:.0f}ms): {len(p.tail_latency_tokens)}"
        )
        lines.append(f"    Positions: {p.tail_latency_tokens[:10]}")

    # Latency distribution histogram (text-based)
    if p.token_latencies and len(p.token_latencies) > 1:
        lines.append("")
        lines.append("  TPOT Distribution:")
        latencies = [tl.latency_ms for tl in p.token_latencies if not tl.is_first]
        if latencies:
            _append_histogram(lines, latencies)

    return "\n".join(lines)


def _append_histogram(lines: list[str], values: list[float], bins: int = 10) -> None:
    """Append a text histogram to lines."""
    if not values:
        return
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        lines.append(f"    All values: {min_v:.1f}ms")
        return

    bin_width = (max_v - min_v) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - min_v) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts) if counts else 1
    for i in range(bins):
        lo = min_v + i * bin_width
        hi = lo + bin_width
        bar_len = int(counts[i] / max_count * 30) if max_count > 0 else 0
        bar = "#" * bar_len
        lines.append(f"    {lo:>6.1f}-{hi:>6.1f}ms [{counts[i]:>4}] {bar}")
