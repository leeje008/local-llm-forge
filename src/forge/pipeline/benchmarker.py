"""Standardized benchmarking suite for optimized models."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge.optimizer.profiler import ProfileMetrics, quick_bench


# Standard benchmark prompts of varying complexity
BENCH_PROMPTS = {
    "short": "What is 2 + 2?",
    "medium": "Explain the concept of recursion in computer science with a simple example.",
    "long": (
        "Write a detailed comparison of merge sort and quick sort algorithms. "
        "Include time complexity analysis for best, average, and worst cases. "
        "Provide pseudocode for both algorithms and discuss when to prefer one over the other."
    ),
    "code": "Write a Python function that implements binary search on a sorted list.",
}


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a model."""

    model_path: str
    timestamp: str = ""
    prompts_tested: int = 0
    avg_ttft_seconds: float = 0.0
    avg_tps: float = 0.0
    min_tps: float = 0.0
    max_tps: float = 0.0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_benchmark(
    model_path: str | Path,
    max_tokens: int = 100,
    runtime: str = "mlx-lm",
    prompts: dict[str, str] | None = None,
) -> BenchmarkResult:
    """Run a standardized benchmark suite on a model."""
    if prompts is None:
        prompts = BENCH_PROMPTS

    bench = BenchmarkResult(
        model_path=str(model_path),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    tps_values: list[float] = []
    ttft_values: list[float] = []
    start_total = time.monotonic()

    for name, prompt in prompts.items():
        metrics = quick_bench(
            model_path=model_path,
            prompt=prompt,
            max_tokens=max_tokens,
            runtime=runtime,
        )

        result_entry = {
            "name": name,
            "prompt_length": len(prompt),
            "tps": metrics.tps,
            "ttft": metrics.ttft_seconds,
            "tokens": metrics.tokens_generated,
            "error": metrics.error,
        }
        bench.results.append(result_entry)

        if metrics.error:
            bench.errors.append(f"{name}: {metrics.error}")
        else:
            if metrics.tps > 0:
                tps_values.append(metrics.tps)
            if metrics.ttft_seconds > 0:
                ttft_values.append(metrics.ttft_seconds)
            bench.total_tokens += metrics.tokens_generated

    bench.total_time_seconds = time.monotonic() - start_total
    bench.prompts_tested = len(prompts)

    if tps_values:
        bench.avg_tps = sum(tps_values) / len(tps_values)
        bench.min_tps = min(tps_values)
        bench.max_tps = max(tps_values)
    if ttft_values:
        bench.avg_ttft_seconds = sum(ttft_values) / len(ttft_values)

    return bench


def save_results(bench: BenchmarkResult, output_path: Path) -> Path:
    """Save benchmark results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(bench), indent=2))
    return output_path


def format_report(bench: BenchmarkResult) -> str:
    """Format benchmark results for display."""
    lines = [
        "Benchmark Results",
        "=" * 50,
        f"  Model:          {bench.model_path}",
        f"  Timestamp:      {bench.timestamp}",
        f"  Prompts Tested: {bench.prompts_tested}",
        "",
        f"  Avg TPS:        {bench.avg_tps:.1f} tok/s",
        f"  Min TPS:        {bench.min_tps:.1f} tok/s",
        f"  Max TPS:        {bench.max_tps:.1f} tok/s",
        f"  Avg TTFT:       {bench.avg_ttft_seconds:.2f}s",
        f"  Total Tokens:   {bench.total_tokens}",
        f"  Total Time:     {bench.total_time_seconds:.1f}s",
    ]

    if bench.results:
        lines.append("")
        lines.append(f"  {'Prompt':<10} {'TPS':>8} {'TTFT':>8} {'Tokens':>8}")
        lines.append(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for r in bench.results:
            if r.get("error"):
                lines.append(f"  {r['name']:<10} {'ERROR':>8}")
            else:
                lines.append(
                    f"  {r['name']:<10} {r['tps']:>7.1f} {r['ttft']:>7.2f}s {r['tokens']:>8}"
                )

    if bench.errors:
        lines.append("")
        lines.append("  Errors:")
        for e in bench.errors:
            lines.append(f"    - {e}")

    return "\n".join(lines)
