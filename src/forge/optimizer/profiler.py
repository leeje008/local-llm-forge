"""Profile-based automatic parameter tuning."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProfileMetrics:
    """Performance metrics from a profiling run."""

    ttft_seconds: float = 0.0  # Time to first token
    tps: float = 0.0  # Tokens per second (decode)
    peak_memory_gb: float = 0.0
    tokens_generated: int = 0
    total_time_seconds: float = 0.0
    oom: bool = False
    error: str | None = None


def quick_bench(
    model_path: str | Path,
    prompt: str = "Explain the concept of recursion in computer science.",
    max_tokens: int = 100,
    runtime: str = "mlx-lm",
) -> ProfileMetrics:
    """Run a quick benchmark to measure inference performance.

    Currently supports mlx-lm runtime.
    """
    if runtime != "mlx-lm":
        return ProfileMetrics(error=f"Profiling not yet supported for runtime: {runtime}")

    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", str(model_path),
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
    ]

    try:
        start = time.monotonic()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        elapsed = time.monotonic() - start

        if result.returncode != 0:
            stderr = result.stderr or ""
            if "out of memory" in stderr.lower() or "oom" in stderr.lower():
                return ProfileMetrics(oom=True, error="Out of memory")
            return ProfileMetrics(error=stderr[:300])

        # Parse output for timing info
        # mlx-lm typically prints: "Prompt: X tokens, Y.Z tok/s, Z.Z s"
        # and "Generation: X tokens, Y.Z tok/s, Z.Z s"
        metrics = ProfileMetrics(total_time_seconds=elapsed)
        output = result.stdout + result.stderr

        for line in output.splitlines():
            line_lower = line.lower()
            if "prompt" in line_lower and "tok/s" in line_lower:
                # Try to parse TTFT
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if part.endswith("s") and not part.endswith("tok/s"):
                        try:
                            metrics.ttft_seconds = float(part.rstrip("s").strip())
                        except ValueError:
                            pass
            elif "generation" in line_lower and "tok/s" in line_lower:
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if "tok/s" in part:
                        try:
                            metrics.tps = float(part.replace("tok/s", "").strip())
                        except ValueError:
                            pass
                    elif "token" in part:
                        try:
                            metrics.tokens_generated = int(
                                part.split()[0].strip()
                            )
                        except (ValueError, IndexError):
                            pass

        # Fallback: estimate TPS from elapsed time
        if metrics.tps == 0 and elapsed > 0 and max_tokens > 0:
            metrics.tps = max_tokens / elapsed
            metrics.tokens_generated = max_tokens

        return metrics

    except subprocess.TimeoutExpired:
        return ProfileMetrics(error="Benchmark timed out (5 min)")
    except Exception as e:
        return ProfileMetrics(error=str(e))


def auto_tune(
    model_path: str | Path,
    initial_context: int,
    usable_memory_gb: float,
    runtime: str = "mlx-lm",
) -> tuple[int, ProfileMetrics]:
    """Run a quick benchmark and auto-adjust context length.

    Returns (adjusted_context, metrics).
    """
    metrics = quick_bench(model_path, max_tokens=50, runtime=runtime)

    if metrics.oom:
        # Reduce context by half and retry
        new_context = max(initial_context // 2, 2048)
        if new_context < initial_context:
            return auto_tune(model_path, new_context, usable_memory_gb, runtime)
        return new_context, metrics

    if metrics.error:
        return initial_context, metrics

    # If memory usage is low, we could increase context
    # (Currently we can't measure memory directly, so keep as-is)
    return initial_context, metrics
