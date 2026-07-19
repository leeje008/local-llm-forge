"""Automated evaluation pipeline for quantized models.

Integrates with lm-eval-harness and provides:
- Standard benchmark evaluation (MMLU, HumanEval, GSM8K, etc.)
- Quality-speed Pareto frontier visualization
- Quantization degradation analysis
- Per-task quality comparison across configurations
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Standard benchmark tasks
BENCHMARK_SUITES = {
    "quick": ["hellaswag", "arc_easy"],
    "standard": ["mmlu", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
    "reasoning": ["gsm8k", "arc_challenge", "winogrande"],
    "code": ["humaneval"],
    "full": [
        "mmlu", "hellaswag", "arc_easy", "arc_challenge",
        "winogrande", "gsm8k", "truthfulqa_mc2",
    ],
}


@dataclass
class EvalResult:
    """Result from a single benchmark evaluation."""

    model_path: str
    task: str
    metric: str = ""
    score: float = 0.0
    stderr: float = 0.0
    num_samples: int = 0


@dataclass
class EvalReport:
    """Complete evaluation report for a model."""

    model_path: str
    suite: str
    timestamp: str = ""
    results: list[EvalResult] = field(default_factory=list)
    avg_score: float = 0.0
    total_time_seconds: float = 0.0
    config: dict = field(default_factory=dict)
    error: str | None = None


def run_eval(
    model_path: str | Path,
    tasks: list[str] | None = None,
    suite: str = "quick",
    num_fewshot: int = 0,
    batch_size: int = 1,
    limit: int | None = None,
    output_path: Path | None = None,
) -> EvalReport:
    """Run evaluation benchmarks on a model using lm-eval-harness.

    Uses MLX backend for Apple Silicon optimized inference.
    """
    if tasks is None:
        tasks = BENCHMARK_SUITES.get(suite, BENCHMARK_SUITES["quick"])

    report = EvalReport(
        model_path=str(model_path),
        suite=suite,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    task_str = ",".join(tasks)

    # Try lm_eval CLI
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", task_str,
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", str(output_path or "/tmp/forge-eval"),
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
        )
        report.total_time_seconds = time.monotonic() - start

        if result.returncode != 0:
            # Try to parse partial results
            report.error = (result.stderr or result.stdout)[:500]

            # Fallback: try mlx-lm generate with simple prompts
            if "lm_eval" in report.error:
                return _fallback_eval(model_path, tasks, report)
            return report

        # Parse output JSON
        output_dir = Path(output_path or "/tmp/forge-eval")
        result_files = list(output_dir.rglob("results*.json"))
        if result_files:
            with open(sorted(result_files)[-1]) as f:
                raw = json.load(f)
            report = _parse_lm_eval_results(raw, report)

    except subprocess.TimeoutExpired:
        report.error = "Evaluation timed out (1h limit)"
    except FileNotFoundError:
        report.error = "lm-eval-harness not installed. Run: pip install lm-eval"
        return _fallback_eval(model_path, tasks, report)
    except Exception as e:
        report.error = str(e)

    return report


def _parse_lm_eval_results(raw: dict, report: EvalReport) -> EvalReport:
    """Parse lm-eval-harness JSON output."""
    results = raw.get("results", {})
    scores = []

    for task_name, metrics in results.items():
        # Find the primary metric (usually acc or acc_norm)
        for metric_key in ["acc_norm,none", "acc,none", "exact_match,strict-match"]:
            if metric_key in metrics:
                score = metrics[metric_key]
                stderr = metrics.get(f"{metric_key}_stderr", 0.0)
                report.results.append(EvalResult(
                    model_path=report.model_path,
                    task=task_name,
                    metric=metric_key.split(",")[0],
                    score=score,
                    stderr=stderr,
                ))
                scores.append(score)
                break

    if scores:
        report.avg_score = sum(scores) / len(scores)

    return report


def _fallback_eval(
    model_path: str | Path,
    tasks: list[str],
    report: EvalReport,
) -> EvalReport:
    """Fallback evaluation using direct MLX inference with simple prompts.

    When lm-eval-harness is not available, run basic quality checks.
    """
    report.error = None  # Clear previous error

    eval_prompts = {
        "math": ("What is 15 * 23?", "345"),
        "reasoning": (
            "If all cats are animals, and Whiskers is a cat, is Whiskers an animal?",
            "yes",
        ),
        "code": (
            "Write a Python function that returns the sum of a list: def sum_list(lst):",
            "return sum",
        ),
        "knowledge": ("What is the capital of France?", "Paris"),
        "logic": ("Complete: 2, 4, 8, 16, __", "32"),
    }

    try:
        import mlx_lm

        model, tokenizer = mlx_lm.load(str(model_path))

        correct = 0
        total = 0

        for name, (prompt, expected) in eval_prompts.items():
            output = mlx_lm.generate(
                model, tokenizer, prompt=prompt,
                max_tokens=50, verbose=False,
            )
            passed = expected.lower() in output.lower()
            report.results.append(EvalResult(
                model_path=report.model_path,
                task=f"basic_{name}",
                metric="contains_answer",
                score=1.0 if passed else 0.0,
            ))
            if passed:
                correct += 1
            total += 1

        report.avg_score = correct / total if total > 0 else 0.0
        report.suite = "basic_fallback"

    except Exception as e:
        report.error = f"Fallback eval also failed: {e}"

    return report


def compare_configs(
    reports: list[EvalReport],
    tps_values: list[float] | None = None,
) -> str:
    """Generate comparison table across multiple configurations."""
    if not reports:
        return "No reports to compare."

    # Collect all tasks
    all_tasks = set()
    for r in reports:
        for res in r.results:
            all_tasks.add(res.task)
    all_tasks = sorted(all_tasks)

    lines = ["Configuration Comparison", "=" * 70]

    # Header
    config_names = [Path(r.model_path).name for r in reports]
    header = f"  {'Task':<20}" + "".join(f" {n[:15]:>15}" for n in config_names)
    lines.append(header)
    lines.append("  " + "-" * (20 + 15 * len(config_names)))

    # Per-task scores
    for task in all_tasks:
        row = f"  {task:<20}"
        for r in reports:
            score = next((res.score for res in r.results if res.task == task), None)
            row += f" {score:>14.1%}" if score is not None else f" {'N/A':>15}"
        lines.append(row)

    # Average
    lines.append("  " + "-" * (20 + 15 * len(config_names)))
    avg_row = f"  {'AVERAGE':<20}"
    for r in reports:
        avg_row += f" {r.avg_score:>14.1%}"
    lines.append(avg_row)

    # Speed if provided
    if tps_values and len(tps_values) == len(reports):
        speed_row = f"  {'Speed (tok/s)':<20}"
        for tps in tps_values:
            speed_row += f" {tps:>14.1f}"
        lines.append(speed_row)

    return "\n".join(lines)


def format_eval_report(report: EvalReport) -> str:
    """Format evaluation report for display."""
    lines = [
        "Evaluation Report",
        "=" * 50,
        f"  Model:     {report.model_path}",
        f"  Suite:     {report.suite}",
        f"  Timestamp: {report.timestamp}",
        f"  Time:      {report.total_time_seconds:.1f}s",
        "",
    ]

    if report.error:
        lines.append(f"  Error: {report.error}")
        return "\n".join(lines)

    lines.append(f"  {'Task':<20} {'Metric':<15} {'Score':>10} {'StdErr':>10}")
    lines.append(f"  {'-'*20} {'-'*15} {'-'*10} {'-'*10}")

    for r in report.results:
        stderr_str = f"{r.stderr:.4f}" if r.stderr else ""
        lines.append(f"  {r.task:<20} {r.metric:<15} {r.score:>9.1%} {stderr_str:>10}")

    lines.append("")
    lines.append(f"  Average Score: {report.avg_score:.1%}")

    return "\n".join(lines)
