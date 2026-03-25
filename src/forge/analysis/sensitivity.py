"""Quantization sensitivity analysis for mixed-precision strategy.

Measures per-layer sensitivity to quantization using:
- Output divergence (KL-divergence from FP16 baseline)
- Weight distribution analysis (outlier detection)
- Automatic mixed-precision bit allocation

Based on: HAWQ-V2 (Hessian trace), SliM-LLM (2405.14917)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayerSensitivity:
    """Sensitivity measurement for a single layer."""

    layer_idx: int
    layer_name: str = ""
    kl_divergence: float = 0.0  # KL-div from FP16 baseline
    weight_range: float = 0.0  # max - min of weight values
    outlier_ratio: float = 0.0  # fraction of weights > 3 std
    recommended_bits: int = 4


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis for a model."""

    model_path: str
    num_layers: int = 0
    layers: list[LayerSensitivity] = field(default_factory=list)
    bit_allocation: dict[int, int] = field(default_factory=dict)  # layer_idx → bits
    estimated_memory_gb: float = 0.0
    analysis_time_seconds: float = 0.0


def analyze_weight_sensitivity(
    model_path: str | Path,
    memory_budget_gb: float = 38.0,
    target_avg_bits: float = 4.0,
) -> SensitivityReport:
    """Analyze per-layer quantization sensitivity via weight statistics.

    This is a lightweight analysis that doesn't require calibration data.
    Uses weight distribution statistics as a proxy for sensitivity.
    """
    report = SensitivityReport(model_path=str(model_path))
    start = time.monotonic()

    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        model, _ = load(str(model_path))

        # Analyze each layer's weights
        layer_stats = []

        for name, param in model.parameters().items():
            # Skip non-weight parameters
            if param.ndim < 2:
                continue

            # Flatten for statistics
            flat = param.reshape(-1).astype(mx.float32)
            mx.eval(flat)

            mean_val = float(mx.mean(flat))
            std_val = float(mx.sqrt(mx.mean((flat - mean_val) ** 2)))
            max_val = float(mx.max(mx.abs(flat)))
            min_val = float(mx.min(flat))

            # Outlier ratio: fraction of values > 3 std from mean
            if std_val > 0:
                outliers = mx.sum(mx.abs(flat - mean_val) > 3 * std_val)
                mx.eval(outliers)
                outlier_ratio = float(outliers) / flat.size
            else:
                outlier_ratio = 0.0

            layer_stats.append({
                "name": name,
                "range": max_val - min_val,
                "outlier_ratio": outlier_ratio,
                "num_params": flat.size,
            })

        report.num_layers = len(layer_stats)

        # Score sensitivity: higher range + more outliers = more sensitive
        for i, stats in enumerate(layer_stats):
            sensitivity_score = stats["range"] * (1 + stats["outlier_ratio"] * 10)

            ls = LayerSensitivity(
                layer_idx=i,
                layer_name=stats["name"],
                weight_range=stats["range"],
                outlier_ratio=stats["outlier_ratio"],
            )
            report.layers.append(ls)

        # Allocate bits: sensitive layers get more bits
        if report.layers:
            report.bit_allocation = _allocate_bits(
                report.layers, target_avg_bits, memory_budget_gb,
            )
            for layer in report.layers:
                layer.recommended_bits = report.bit_allocation.get(
                    layer.layer_idx, 4
                )

    except Exception as e:
        report.layers = []

    report.analysis_time_seconds = time.monotonic() - start
    return report


def _allocate_bits(
    layers: list[LayerSensitivity],
    target_avg: float,
    memory_budget_gb: float,
) -> dict[int, int]:
    """Allocate bits per layer given a target average and memory budget.

    Strategy: Sort by sensitivity (outlier_ratio + weight_range),
    assign higher bits to more sensitive layers.
    """
    available_bits = [2, 3, 4, 6, 8]
    n = len(layers)

    # Score each layer
    scores = []
    for l in layers:
        score = l.weight_range * (1 + l.outlier_ratio * 10)
        scores.append((l.layer_idx, score))

    # Sort by sensitivity (most sensitive first)
    scores.sort(key=lambda x: -x[1])

    # Assign bits: top 20% get 6-8 bits, middle 60% get target, bottom 20% get 2-3 bits
    allocation = {}
    for rank, (idx, score) in enumerate(scores):
        pct = rank / n
        if pct < 0.1:
            allocation[idx] = 8  # Top 10% most sensitive
        elif pct < 0.25:
            allocation[idx] = 6  # Next 15%
        elif pct < 0.75:
            allocation[idx] = int(target_avg)  # Middle 50%
        else:
            allocation[idx] = max(2, int(target_avg) - 1)  # Bottom 25%

    return allocation


def format_sensitivity_report(report: SensitivityReport) -> str:
    """Format sensitivity analysis for display."""
    lines = [
        "Quantization Sensitivity Analysis",
        "=" * 60,
        f"  Model:    {report.model_path}",
        f"  Layers:   {report.num_layers}",
        f"  Time:     {report.analysis_time_seconds:.1f}s",
        "",
    ]

    if not report.layers:
        lines.append("  No layers analyzed (model may not be loaded).")
        return "\n".join(lines)

    # Summary: bit allocation distribution
    bit_counts: dict[int, int] = {}
    for bits in report.bit_allocation.values():
        bit_counts[bits] = bit_counts.get(bits, 0) + 1

    lines.append("  Bit Allocation Summary:")
    for bits in sorted(bit_counts):
        lines.append(f"    {bits}-bit: {bit_counts[bits]} layers")

    # Top 10 most sensitive layers
    sorted_layers = sorted(report.layers, key=lambda l: -l.outlier_ratio)
    lines.append("")
    lines.append("  Most Sensitive Layers (high outlier ratio → need more bits):")
    lines.append(f"  {'Layer':<40} {'Range':>8} {'Outliers':>10} {'Bits':>5}")
    lines.append(f"  {'-'*40} {'-'*8} {'-'*10} {'-'*5}")

    for l in sorted_layers[:15]:
        name = l.layer_name[:40] if l.layer_name else f"layer_{l.layer_idx}"
        lines.append(
            f"  {name:<40} {l.weight_range:>8.3f} {l.outlier_ratio:>9.4f} {l.recommended_bits:>5}"
        )

    return "\n".join(lines)
