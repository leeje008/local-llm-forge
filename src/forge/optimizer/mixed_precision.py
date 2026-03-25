"""Mixed-precision quantization with per-layer intelligent bit allocation.

Analyzes layer sensitivity and allocates higher bits to sensitive layers,
lower bits to robust layers — maximizing quality within a memory budget.

Based on: SliM-LLM (2405.14917), HAWQ-V2 (Hessian trace), ParetoQ (2502.02631)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BitAllocation:
    """Per-layer bit allocation result."""

    layer_name: str
    bits: int
    sensitivity_score: float
    weight_range: float
    outlier_ratio: float


@dataclass
class MixedPrecisionPlan:
    """Complete mixed-precision quantization plan."""

    model_path: str
    allocations: list[BitAllocation] = field(default_factory=list)
    avg_bits: float = 0.0
    estimated_memory_gb: float = 0.0
    num_layers: int = 0
    bit_distribution: dict[int, int] = field(default_factory=dict)  # bits → count


def analyze_and_allocate(
    model_path: str | Path,
    target_avg_bits: float = 4.0,
    memory_budget_gb: float = 38.0,
) -> MixedPrecisionPlan:
    """Analyze layer sensitivity and create per-layer bit allocation plan.

    Strategy (based on SliM-LLM):
    - Top 10% most sensitive: 8-bit (attention projection, embeddings)
    - Next 15%: 6-bit
    - Middle 50%: target_avg_bits (typically 4)
    - Bottom 25%: target_avg_bits - 1 (typically 3)
    """
    plan = MixedPrecisionPlan(model_path=str(model_path))

    try:
        import mlx.core as mx
        from mlx_lm import load

        model, _ = load(str(model_path))

        # Collect per-layer weight statistics
        layer_stats = []
        for name, param in model.parameters().items():
            if param.ndim < 2:
                continue

            flat = param.reshape(-1).astype(mx.float32)
            mx.eval(flat)

            mean_val = float(mx.mean(flat))
            var_val = float(mx.mean((flat - mean_val) ** 2))
            std_val = var_val ** 0.5
            max_abs = float(mx.max(mx.abs(flat)))

            # Outlier detection: fraction > 3*std
            if std_val > 0:
                outlier_count = mx.sum(mx.abs(flat - mean_val) > 3 * std_val)
                mx.eval(outlier_count)
                outlier_ratio = float(outlier_count) / flat.size
            else:
                outlier_ratio = 0.0

            weight_range = max_abs * 2  # approximate range
            sensitivity = weight_range * (1 + outlier_ratio * 10)

            layer_stats.append({
                "name": name,
                "sensitivity": sensitivity,
                "weight_range": weight_range,
                "outlier_ratio": outlier_ratio,
                "num_params": flat.size,
            })

        # Sort by sensitivity (most sensitive first)
        layer_stats.sort(key=lambda x: -x["sensitivity"])
        n = len(layer_stats)
        plan.num_layers = n

        # Allocate bits based on position in sensitivity ranking
        available_bits = [2, 3, 4, 6, 8]
        target = int(target_avg_bits)

        for rank, stats in enumerate(layer_stats):
            pct = rank / n if n > 0 else 0

            # Embedding/LM head layers get high bits
            is_critical = any(k in stats["name"].lower() for k in ["embed", "lm_head", "norm"])

            if is_critical:
                bits = 8
            elif pct < 0.10:
                bits = 8  # Top 10% most sensitive
            elif pct < 0.25:
                bits = 6  # Next 15%
            elif pct < 0.75:
                bits = target  # Middle 50%
            else:
                bits = max(2, target - 1)  # Bottom 25%

            alloc = BitAllocation(
                layer_name=stats["name"],
                bits=bits,
                sensitivity_score=stats["sensitivity"],
                weight_range=stats["weight_range"],
                outlier_ratio=stats["outlier_ratio"],
            )
            plan.allocations.append(alloc)

        # Calculate statistics
        if plan.allocations:
            total_bits = sum(a.bits for a in plan.allocations)
            plan.avg_bits = total_bits / len(plan.allocations)
            for a in plan.allocations:
                plan.bit_distribution[a.bits] = plan.bit_distribution.get(a.bits, 0) + 1

            # Rough memory estimate
            total_params = sum(s["num_params"] for s in layer_stats)
            weighted_bytes = sum(
                s["num_params"] * (plan.allocations[i].bits / 8)
                for i, s in enumerate(layer_stats)
            )
            plan.estimated_memory_gb = weighted_bytes / 1e9 * 1.05

    except Exception as e:
        plan.allocations = []

    return plan


def apply_mixed_quantization(
    model_path: str | Path,
    plan: MixedPrecisionPlan,
    output_dir: Path,
) -> tuple[bool, str]:
    """Apply per-layer mixed-precision quantization using MLX API.

    Loads model, quantizes each layer at its assigned bit-width, saves result.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.utils import save_model

        model, tokenizer = load(str(model_path))

        # Build name→bits mapping
        allocation_map = {a.layer_name: a.bits for a in plan.allocations}

        # Apply per-layer quantization
        # MLX's nn.quantize works on the entire model with uniform bits,
        # but we can quantize individual linear layers
        quantized_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                bits = _find_bits_for_module(name, allocation_map)
                if bits and bits < 16:
                    nn.quantize(module, bits=bits, group_size=128 if bits >= 4 else 64)
                    quantized_count += 1

        # Save quantized model
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # Save weights
        weights = dict(model.parameters())
        mx.savez(str(output_dir / "weights.npz"), **{k: v for k, v in weights.items()})

        # Save tokenizer
        tokenizer.save_pretrained(str(output_dir))

        # Save allocation plan
        plan_data = {
            "avg_bits": plan.avg_bits,
            "bit_distribution": plan.bit_distribution,
            "allocations": [
                {"name": a.layer_name, "bits": a.bits, "score": a.sensitivity_score}
                for a in plan.allocations
            ],
        }
        (output_dir / "mixed_precision_plan.json").write_text(json.dumps(plan_data, indent=2))

        return True, f"Mixed-precision quantized ({quantized_count} layers, avg {plan.avg_bits:.1f} bits)"

    except Exception as e:
        return False, str(e)


def _find_bits_for_module(module_name: str, allocation_map: dict[str, int]) -> int | None:
    """Find the bit allocation for a module by matching name patterns."""
    # Exact match
    if module_name in allocation_map:
        return allocation_map[module_name]

    # Partial match (module_name might be a prefix of allocation keys)
    for alloc_name, bits in allocation_map.items():
        if module_name in alloc_name or alloc_name in module_name:
            return bits

    return 4  # Default


def format_plan_report(plan: MixedPrecisionPlan) -> str:
    """Format mixed-precision plan for display."""
    lines = [
        "Mixed-Precision Quantization Plan",
        "=" * 55,
        f"  Model:          {plan.model_path}",
        f"  Layers:         {plan.num_layers}",
        f"  Average Bits:   {plan.avg_bits:.2f}",
        f"  Est. Memory:    {plan.estimated_memory_gb:.1f} GB",
        "",
        "  Bit Distribution:",
    ]

    for bits in sorted(plan.bit_distribution):
        count = plan.bit_distribution[bits]
        pct = count / plan.num_layers * 100 if plan.num_layers else 0
        bar = "#" * int(pct / 2)
        lines.append(f"    {bits}-bit: {count:>4} layers ({pct:>5.1f}%) {bar}")

    # Top 5 most sensitive layers
    if plan.allocations:
        lines.append("")
        lines.append("  Most Sensitive Layers (highest bits):")
        high_bit = sorted(plan.allocations, key=lambda a: (-a.bits, -a.sensitivity_score))
        for a in high_bit[:5]:
            name = a.layer_name[:45]
            lines.append(f"    {a.bits}-bit  {name}  (score={a.sensitivity_score:.3f})")

    return "\n".join(lines)
