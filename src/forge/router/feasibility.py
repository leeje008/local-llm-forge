"""5-stage model feasibility routing decision tree.

Given a model and hardware profile, determines the best execution path:
  1. FULL_PRECISION — model fits in FP16
  2. QUANTIZED      — model fits with quantization (int8→int4→int3→int2)
  3. OFFLOADED      — model fits with partial GPU offload (llama.cpp -ngl)
  4. DOWNSIZE       — recommend smaller model in same family
  5. CLOUD_FALLBACK — recommend cloud API
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.memory_calculator import (
    MemoryBudget,
    calc_theoretical_tps,
    calc_weight_memory,
)
from forge.analyzer.model_inspector import ModelProfile
from forge.router.alternatives import find_alternatives
from forge.router.offload import estimate_offload


class Route:
    FULL_PRECISION = "FULL_PRECISION"
    QUANTIZED = "QUANTIZED"
    OFFLOADED = "OFFLOADED"
    DOWNSIZE = "DOWNSIZE"
    CLOUD_FALLBACK = "CLOUD_FALLBACK"


@dataclass
class RoutePlan:
    """A single execution path option."""

    route: str
    feasible: bool
    label: str = ""
    quant: str = ""
    estimated_tps: float = 0.0
    estimated_memory_gb: float = 0.0
    context_length: int = 0
    command: str = ""  # CLI command to execute this route
    notes: str = ""


@dataclass
class RouteDecision:
    """Complete routing analysis for a model."""

    model_id: str
    model_params_b: float
    available_memory_gb: float
    recommended: RoutePlan | None = None
    all_routes: list[RoutePlan] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)


def route(
    model: ModelProfile,
    hardware: HardwareProfile,
    budget: MemoryBudget,
) -> RouteDecision:
    """Run the 5-stage routing decision tree."""
    decision = RouteDecision(
        model_id=model.model_id,
        model_params_b=model.total_params_b,
        available_memory_gb=hardware.usable_memory_gb,
    )
    # ── Stage 1: Full Precision (FP16) ──
    fp16_est = _find_estimate(budget, "fp16")
    if fp16_est and fp16_est.fits_with_kv:
        plan = RoutePlan(
            route=Route.FULL_PRECISION,
            feasible=True,
            label="Full Precision (FP16)",
            quant="fp16",
            estimated_tps=calc_theoretical_tps(model, hardware, "fp16"),
            estimated_memory_gb=fp16_est.total_memory_gb,
            context_length=min(fp16_est.max_context, 8192),
            command=f"forge optimize {model.model_id}",
        )
        decision.all_routes.append(plan)
        if decision.recommended is None:
            decision.recommended = plan
            decision.reasoning.append("Model fits in FP16 — best quality")
    else:
        fp16_mem = calc_weight_memory(model, "fp16")
        decision.all_routes.append(RoutePlan(
            route=Route.FULL_PRECISION, feasible=False,
            label="Full Precision (FP16)",
            quant="fp16", estimated_memory_gb=fp16_mem,
            notes=f"{fp16_mem:.0f}GB needed, {hardware.usable_memory_gb:.0f}GB available",
        ))

    # ── Stage 2: Quantized ──
    for quant in ["int8", "int4", "int3", "int2"]:
        est = _find_estimate(budget, quant)
        if est and est.fits_with_kv:
            bits = int(quant.replace("int", ""))
            plan = RoutePlan(
                route=Route.QUANTIZED,
                feasible=True,
                label=f"Quantized ({quant})",
                quant=quant,
                estimated_tps=calc_theoretical_tps(model, hardware, quant),
                estimated_memory_gb=est.total_memory_gb,
                context_length=min(est.max_context, 8192),
                command=f"forge optimize {model.model_id} --bits {bits}",
                notes=f"{est.quality_pct:.0f}% quality",
            )
            decision.all_routes.append(plan)
            if decision.recommended is None:
                decision.recommended = plan
                decision.reasoning.append(
                    f"Model fits with {quant} quantization ({est.quality_pct:.0f}% quality)"
                )
            break  # Only add the best quantization that fits
        elif est:
            decision.all_routes.append(RoutePlan(
                route=Route.QUANTIZED, feasible=False,
                label=f"Quantized ({quant})",
                quant=quant, estimated_memory_gb=est.weight_memory_gb,
                notes=f"{est.weight_memory_gb:.0f}GB weights alone",
            ))

    # ── Stage 3: Offloaded (partial GPU via llama.cpp) ──
    offload = estimate_offload(model, hardware)
    if offload["feasible"]:
        plan = RoutePlan(
            route=Route.OFFLOADED,
            feasible=True,
            label=f"Disk Offload ({offload['gpu_layers']}/{model.num_layers} GPU layers)",
            quant="int4",
            estimated_tps=offload["estimated_tps"],
            estimated_memory_gb=offload["gpu_memory_gb"],
            context_length=offload["context_length"],
            command=offload["command"],
            notes=f"{offload['gpu_pct']:.0f}% on GPU, rest on CPU/disk",
        )
        decision.all_routes.append(plan)
        if decision.recommended is None:
            decision.recommended = plan
            decision.reasoning.append(
                f"Model fits with partial offload ({offload['gpu_pct']:.0f}% GPU)"
            )
    else:
        decision.all_routes.append(RoutePlan(
            route=Route.OFFLOADED, feasible=False,
            label="Disk Offload",
            notes=offload.get("reason", "Not feasible"),
        ))

    # ── Stage 4: Downsize (smaller model alternatives) ──
    alts = find_alternatives(model, hardware)
    if alts:
        decision.alternatives = alts
        best_alt = alts[0]
        plan = RoutePlan(
            route=Route.DOWNSIZE,
            feasible=True,
            label=f"Smaller Alternative: {best_alt['name']}",
            quant=best_alt.get("quant", "int4"),
            estimated_tps=best_alt.get("estimated_tps", 0),
            command=f"forge optimize {best_alt['model_id']}",
            notes=best_alt.get("reason", ""),
        )
        decision.all_routes.append(plan)
        if decision.recommended is None:
            decision.recommended = plan
            decision.reasoning.append(f"Original model too large — recommending {best_alt['name']}")

    # ── Stage 5: Cloud Fallback ──
    plan = RoutePlan(
        route=Route.CLOUD_FALLBACK,
        feasible=True,
        label="Cloud API Fallback",
        command="# Use OpenAI/Anthropic/OpenRouter API",
        notes="No local execution possible at acceptable quality",
    )
    decision.all_routes.append(plan)
    if decision.recommended is None:
        decision.recommended = plan
        decision.reasoning.append("Model cannot run locally — cloud API recommended")

    return decision


def _find_estimate(budget: MemoryBudget, quant: str):
    for est in budget.estimates:
        if est.quant == quant:
            return est
    return None


def format_route_report(decision: RouteDecision) -> str:
    """Format routing analysis for display."""
    lines = [
        "Model Routing Analysis",
        "=" * 60,
        f"  Model:     {decision.model_id}",
        f"  Params:    {decision.model_params_b:.1f}B",
        f"  Memory:    {decision.available_memory_gb:.0f}GB available",
        "",
    ]

    if decision.recommended:
        lines.append(f"  Recommended: {decision.recommended.label}")
        if decision.recommended.command:
            lines.append(f"  Command:     {decision.recommended.command}")
        lines.append("")

    lines.append(f"  {'#':<4} {'Route':<35} {'Status':<6} {'Speed':>10} {'Memory':>10} {'Notes'}")
    lines.append(f"  {'─'*4} {'─'*35} {'─'*6} {'─'*10} {'─'*10} {'─'*30}")

    for i, r in enumerate(decision.all_routes, 1):
        status = "✓" if r.feasible else "✗"
        tps = f"~{r.estimated_tps:.0f} tok/s" if r.estimated_tps > 0 else ""
        mem = f"~{r.estimated_memory_gb:.0f}GB" if r.estimated_memory_gb > 0 else ""
        is_rec = (
            " ←"
            if decision.recommended and r.route == decision.recommended.route and r.feasible
            else ""
        )
        lines.append(f"  [{i}] {r.label:<35} {status:<6} {tps:>10} {mem:>10} {r.notes}{is_rec}")
        if r.feasible and r.command:
            lines.append(f"      → {r.command}")

    if decision.alternatives:
        lines.append("")
        lines.append("  Model Alternatives:")
        for alt in decision.alternatives[:3]:
            lines.append(f"    - {alt['name']}: {alt['reason']}")
            if alt.get("command"):
                lines.append(f"      → {alt['command']}")

    if decision.reasoning:
        lines.append("")
        lines.append("  Reasoning:")
        for r in decision.reasoning:
            lines.append(f"    - {r}")

    return "\n".join(lines)
