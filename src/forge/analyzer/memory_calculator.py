"""Memory budget calculation for LLM deployment."""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.model_inspector import ModelProfile


# Bytes per parameter for each quantization level
BITS_TO_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "int3": 0.375,
    "int2": 0.25,
}

QUANT_LEVELS = ["fp16", "int8", "int4", "int3", "int2"]

# Quality impact estimate (% of FP16 quality retained)
QUALITY_ESTIMATE: dict[str, float] = {
    "fp16": 100.0,
    "int8": 99.5,
    "int4": 97.0,
    "int3": 92.0,
    "int2": 80.0,
}


@dataclass
class QuantEstimate:
    """Memory estimate for a specific quantization level."""

    quant: str
    weight_memory_gb: float
    fits_with_kv: bool  # fits with default context
    max_context: int  # max context that fits
    quality_pct: float
    total_memory_gb: float  # weights + kv(default) + activation + overhead


@dataclass
class MemoryBudget:
    """Complete memory budget analysis for a model on specific hardware."""

    model_id: str
    usable_memory_gb: float
    kv_per_token_gb: float  # KV cache growth per token
    activation_base_gb: float  # base activation memory
    estimates: list[QuantEstimate] = field(default_factory=list)
    recommended_quant: str = ""
    recommended_context: int = 0
    recommended_memory_gb: float = 0.0
    can_run: bool = False


def calc_weight_memory(model: ModelProfile, quant: str) -> float:
    """Calculate model weight memory in GB for given quantization."""
    bpp = BITS_TO_BYTES.get(quant, 0.5)
    overhead = 1.05  # 5% metadata/alignment overhead
    return (model.total_params_b * 1e9 * bpp * overhead) / 1e9


def calc_kv_per_token(model: ModelProfile, kv_dtype_bytes: int = 2) -> float:
    """Calculate KV cache memory per token in GB.

    Each token needs: 2 (K+V) * num_layers * num_kv_heads * head_dim * dtype_bytes
    """
    kv_heads = model.num_kv_heads or model.num_attention_heads
    per_token_bytes = 2 * model.num_layers * kv_heads * model.head_dim * kv_dtype_bytes
    return per_token_bytes / 1e9


def calc_kv_memory(model: ModelProfile, seq_len: int, batch: int = 1) -> float:
    """Calculate total KV cache memory in GB."""
    return calc_kv_per_token(model) * seq_len * batch


def calc_activation_memory(model: ModelProfile, seq_len: int = 1) -> float:
    """Estimate activation/scratch memory in GB.

    Rough estimate: hidden_size * seq_len * num_layers * factor / 1e9
    For single-token decode this is tiny; for prefill it grows.
    """
    # Conservative estimate for decode (batch=1, seq=1)
    base = model.hidden_size * model.num_layers * 4 * 2 / 1e9  # ~few MB
    # Prefill adds quadratic attention component
    if seq_len > 1:
        attn = (seq_len * seq_len * model.num_attention_heads * 2) / 1e9
        base += attn
    return base


def calculate(
    model: ModelProfile,
    hardware: HardwareProfile,
    default_context: int = 8192,
    safety_margin: float = 0.85,
) -> MemoryBudget:
    """Calculate complete memory budget for running a model on given hardware."""
    usable = hardware.usable_memory_gb
    safe_limit = usable * safety_margin

    kv_per_tok = calc_kv_per_token(model)
    activation = calc_activation_memory(model, seq_len=1)

    budget = MemoryBudget(
        model_id=model.model_id,
        usable_memory_gb=usable,
        kv_per_token_gb=kv_per_tok,
        activation_base_gb=activation,
    )

    best_quant = None
    best_context = 0

    for quant in QUANT_LEVELS:
        weight_mem = calc_weight_memory(model, quant)
        remaining = safe_limit - weight_mem - activation

        if remaining <= 0:
            est = QuantEstimate(
                quant=quant,
                weight_memory_gb=weight_mem,
                fits_with_kv=False,
                max_context=0,
                quality_pct=QUALITY_ESTIMATE.get(quant, 0),
                total_memory_gb=weight_mem + activation,
            )
            budget.estimates.append(est)
            continue

        # Max context that fits
        if kv_per_tok > 0:
            max_ctx = int(remaining / kv_per_tok)
        else:
            max_ctx = model.max_context

        max_ctx = min(max_ctx, model.max_context) if model.max_context > 0 else max_ctx
        max_ctx = max(max_ctx, 0)

        # Check if default context fits
        ctx_for_check = min(default_context, max_ctx)
        kv_mem = calc_kv_memory(model, ctx_for_check)
        total = weight_mem + kv_mem + activation
        fits = total <= safe_limit and max_ctx >= 2048

        est = QuantEstimate(
            quant=quant,
            weight_memory_gb=weight_mem,
            fits_with_kv=fits,
            max_context=max_ctx,
            quality_pct=QUALITY_ESTIMATE.get(quant, 0),
            total_memory_gb=total,
        )
        budget.estimates.append(est)

        # Pick the highest quality that fits
        if fits and best_quant is None:
            best_quant = quant
            best_context = min(ctx_for_check, max_ctx)

    if best_quant:
        budget.can_run = True
        budget.recommended_quant = best_quant
        budget.recommended_context = best_context
        weight = calc_weight_memory(model, best_quant)
        kv = calc_kv_memory(model, best_context)
        budget.recommended_memory_gb = weight + kv + activation
    else:
        budget.can_run = False

    return budget


def calc_theoretical_tps(
    model: ModelProfile,
    hardware: HardwareProfile,
    quant: str,
    efficiency: float = 0.37,
) -> float:
    """Estimate theoretical tokens/second based on memory bandwidth.

    Token generation is memory-bound: TPS ≈ bandwidth / model_size * efficiency
    """
    if hardware.memory_bandwidth_gbs <= 0:
        return 0.0
    weight_gb = calc_weight_memory(model, quant)
    if weight_gb <= 0:
        return 0.0
    return (hardware.memory_bandwidth_gbs / weight_gb) * efficiency


def format_report(budget: MemoryBudget) -> str:
    """Format a human-readable memory budget report."""
    lines = [
        "Memory Budget",
        "=" * 50,
        f"  Model:            {budget.model_id}",
        f"  Usable Memory:    {budget.usable_memory_gb:.1f} GB",
        f"  KV Cache/Token:   {budget.kv_per_token_gb * 1e6:.1f} KB",
        "",
        "  Quantization Options:",
        f"  {'Quant':<8} {'Weights':>8} {'Max Ctx':>8} {'Total':>8} {'Quality':>8} {'Fits':>5}",
        f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}",
    ]
    for est in budget.estimates:
        fits_str = "Yes" if est.fits_with_kv else "No"
        ctx_str = f"{est.max_context:,}" if est.max_context > 0 else "N/A"
        lines.append(
            f"  {est.quant:<8} {est.weight_memory_gb:>7.1f}G {ctx_str:>8} "
            f"{est.total_memory_gb:>7.1f}G {est.quality_pct:>7.0f}% {fits_str:>5}"
        )

    lines.append("")
    if budget.can_run:
        lines.append(f"  Recommended: {budget.recommended_quant} @ {budget.recommended_context:,} context")
        lines.append(f"  Est. Memory: {budget.recommended_memory_gb:.1f} GB")
    else:
        lines.append("  Result: Model cannot fit in available memory at any quantization level.")

    return "\n".join(lines)
