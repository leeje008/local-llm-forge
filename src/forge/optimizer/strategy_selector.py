"""Automatic optimization strategy selection based on model + hardware analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.memory_calculator import (
    MemoryBudget,
    calc_theoretical_tps,
    calc_weight_memory,
)
from forge.analyzer.model_inspector import ModelProfile


@dataclass
class OptimizationStrategy:
    """Complete optimization plan for a model on specific hardware."""

    # Core settings
    quantization: str = "int4"  # fp16, int8, int4, int3, int2
    quant_method: str = "mlx_native"  # mlx_native, hqq, aqlm
    format: str = "mlx"  # mlx, gguf
    runtime: str = "mlx-lm"  # mlx-lm, ollama, llama.cpp
    context_length: int = 4096
    batch_size: int = 1

    # Advanced optimizations
    use_speculative: bool = False
    draft_model: str | None = None
    use_prompt_cache: bool = True
    use_paged_attention: bool = False
    expert_cache_size: int | None = None  # MoE only
    mixed_quant_recipe: str | None = None  # e.g. "mixed_3_6"

    # Estimates
    estimated_tps: float = 0.0
    estimated_memory_gb: float = 0.0
    estimated_quality_pct: float = 0.0

    # Metadata
    warnings: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)


# Draft model suggestions per architecture family
_DRAFT_MODELS: dict[str, str] = {
    "llama": "meta-llama/Llama-3.2-1B",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "qwen3": "Qwen/Qwen3-0.6B",
    "mistral": "mistralai/Mistral-7B-v0.3",  # no small variant
    "gemma": "google/gemma-2b",
    "phi": "microsoft/phi-2",
}


def _select_quant_method(quant: str, hardware: HardwareProfile) -> tuple[str, str | None]:
    """Select quantization method and optional mixed recipe."""
    if quant in ("fp16", "int8", "int4"):
        return "mlx_native", None

    if quant == "int3":
        if hardware.has_mlx:
            return "mlx_native", "mixed_3_6"
        return "hqq", None

    if quant == "int2":
        return "hqq", None

    return "mlx_native", None


def _select_runtime(hardware: HardwareProfile, format: str) -> str:
    if format == "mlx" and hardware.has_mlx:
        return "mlx-lm"
    if hardware.has_ollama:
        return "ollama"
    return "llama.cpp"


def _find_draft_model(model: ModelProfile) -> str | None:
    """Find a suitable draft model for speculative decoding."""
    arch = model.architecture.lower()
    for key, draft in _DRAFT_MODELS.items():
        if key in arch:
            return draft
    return "HuggingFaceTB/SmolLM2-360M"


def select(
    model: ModelProfile,
    hardware: HardwareProfile,
    budget: MemoryBudget,
    enable_speculative: bool = False,
    force_quant: str | None = None,
    force_runtime: str | None = None,
) -> OptimizationStrategy:
    """Select the optimal optimization strategy."""
    strategy = OptimizationStrategy()
    reasoning = []

    if not budget.can_run and force_quant is None:
        strategy.warnings.append("Model cannot fit in available memory at any quantization level")
        strategy.reasoning = ["Model too large for available hardware"]
        return strategy

    # 1. Quantization level
    if force_quant:
        strategy.quantization = force_quant
        reasoning.append(f"Quantization forced to {force_quant}")
    else:
        strategy.quantization = budget.recommended_quant
        reasoning.append(
            f"Selected {budget.recommended_quant} quantization "
            f"(highest quality that fits in {hardware.usable_memory_gb:.0f}GB)"
        )

    # 2. Quantization method
    method, recipe = _select_quant_method(strategy.quantization, hardware)
    strategy.quant_method = method
    strategy.mixed_quant_recipe = recipe
    if recipe:
        reasoning.append(f"Using mixed quantization recipe: {recipe}")

    # 3. Format & Runtime
    if strategy.quant_method == "hqq":
        # HQQ outputs PyTorch tensors; we convert to MLX if possible
        strategy.format = "mlx" if hardware.has_mlx else "gguf"
    else:
        strategy.format = "mlx" if hardware.has_mlx else "gguf"

    if force_runtime:
        strategy.runtime = force_runtime
    else:
        strategy.runtime = _select_runtime(hardware, strategy.format)
    reasoning.append(f"Runtime: {strategy.runtime} (format: {strategy.format})")

    # 4. Context length
    if force_quant:
        # Recalculate context for forced quant
        from forge.analyzer.memory_calculator import calc_kv_per_token

        weight_mem = calc_weight_memory(model, force_quant)
        remaining = hardware.usable_memory_gb * 0.85 - weight_mem - 0.1
        kv_pt = calc_kv_per_token(model)
        if kv_pt > 0 and remaining > 0:
            max_ctx = int(remaining / kv_pt)
            strategy.context_length = min(max_ctx, model.max_context or max_ctx, 32768)
        else:
            strategy.context_length = 2048
    else:
        strategy.context_length = budget.recommended_context
    strategy.context_length = max(strategy.context_length, 2048)
    reasoning.append(f"Context length: {strategy.context_length:,}")

    # 5. Batch size (always 1 for interactive inference)
    strategy.batch_size = 1

    # 6. Speculative decoding
    if enable_speculative and model.total_params_b >= 7.0:
        draft = _find_draft_model(model)
        draft_mem = 0.5  # rough estimate for small draft model in 4-bit
        weight_mem = calc_weight_memory(model, strategy.quantization)
        if weight_mem + draft_mem < hardware.usable_memory_gb * 0.7:
            strategy.use_speculative = True
            strategy.draft_model = draft
            reasoning.append(f"Speculative decoding enabled with draft: {draft}")
        else:
            reasoning.append("Speculative decoding skipped (insufficient memory for draft model)")

    # 7. Prompt cache (always enabled)
    strategy.use_prompt_cache = True

    # 8. PagedAttention (for long contexts)
    if strategy.context_length >= 8192:
        strategy.use_paged_attention = True
        reasoning.append("PagedAttention enabled for long context")

    # 9. MoE expert cache
    if model.model_type == "moe" and model.num_active_experts:
        strategy.expert_cache_size = model.num_active_experts * 2
        reasoning.append(f"Expert cache: {strategy.expert_cache_size} entries")

    # 10. Performance estimates
    strategy.estimated_tps = calc_theoretical_tps(
        model, hardware, strategy.quantization
    )
    strategy.estimated_memory_gb = budget.recommended_memory_gb
    # Find quality estimate
    for est in budget.estimates:
        if est.quant == strategy.quantization:
            strategy.estimated_quality_pct = est.quality_pct
            break

    strategy.reasoning = reasoning
    return strategy


def format_report(s: OptimizationStrategy) -> str:
    """Format a human-readable strategy report."""
    lines = [
        "Optimization Strategy",
        "=" * 50,
        f"  Quantization:     {s.quantization} ({s.quant_method})",
    ]
    if s.mixed_quant_recipe:
        lines.append(f"  Mixed Recipe:     {s.mixed_quant_recipe}")
    lines.extend([
        f"  Format:           {s.format}",
        f"  Runtime:          {s.runtime}",
        f"  Context Length:   {s.context_length:,}",
        f"  Batch Size:       {s.batch_size}",
        "",
        "  Optimizations:",
        f"    Prompt Cache:     {'On' if s.use_prompt_cache else 'Off'}",
        f"    PagedAttention:   {'On' if s.use_paged_attention else 'Off'}",
        f"    Speculative:      {'On' if s.use_speculative else 'Off'}",
    ])
    if s.use_speculative and s.draft_model:
        lines.append(f"    Draft Model:      {s.draft_model}")
    if s.expert_cache_size:
        lines.append(f"    Expert Cache:     {s.expert_cache_size} entries")
    lines.extend([
        "",
        "  Estimates:",
        f"    Speed:   ~{s.estimated_tps:.1f} tok/s",
        f"    Memory:  ~{s.estimated_memory_gb:.1f} GB",
        f"    Quality: ~{s.estimated_quality_pct:.0f}% of FP16",
    ])
    if s.reasoning:
        lines.append("")
        lines.append("  Reasoning:")
        for r in s.reasoning:
            lines.append(f"    - {r}")
    if s.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in s.warnings:
            lines.append(f"    ! {w}")
    return "\n".join(lines)
