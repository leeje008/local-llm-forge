"""ReDrafter integration — Apple Research RNN-based speculative decoding.

ReDrafter uses a recurrent neural network (RNN) as the draft model instead
of a separate autoregressive transformer. This enables:
- Dynamic tree attention for multi-candidate verification
- Knowledge distillation from target model
- Verified MLX performance: M1 Max 1.37x, M2 Ultra 1.52x (arXiv:2403.09919)

Reference: https://github.com/apple/ml-recurrent-drafter
"""

from __future__ import annotations

from dataclasses import dataclass

# Known ReDrafter models published by Apple Research
# Format: {target_model_pattern: redrafter_model_id}
# Note: Apple's ReDrafter models are trained per-target-model
REDRAFTER_MODELS: dict[str, str] = {
    # Apple's published ReDrafter checkpoints (when available)
    # These are hypothetical IDs — actual availability depends on Apple's releases
    "vicuna": "apple/redrafter-vicuna-7b",
    "llama-2-7b": "apple/redrafter-llama2-7b",
    "llama-2-13b": "apple/redrafter-llama2-13b",
}

# Fallback: use small models as draft when no ReDrafter available
REDRAFTER_FALLBACKS: dict[str, str] = {
    "llama": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "qwen2": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "qwen3": "mlx-community/Qwen3-0.6B-4bit",
    "mistral": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "gemma": "mlx-community/gemma-2-2b-it-4bit",
    "phi": "mlx-community/Phi-3.5-mini-instruct-4bit",
    "default": "mlx-community/SmolLM2-360M-Instruct-4bit",
}


@dataclass
class ReDrafterInfo:
    """Information about a selected ReDrafter model."""

    model_id: str
    is_true_redrafter: bool  # True if actual ReDrafter, False if fallback
    estimated_speedup: float
    estimated_memory_gb: float
    source: str  # "redrafter" | "fallback_arch_match" | "fallback_default"


def select_redrafter(
    target_architecture: str,
    target_model_id: str = "",
    available_memory_gb: float = 2.0,
) -> ReDrafterInfo:
    """Select the best ReDrafter or draft model for a target.

    Priority:
    1. True ReDrafter model (if available for this target)
    2. Architecture-matched small model from mlx-community
    3. Universal fallback (SmolLM2-360M)
    """
    arch = target_architecture.lower()
    target_id = target_model_id.lower()

    # 1. Check for true ReDrafter model
    for pattern, redrafter_id in REDRAFTER_MODELS.items():
        if pattern in target_id:
            return ReDrafterInfo(
                model_id=redrafter_id,
                is_true_redrafter=True,
                estimated_speedup=1.5,  # ReDrafter typically 1.37-1.52x on Apple Silicon
                estimated_memory_gb=0.5,
                source="redrafter",
            )

    # 2. Architecture-matched fallback
    for pattern, fallback_id in REDRAFTER_FALLBACKS.items():
        if pattern in arch:
            size_est = 0.3 if "0.5B" in fallback_id or "360M" in fallback_id else 1.0
            if size_est <= available_memory_gb:
                return ReDrafterInfo(
                    model_id=fallback_id,
                    is_true_redrafter=False,
                    estimated_speedup=1.2,  # Standard draft typically 1.1-1.3x
                    estimated_memory_gb=size_est,
                    source="fallback_arch_match",
                )

    # 3. Universal fallback
    return ReDrafterInfo(
        model_id=REDRAFTER_FALLBACKS["default"],
        is_true_redrafter=False,
        estimated_speedup=1.1,
        estimated_memory_gb=0.3,
        source="fallback_default",
    )


def load_redrafter(model_id: str) -> tuple:
    """Load a ReDrafter or draft model using mlx_lm.

    Returns (model, tokenizer) tuple compatible with mlx_lm.stream_generate(draft_model=...).
    """
    import mlx_lm

    model, tokenizer = mlx_lm.load(model_id)
    return model, tokenizer


def estimate_redrafter_speedup(
    target_params_b: float,
    is_true_redrafter: bool = False,
    acceptance_rate: float = 0.7,
    num_candidates: int = 5,  # ReDrafter generates multiple candidates
) -> float:
    """Estimate speculative decoding speedup with ReDrafter.

    ReDrafter advantages over standard draft model:
    - Multiple candidate sequences via tree attention
    - Higher acceptance rate due to RNN's sequential awareness
    - Lower overhead due to tiny RNN size
    """
    if is_true_redrafter:
        # ReDrafter-specific: tree attention enables higher effective acceptance
        # Empirical: 1.37x (M1 Max) to 2.3x (H100 reference)
        # For Apple Silicon, conservatively estimate based on chip generation
        base_speedup = 1.4

        # Larger models benefit more from speculation (more compute to amortize)
        if target_params_b >= 30:
            base_speedup *= 1.1
        elif target_params_b >= 7:
            base_speedup *= 1.0
        else:
            base_speedup *= 0.9  # Small models: less benefit

        return base_speedup
    else:
        # Standard draft model: typical acceptance-based formula
        theoretical = 1.0 / (1.0 - acceptance_rate + acceptance_rate / num_candidates)
        overhead = 0.85  # Verification overhead
        return theoretical * overhead


def format_redrafter_info(info: ReDrafterInfo) -> str:
    """Format ReDrafter selection info."""
    lines = [
        "ReDrafter Selection",
        "=" * 50,
        f"  Model:        {info.model_id}",
        f"  Type:         {'True ReDrafter (RNN)' if info.is_true_redrafter else 'Standard Draft'}",
        f"  Est. Speedup: {info.estimated_speedup:.2f}x",
        f"  Est. Memory:  {info.estimated_memory_gb:.1f} GB",
        f"  Source:       {info.source}",
    ]
    return "\n".join(lines)
