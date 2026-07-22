from __future__ import annotations

from dataclasses import dataclass

# Known draft model mappings: architecture → small model
DRAFT_MODELS: dict[str, list[str]] = {
    "llama": [
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "meta-llama/Llama-3.2-1B",
    ],
    "qwen2": [
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ],
    "qwen3": [
        "mlx-community/Qwen3-0.6B-4bit",
        "Qwen/Qwen3-0.6B",
    ],
    "mistral": [
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    ],
    "gemma": [
        "mlx-community/gemma-2-2b-it-4bit",
    ],
    "phi": [
        "mlx-community/Phi-3.5-mini-instruct-4bit",
    ],
}

# Universal fallback
FALLBACK_DRAFTS = [
    "mlx-community/SmolLM2-360M-Instruct-4bit",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]


@dataclass
class DraftModelInfo:
    """Information about a selected draft model."""

    model_id: str
    estimated_size_gb: float
    architecture_match: bool
    source: str  # "architecture_match" | "fallback"


def select_draft_model(
    target_architecture: str,
    available_memory_gb: float = 2.0,
    prefer_mlx_community: bool = True,
) -> DraftModelInfo | None:
    """Select the best draft model for speculative decoding.

    Prioritizes mlx-community quantized models (already in MLX format).
    """
    arch = target_architecture.lower()

    # Try architecture-matched models first
    for key, candidates in DRAFT_MODELS.items():
        if key in arch:
            for candidate in candidates:
                is_mlx = "mlx-community" in candidate
                if prefer_mlx_community and not is_mlx:
                    continue
                # Rough size estimate: mlx-community 4-bit models are tiny
                est_size = 0.3 if "0.5B" in candidate or "360M" in candidate else 0.8
                if est_size < available_memory_gb:
                    return DraftModelInfo(
                        model_id=candidate,
                        estimated_size_gb=est_size,
                        architecture_match=True,
                        source="architecture_match",
                    )
    # Fallback to universal small model (respecting the memory budget)
    for fallback in FALLBACK_DRAFTS:
        est_size = 0.3
        if est_size < available_memory_gb:
            return DraftModelInfo(
                model_id=fallback,
                estimated_size_gb=est_size,
                architecture_match=False,
                source="fallback",
            )

    return None


def estimate_speedup(
    target_params_b: float,
    draft_params_b: float = 0.5,
    acceptance_rate: float = 0.7,
    num_draft_tokens: int = 3,
) -> float:
    """Estimate speculative decoding speedup factor.

    Based on the formula: speedup ≈ 1 / (1 - acceptance_rate + acceptance_rate/num_draft_tokens)
    adjusted for the overhead of running the draft model.
    """
    if acceptance_rate <= 0:
        return 1.0

    # Theoretical speedup from speculation
    theoretical = 1.0 / (1.0 - acceptance_rate + acceptance_rate / num_draft_tokens)

    # Overhead factor: draft model adds verification cost
    # Larger ratio of draft/target → less overhead
    overhead = 1.0 - (draft_params_b / target_params_b) * 0.1

    return theoretical * overhead
