from __future__ import annotations

from dataclasses import dataclass

from .cas_spec import CASSpecConfig, build_default_cas_spec_config
from .draft_models import select_draft_model

# ===========================================================================
# Unified Speculative Decoding Strategy Selector
# ===========================================================================

class SpeculativeMethod(str):
    NONE = "none"
    DRAFT_MODEL = "draft_model"     # Traditional: separate draft model
    NGRAM = "ngram"                 # N-gram self-speculation
    OMNIDRAFT = "omnidraft"         # Universal 68M drafter
    REDRAFTER = "redrafter"         # Apple ReDrafter
    EAGLE3 = "eagle3"               # Target-attached EAGLE-3 head (Phase 7.5)
    CAS_SPEC = "cas_spec"           # Cascade Adaptive Self-Speculative (Phase 12, NeurIPS 2025)


@dataclass
class SpeculativeConfig:
    """Unified speculative decoding configuration."""

    method: str = SpeculativeMethod.NONE
    # Draft model
    draft_model_path: str | None = None
    # N-gram
    ngram_order: int = 3
    ngram_max_draft: int = 5
    # Tree
    use_tree: bool = False
    tree_branches: int = 3
    tree_depth: int = 5
    tree_max_nodes: int = 32
    # Adaptive K
    use_adaptive_k: bool = True
    initial_k: int = 3
    # PEARL
    use_pearl: bool = True
    # OmniDraft
    omnidraft_model: str | None = None
    # CAS-Spec (Phase 12) — populated lazily; forward reference to avoid
    # ordering issues since CASSpecConfig is defined below this file.
    cas_spec_config: "CASSpecConfig | None" = None


def select_best_speculative_strategy(
    target_architecture: str,
    target_params_b: float,
    available_memory_gb: float,
) -> SpeculativeConfig:
    """Auto-select the best speculative decoding strategy.

    Decision logic:
    1. If memory allows a draft model → use architecture-matched draft
    2. If target is large (30B+) → use N-gram (saves memory for the model)
    3. If OmniDraft available → universal drafter
    4. Default → N-gram (always works, zero memory)
    """
    config = SpeculativeConfig(use_adaptive_k=True, use_pearl=True)

    # N-gram is always available as baseline
    config.ngram_order = 3
    config.ngram_max_draft = 5

    # Memory check for draft model
    draft_memory_budget = min(2.0, available_memory_gb * 0.05)

    if target_params_b >= 30:
        # Large model: prefer N-gram to save memory
        config.method = SpeculativeMethod.NGRAM
        return config

    # Phase 12 Tier S: CAS-Spec cascade for mid-size models with sufficient memory.
    # Cascaded self-speculation (layer-skip + int4 fast decode + ngram) beats
    # single-tier approaches on 1-15B models when we have headroom for activation
    # quantization buffers (~16GB minimum working set including the target).
    if target_params_b < 15 and available_memory_gb >= 16:
        config.method = SpeculativeMethod.CAS_SPEC
        config.cas_spec_config = build_default_cas_spec_config(
            target_params_b=target_params_b,
            available_memory_gb=available_memory_gb,
        )
        return config

    # Try architecture-matched draft
    draft = select_draft_model(target_architecture, draft_memory_budget)
    if draft and draft.estimated_size_gb <= draft_memory_budget:
        config.method = SpeculativeMethod.DRAFT_MODEL
        config.draft_model_path = draft.model_id
        config.use_tree = True  # Use tree for better acceptance
        return config

    # Fallback to N-gram
    config.method = SpeculativeMethod.NGRAM
    return config


def format_speculative_report(config: SpeculativeConfig) -> str:
    """Format speculative decoding configuration for display."""
    lines = [
        "Speculative Decoding Configuration",
        "=" * 55,
        f"  Method:       {config.method}",
    ]

    if config.draft_model_path:
        lines.append(f"  Draft Model:  {config.draft_model_path}")
    if config.method == SpeculativeMethod.NGRAM:
        lines.append(f"  N-gram Order: {config.ngram_order}")
        lines.append(f"  Max Draft:    {config.ngram_max_draft}")
    if config.use_tree:
        lines.append(f"  Tree:         branches={config.tree_branches}, depth={config.tree_depth}")
    if config.use_adaptive_k:
        lines.append(f"  Adaptive K:   initial={config.initial_k}")
    if config.use_pearl:
        lines.append("  PEARL:        pre-verify + post-verify enabled")

    return "\n".join(lines)
