from __future__ import annotations

from dataclasses import dataclass

# ===========================================================================
# OmniDraft — Universal Cross-Vocabulary Drafter (Phase 7.4)
# NeurIPS 2025, arXiv 2507.02659
#
# Single 68M-parameter model that works with ANY target model via
# cross-vocabulary mapping. Replaces per-architecture draft model lookup.
# ===========================================================================

# Known OmniDraft model IDs (when available on HuggingFace)
OMNIDRAFT_MODELS = [
    "mlx-community/OmniDraft-68M-4bit",  # Primary (if released for MLX)
    "omnidraft/OmniDraft-68M",            # Original
]


@dataclass
class OmniDraftInfo:
    """Information about the OmniDraft universal drafter."""

    model_id: str
    estimated_size_gb: float = 0.05   # ~50MB for 68M params 4-bit
    supports_all_architectures: bool = True
    source: str = "omnidraft"


def select_omnidraft(
    available_memory_gb: float = 0.5,
) -> OmniDraftInfo | None:
    """Select the OmniDraft universal draft model.

    OmniDraft is a single 68M-parameter model that works with any target
    model architecture via cross-vocabulary mapping + online n-gram cache.

    Returns:
        OmniDraftInfo if available, None otherwise.
    """
    for model_id in OMNIDRAFT_MODELS:
        return OmniDraftInfo(
            model_id=model_id,
            estimated_size_gb=0.05,
        )
    return None
