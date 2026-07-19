"""Speculative decoding support — draft model selection, N-gram self-speculation,
PEARL scheduling, tree attention verification, and adaptive draft length.

Phase 7 enhancements:
- N-gram self-speculation: Draft from token history, no separate model needed
- PEARL scheduling (ICLR 2025): Pre-verify + post-verify overlap, +50% on any SD
- Tree attention: Branch candidates instead of chain, higher acceptance rates
- Adaptive K: Dynamic draft length based on observed acceptance rate
- OmniDraft: Single 68M universal drafter for all target architectures
"""

from __future__ import annotations

from .adaptive_k import AdaptiveKConfig, AdaptiveKController
from .cas_spec import (
    CascadeLevel,
    CASSpecConfig,
    CASSpecDrafter,
    CASSpecStats,
    DyTCRouter,
    build_default_cas_spec_config,
    format_cas_spec_report,
)
from .decoder import NGramSpecResult, NGramSpeculativeDecoder, _argmax_row
from .draft_models import (
    DRAFT_MODELS,
    FALLBACK_DRAFTS,
    DraftModelInfo,
    estimate_speedup,
    select_draft_model,
)
from .ngram import NGramDrafter
from .omnidraft import OMNIDRAFT_MODELS, OmniDraftInfo, select_omnidraft
from .pearl import DraftPlan, PEARLConfig, PEARLScheduler, PEARLStats
from .strategy import (
    SpeculativeConfig,
    SpeculativeMethod,
    format_speculative_report,
    select_best_speculative_strategy,
)
from .tree import TreeConfig, TreeDrafter, TreeNode

__all__ = [
    "DRAFT_MODELS",
    "FALLBACK_DRAFTS",
    "DraftModelInfo",
    "select_draft_model",
    "estimate_speedup",
    "NGramDrafter",
    "PEARLConfig",
    "PEARLStats",
    "PEARLScheduler",
    "DraftPlan",
    "TreeNode",
    "TreeConfig",
    "TreeDrafter",
    "AdaptiveKConfig",
    "AdaptiveKController",
    "NGramSpecResult",
    "NGramSpeculativeDecoder",
    "_argmax_row",
    "OMNIDRAFT_MODELS",
    "OmniDraftInfo",
    "select_omnidraft",
    "SpeculativeMethod",
    "SpeculativeConfig",
    "select_best_speculative_strategy",
    "format_speculative_report",
    "CascadeLevel",
    "CASSpecConfig",
    "CASSpecStats",
    "DyTCRouter",
    "CASSpecDrafter",
    "build_default_cas_spec_config",
    "format_cas_spec_report",
]
