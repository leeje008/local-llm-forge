from __future__ import annotations

from dataclasses import dataclass

# ===========================================================================
# PEARL Scheduling (Phase 7.2) — ICLR 2025, arXiv 2408.11850
#
# Two key optimizations that overlap draft/verify phases:
# 1. Pre-verify: Verify the first draft token during drafting (parallel)
# 2. Post-verify: Generate more draft tokens during target verification
#
# Result: +50% speedup on top of any existing speculative decoding setup.
# ===========================================================================

@dataclass
class PEARLConfig:
    """Configuration for PEARL scheduling."""

    enable_pre_verify: bool = True   # Verify first draft token during drafting
    enable_post_verify: bool = True  # Generate extra drafts during verification
    post_verify_tokens: int = 2      # Extra draft tokens during post-verify phase


@dataclass
class PEARLStats:
    """Runtime statistics for PEARL scheduling."""

    total_steps: int = 0
    pre_verify_hits: int = 0          # First draft accepted without full verify
    post_verify_extra_accepted: int = 0  # Extra tokens from post-verify
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted_tokens / max(self.total_draft_tokens, 1)

    @property
    def pre_verify_hit_rate(self) -> float:
        return self.pre_verify_hits / max(self.total_steps, 1)


class PEARLScheduler:
    """PEARL: Parallel speculative decoding with adaptive draft length.

    Orchestrates the overlap between draft generation and target verification:

    Standard SD:  [--draft--][--verify--][--draft--][--verify--]
    PEARL:        [--draft--]                        (pre-verify first token)
                      [--verify + post-draft--]      (overlap)
                                    [--draft--]      (next round)

    The scheduler manages the state machine for pre-verify and post-verify phases.
    """

    def __init__(self, config: PEARLConfig | None = None):
        self.config = config or PEARLConfig()
        self.stats = PEARLStats()

    def reset(self):
        self.stats = PEARLStats()

    def plan_draft_round(
        self,
        base_draft_len: int,
        recent_acceptance_rate: float,
    ) -> DraftPlan:
        """Plan the next draft round with PEARL optimizations.

        Args:
            base_draft_len: Base number of draft tokens (from adaptive K).
            recent_acceptance_rate: Recent acceptance rate (for adjustment).

        Returns:
            DraftPlan with token counts for each phase.
        """
        self.stats.total_steps += 1

        # Pre-verify: if enabled, we'll verify the first draft token
        # concurrently with generating tokens 2..K
        pre_verify = self.config.enable_pre_verify and base_draft_len > 1

        # Post-verify: during target model verification, generate extra drafts
        post_extra = 0
        if self.config.enable_post_verify and recent_acceptance_rate > 0.5:
            # Only post-verify when acceptance is reasonable
            post_extra = self.config.post_verify_tokens

        return DraftPlan(
            main_draft_len=base_draft_len,
            pre_verify_first=pre_verify,
            post_verify_extra=post_extra,
            total_candidates=base_draft_len + post_extra,
        )

    def record_result(
        self,
        drafted: int,
        accepted: int,
        pre_verify_accepted: bool = False,
        post_verify_accepted: int = 0,
    ):
        """Record the result of a speculative decoding round."""
        self.stats.total_draft_tokens += drafted
        self.stats.total_accepted_tokens += accepted
        if pre_verify_accepted:
            self.stats.pre_verify_hits += 1
        self.stats.post_verify_extra_accepted += post_verify_accepted


@dataclass
class DraftPlan:
    """Plan for a single speculative decoding round."""

    main_draft_len: int          # Number of main draft tokens
    pre_verify_first: bool       # Whether to pre-verify the first token
    post_verify_extra: int       # Extra tokens to draft during verification
    total_candidates: int        # Total candidate tokens
