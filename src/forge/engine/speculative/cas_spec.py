from __future__ import annotations

import math
from dataclasses import dataclass, field

from .ngram import NGramDrafter
from .tree import TreeNode

# ===========================================================================
# CAS-Spec — Cascade Adaptive Self-Speculative Decoding (Phase 12, Tier S)
# NeurIPS 2025, arXiv 2510.26843
#
# Self-speculation via *cascaded* drafters at different cost/quality points:
#   Tier 0 (cheapest): n-gram lookup          — ~10ms / token, ~70% accept
#   Tier 1: layer-skip partial forward        — ~25ms / token, ~55% accept
#   Tier 2: int4 fast-path decode             — ~60ms / token, ~40% accept
#
# The Dynamic Tree Cascade (DyTC) router picks the tier whose expected
# throughput (acceptance_rate * k / latency) is highest for the current
# context, and can fuse multiple tiers into a small speculation tree so that
# a cheap tier's branches are re-expanded by a more expensive tier when they
# look promising. Tiers whose learned acceptance falls below `prune_threshold`
# are dropped from the tree.
#
# Reported gains in the paper:
#   +47% over a single-tier cascade baseline
#   1.1x - 2.3x end-to-end vs. vanilla decode
#
# NOTE: Real layer-skip and int4-fast-path drafting requires hooks inside
# MLXEngine (exposing a partial-forward / quantized-forward path). Those are
# added in `engine/mlx_engine.py` in a companion change; this file only
# defines the structural router and drafter and uses NGramDrafter as a
# placeholder for the deeper tiers until those hooks land.
# ===========================================================================


@dataclass
class CascadeLevel:
    """One tier of the CAS-Spec cascade.

    A level bundles a drafting strategy ID, its calibrated cost (latency per
    token), its learned quality (expected acceptance rate), and a cap on how
    many tokens it may draft per round. The DyTC router uses the
    latency/acceptance pair to rank levels by expected throughput.
    """

    name: str
    draft_fn_id: str              # "ngram" | "layer_skip_<N>" | "int4_decode"
    expected_latency_ms: float    # per token, calibrated
    expected_acceptance_rate: float = 0.5  # learned online via EMA
    max_draft_length: int = 4
    config_overrides: dict = field(default_factory=dict)


@dataclass
class CASSpecConfig:
    """Full cascade configuration for CAS-Spec."""

    levels: list[CascadeLevel]
    routing_strategy: str = "dytc"        # "dytc" | "fixed" | "greedy"
    acceptance_ema_alpha: float = 0.2     # online update rate
    min_samples_for_route: int = 20       # warmup: cycle through levels
    tree_branch_factor: int = 3           # DyTC tree branching per level
    tree_max_depth: int = 4
    prune_threshold: float = 0.15         # drop branches below this acceptance
    history_window: int = 64              # per-level acceptance history length


@dataclass
class CASSpecStats:
    """Runtime statistics for a CAS-Spec decoding session."""

    tokens_generated: int = 0
    level_usage_counts: dict[str, int] = field(default_factory=dict)
    level_acceptance_history: dict[str, list[float]] = field(default_factory=dict)
    total_drafts_generated: int = 0
    total_drafts_accepted: int = 0
    avg_speedup_estimate: float = 1.0

    def summary(self) -> dict:
        accept_rate = self.total_drafts_accepted / max(self.total_drafts_generated, 1)
        per_level: dict[str, dict] = {}
        for name, hist in self.level_acceptance_history.items():
            if hist:
                per_level[name] = {
                    "uses": self.level_usage_counts.get(name, 0),
                    "avg_acceptance": round(sum(hist) / len(hist), 3),
                    "last_acceptance": round(hist[-1], 3),
                    "samples": len(hist),
                }
            else:
                per_level[name] = {
                    "uses": self.level_usage_counts.get(name, 0),
                    "avg_acceptance": 0.0,
                    "last_acceptance": 0.0,
                    "samples": 0,
                }
        return {
            "tokens_generated": self.tokens_generated,
            "total_drafts_generated": self.total_drafts_generated,
            "total_drafts_accepted": self.total_drafts_accepted,
            "overall_acceptance_rate": round(accept_rate, 3),
            "avg_speedup_estimate": round(self.avg_speedup_estimate, 3),
            "per_level": per_level,
        }


class DyTCRouter:
    """Dynamic Tree Cascade router for CAS-Spec.

    Maintains per-level EMAs of acceptance rate and latency, scores levels by
    `acceptance * max_draft_length / latency_ms` (expected accepted tokens per
    ms), and exposes:

    - `route()`: pick one level for the current context. During warmup (fewer
      than `min_samples_for_route` observations across all levels) the router
      cycles through levels round-robin to build statistics. A per-context
      hint cache (keyed by `context_hash`) keeps routing stable across turns
      of the same multi-turn conversation.
    - `update()`: EMA update from a completed round.
    - `build_tree()`: build a small speculation tree rooted at `root_level`
      where promising children are re-expanded by cheaper tiers (reusing the
      existing `TreeNode` / `TreeConfig` dataclasses).
    """

    def __init__(self, config: CASSpecConfig):
        self.config = config
        # EMAs start from the calibrated defaults baked into each level.
        self._acceptance_ema: dict[str, float] = {
            lv.name: lv.expected_acceptance_rate for lv in config.levels
        }
        self._latency_ema: dict[str, float] = {
            lv.name: lv.expected_latency_ms for lv in config.levels
        }
        self._sample_counts: dict[str, int] = {lv.name: 0 for lv in config.levels}
        self._stats = CASSpecStats()
        for lv in config.levels:
            self._stats.level_usage_counts[lv.name] = 0
            self._stats.level_acceptance_history[lv.name] = []
        # Per-context hint cache: last-used level per context_hash.
        self._context_hint: dict[str, str] = {}
        self._round_robin_idx = 0

    # ---- routing ---------------------------------------------------------

    def _total_samples(self) -> int:
        return sum(self._sample_counts.values())

    def _score(self, level: CascadeLevel) -> float:
        acc = self._acceptance_ema[level.name]
        lat = max(self._latency_ema[level.name], 1e-3)
        return (acc * level.max_draft_length) / lat

    def _level_by_name(self, name: str) -> CascadeLevel | None:
        for lv in self.config.levels:
            if lv.name == name:
                return lv
        return None

    def route(self, context_hash: str, recent_tokens: list[int]) -> CascadeLevel:
        """Pick a cascade level for the current decoding step."""
        levels = self.config.levels
        if not levels:
            raise ValueError("DyTCRouter: no cascade levels configured")

        # Warmup: round-robin so every level gets observed.
        if self._total_samples() < self.config.min_samples_for_route:
            chosen = levels[self._round_robin_idx % len(levels)]
            self._round_robin_idx += 1
            return chosen

        # Fixed / greedy strategies bypass DyTC scoring.
        if self.config.routing_strategy == "fixed":
            return levels[0]
        if self.config.routing_strategy == "greedy":
            # Greedy = always the highest-acceptance level regardless of cost.
            return max(levels, key=lambda lv: self._acceptance_ema[lv.name])

        # DyTC: throughput-optimal, with a context-stability bias.
        best_name_hint = self._context_hint.get(context_hash)
        scored = sorted(levels, key=self._score, reverse=True)
        best = scored[0]

        # Hysteresis: if the cached level is within 10% of the best score,
        # stick with it to avoid oscillation within a single conversation.
        if best_name_hint:
            hinted = self._level_by_name(best_name_hint)
            if hinted is not None:
                if self._score(hinted) >= self._score(best) * 0.9:
                    best = hinted

        self._context_hint[context_hash] = best.name
        return best

    # ---- online updates --------------------------------------------------

    def update(
        self,
        level_name: str,
        drafts_proposed: int,
        drafts_accepted: int,
        latency_ms: float,
    ):
        """EMA update for acceptance + latency after a round."""
        if level_name not in self._acceptance_ema:
            return  # unknown level, ignore
        alpha = self.config.acceptance_ema_alpha

        if drafts_proposed > 0:
            rate = drafts_accepted / drafts_proposed
            self._acceptance_ema[level_name] = (
                (1 - alpha) * self._acceptance_ema[level_name] + alpha * rate
            )
            hist = self._stats.level_acceptance_history[level_name]
            hist.append(rate)
            if len(hist) > self.config.history_window:
                del hist[: len(hist) - self.config.history_window]

        if latency_ms > 0 and drafts_proposed > 0:
            per_token = latency_ms / max(drafts_proposed, 1)
            self._latency_ema[level_name] = (
                (1 - alpha) * self._latency_ema[level_name] + alpha * per_token
            )

        self._sample_counts[level_name] += 1
        self._stats.level_usage_counts[level_name] = (
            self._stats.level_usage_counts.get(level_name, 0) + 1
        )
        self._stats.total_drafts_generated += drafts_proposed
        self._stats.total_drafts_accepted += drafts_accepted
        # Running throughput estimate as a proxy for end-to-end speedup.
        self._stats.avg_speedup_estimate = max(
            1.0,
            1.0 + self._stats.total_drafts_accepted
            / max(self._stats.total_drafts_generated, 1)
            * 1.5,  # heuristic scale calibrated to paper's 1.1-2.3x range
        )

    # ---- tree construction -----------------------------------------------

    def build_tree(
        self,
        root_context: list[int],
        root_level: CascadeLevel,
    ) -> list[TreeNode]:
        """Build a DyTC speculation tree.

        The root is expanded with `root_level.max_draft_length` children
        (one per speculated token), and each child is further expanded using
        progressively cheaper tiers while their expected acceptance stays
        above `prune_threshold`. Branches below the threshold are pruned.

        We reuse the existing `TreeNode` dataclass (line ~368) so the output
        is compatible with `TreeDrafter.build_attention_mask` /
        `select_longest_accepted_path`.
        """
        cfg = self.config
        nodes: list[TreeNode] = [TreeNode(token_id=-1, depth=0, parent_idx=-1)]

        # Order levels from most expensive (root tier) to cheapest.
        ordered = sorted(
            cfg.levels,
            key=lambda lv: lv.expected_latency_ms,
            reverse=True,
        )
        # Find the starting index so the root uses root_level.
        try:
            start = ordered.index(root_level)
        except ValueError:
            start = 0

        def expand(parent_idx: int, depth: int, level_idx: int):
            if depth >= cfg.tree_max_depth:
                return
            if level_idx >= len(ordered):
                return
            level = ordered[level_idx]
            if self._acceptance_ema[level.name] < cfg.prune_threshold:
                return  # prune: this tier is too weak to be worth branching
            branches = min(cfg.tree_branch_factor, level.max_draft_length)
            for b in range(branches):
                # Placeholder token IDs: the actual tokens are filled in by
                # CASSpecDrafter.propose() when it runs the draft function.
                # We pre-allocate structural nodes so that a tree attention
                # mask can be built upfront.
                child = TreeNode(
                    token_id=-2 - b,   # sentinel; replaced at draft-time
                    depth=depth + 1,
                    parent_idx=parent_idx,
                    log_prob=math.log(
                        max(self._acceptance_ema[level.name], 1e-4)
                    ),
                )
                child_idx = len(nodes)
                nodes.append(child)
                nodes[parent_idx].children_idx.append(child_idx)
                # Each child is re-expanded with the *next cheaper* tier.
                expand(child_idx, depth + 1, level_idx + 1)

        expand(0, 0, start)
        return nodes

    # ---- stats -----------------------------------------------------------

    def stats(self) -> CASSpecStats:
        return self._stats


class CASSpecDrafter:
    """Cascade Adaptive Self-Speculative drafter.

    Wires together a `DyTCRouter`, the existing `NGramDrafter`, and stubs for
    the layer-skip / int4-fast-path tiers. On each `step()` the drafter asks
    the router for a tier, drafts `k` tokens using that tier, returns them
    for verification by the target model, and updates the router with the
    observed acceptance rate.

    Real layer-skip / int4 drafting require MLXEngine hooks (a partial
    forward that runs only the first `N` layers, and a forward that runs the
    model under aggressive int4 activation quantization). Until those land
    these tiers fall back to NGramDrafter so the class is end-to-end
    testable today.
    """

    def __init__(
        self,
        target_model_info,
        cascade_config: CASSpecConfig,
        router: DyTCRouter | None = None,
    ):
        self.target_model_info = target_model_info
        self.config = cascade_config
        self.router = router or DyTCRouter(cascade_config)
        # Shared n-gram drafter reused across tiers that currently stub-out
        # to it. A single instance keeps the n-gram table warm.
        self._ngram = NGramDrafter(n=3, max_draft=8)
        self._last_level: CascadeLevel | None = None

    # ---- tier implementations -------------------------------------------

    def _ngram_draft(self, level: CascadeLevel, context: list[int]) -> list[int]:
        self._ngram.max_draft = level.max_draft_length
        return self._ngram.draft(context)[: level.max_draft_length]

    def _layer_skip_draft(
        self,
        n_skip: int,
        level: CascadeLevel,
        context: list[int],
    ) -> list[int]:
        """Layer-skip self-draft stub.

        TODO(phase12): wire to `MLXEngine.partial_forward(layers=N-n_skip)`
        in `src/forge/engine/mlx_engine.py` once that hook is exposed. Until
        then we reuse the n-gram table so this tier is structurally present
        and the DyTC router can learn against it in tests.
        """
        _ = n_skip  # hook placeholder
        return self._ngram_draft(level, context)

    def _int4_fast_decode_draft(
        self,
        level: CascadeLevel,
        context: list[int],
    ) -> list[int]:
        """Int4 activation-quantized fast-path draft stub.

        TODO(phase12): wire to `MLXEngine.int4_fast_decode(...)` once the
        int4 activation-quantization path lands (companion change in
        `engine/mlx_engine.py`). Falls back to n-gram today.
        """
        return self._ngram_draft(level, context)

    # ---- public API ------------------------------------------------------

    def observe(self, token_id: int):
        """Feed an accepted/generated token into the n-gram table."""
        self._ngram.observe(token_id)

    def propose(
        self,
        prompt_tokens: list[int],
        k: int | None = None,
    ) -> list[int]:
        """Route + draft one batch of tokens."""
        context_hash = str(hash(tuple(prompt_tokens[-16:])))
        level = self.router.route(context_hash, prompt_tokens[-8:])
        self._last_level = level

        if k is not None:
            # Temporary override of max_draft_length for this round only.
            effective_max = min(k, level.max_draft_length)
        else:
            effective_max = level.max_draft_length

        did = level.draft_fn_id
        if did == "ngram":
            drafts = self._ngram_draft(level, prompt_tokens)
        elif did.startswith("layer_skip_"):
            try:
                n_skip = int(did.split("_")[-1])
            except ValueError:
                n_skip = 0
            drafts = self._layer_skip_draft(n_skip, level, prompt_tokens)
        elif did == "int4_decode":
            drafts = self._int4_fast_decode_draft(level, prompt_tokens)
        else:
            drafts = self._ngram_draft(level, prompt_tokens)

        return drafts[:effective_max]

    def verify(
        self,
        drafts: list[int],
        target_logits,
    ) -> tuple[int, list[int]]:
        """Simple greedy verification against target top-1 predictions.

        Compatible with the existing tree verification helpers: callers that
        already have target logits per position can compare argmax-by-position
        to the draft sequence. Returns `(accepted_count, accepted_tokens)`.
        """
        if target_logits is None or not drafts:
            return 0, []
        accepted: list[int] = []
        try:
            import mlx.core as mx  # noqa: F401
            has_mlx = True
        except Exception:
            has_mlx = False

        for i, draft_token in enumerate(drafts):
            row = target_logits[i] if hasattr(target_logits, "__len__") else None
            if row is None:
                break
            try:
                if has_mlx and hasattr(row, "shape"):
                    import mlx.core as mx
                    pred = int(mx.argmax(row).item())
                else:
                    pred = int(max(range(len(row)), key=lambda j: row[j]))
            except Exception:
                break
            if pred == draft_token:
                accepted.append(draft_token)
            else:
                break
        return len(accepted), accepted

    def step(self, prompt_state) -> tuple[int, str]:
        """One full round: propose + verify + update.

        `prompt_state` is a light protocol: it must expose `tokens` (a list
        of ints) and optionally `target_logits_for(drafts)` returning the
        per-position target logits used for verification. This keeps the
        drafter decoupled from MLXEngine so it can be unit-tested without
        a real model.
        """
        tokens: list[int] = getattr(prompt_state, "tokens", []) or []
        drafts = self.propose(tokens)

        target_logits = None
        if hasattr(prompt_state, "target_logits_for"):
            try:
                target_logits = prompt_state.target_logits_for(drafts)
            except Exception:
                target_logits = None

        accepted_count, accepted_tokens = self.verify(drafts, target_logits)

        level_name = self._last_level.name if self._last_level else "unknown"
        # Latency accounting: in absence of a real timer we attribute the
        # level's calibrated per-token cost. MLXEngine integration will
        # replace this with a wall-clock measurement.
        latency_ms = (
            (self._last_level.expected_latency_ms if self._last_level else 10.0)
            * max(len(drafts), 1)
        )
        self.router.update(
            level_name=level_name,
            drafts_proposed=len(drafts),
            drafts_accepted=accepted_count,
            latency_ms=latency_ms,
        )
        for tok in accepted_tokens:
            self.observe(tok)
        self.router.stats().tokens_generated += accepted_count
        return accepted_count, level_name


def build_default_cas_spec_config(
    target_params_b: float,
    available_memory_gb: float,
) -> CASSpecConfig:
    """Heuristic default cascade for a given model size.

    - 7B-class models get the full 3-tier cascade (ngram → layer-skip-6 →
      int4-decode). The int4 tier is worthwhile because a 7B forward is
      cheap enough that an activation-quantized fast path is a meaningful
      win over the full-precision target.
    - 30B+ models drop the int4 tier: at that scale even int4 decode costs
      dominate, so we stick with ngram + a deeper layer-skip (skip 10
      layers) to keep drafting cheap relative to verification.
    - Intermediate sizes (15-30B) use the 30B+ profile to be conservative.
    """
    _ = available_memory_gb  # reserved for future headroom-based tuning

    if target_params_b < 15:
        levels = [
            CascadeLevel(
                name="ngram",
                draft_fn_id="ngram",
                expected_latency_ms=10.0,
                expected_acceptance_rate=0.70,
                max_draft_length=4,
            ),
            CascadeLevel(
                name="layer_skip_6",
                draft_fn_id="layer_skip_6",
                expected_latency_ms=25.0,
                expected_acceptance_rate=0.55,
                max_draft_length=5,
            ),
            CascadeLevel(
                name="int4_decode",
                draft_fn_id="int4_decode",
                expected_latency_ms=60.0,
                expected_acceptance_rate=0.40,
                max_draft_length=3,
            ),
        ]
    else:
        levels = [
            CascadeLevel(
                name="ngram",
                draft_fn_id="ngram",
                expected_latency_ms=10.0,
                expected_acceptance_rate=0.70,
                max_draft_length=3,
            ),
            CascadeLevel(
                name="layer_skip_10",
                draft_fn_id="layer_skip_10",
                expected_latency_ms=40.0,
                expected_acceptance_rate=0.50,
                max_draft_length=4,
            ),
        ]

    return CASSpecConfig(
        levels=levels,
        routing_strategy="dytc",
        acceptance_ema_alpha=0.2,
        min_samples_for_route=20,
        tree_branch_factor=3,
        tree_max_depth=4,
        prune_threshold=0.15,
    )


def format_cas_spec_report(
    config: CASSpecConfig,
    stats: CASSpecStats | None = None,
) -> str:
    """Human-readable CAS-Spec configuration + live stats report."""
    lines = [
        "CAS-Spec Cascade Configuration (Phase 12, Tier S)",
        "=" * 65,
        f"  Routing:       {config.routing_strategy}",
        f"  EMA alpha:     {config.acceptance_ema_alpha}",
        f"  Warmup:        {config.min_samples_for_route} samples",
        f"  Tree:          branch={config.tree_branch_factor}, "
        f"depth={config.tree_max_depth}, prune<{config.prune_threshold}",
        "",
        "  Cascade Levels (fast → slow):",
        "  " + "-" * 61,
        f"  {'name':<18}{'draft_fn':<18}{'k':>4}{'lat(ms)':>10}{'accept':>10}",
    ]
    for lv in config.levels:
        lines.append(
            f"  {lv.name:<18}{lv.draft_fn_id:<18}"
            f"{lv.max_draft_length:>4}"
            f"{lv.expected_latency_ms:>10.1f}"
            f"{lv.expected_acceptance_rate:>10.2f}"
        )

    if stats is not None:
        summary = stats.summary()
        lines.append("")
        lines.append("  Runtime Stats:")
        lines.append("  " + "-" * 61)
        lines.append(f"  Tokens generated:    {summary['tokens_generated']}")
        lines.append(f"  Drafts generated:    {summary['total_drafts_generated']}")
        lines.append(f"  Drafts accepted:     {summary['total_drafts_accepted']}")
        lines.append(
            f"  Overall acceptance:  {summary['overall_acceptance_rate']}"
        )
        lines.append(
            f"  Est. speedup:        {summary['avg_speedup_estimate']}x"
        )
        lines.append("  Per-level:")
        for name, per in summary["per_level"].items():
            lines.append(
                f"    {name:<18} uses={per['uses']:<5} "
                f"avg_acc={per['avg_acceptance']:<6} "
                f"last={per['last_acceptance']:<6} "
                f"n={per['samples']}"
            )

    return "\n".join(lines)
