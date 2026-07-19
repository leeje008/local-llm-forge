"""MoE Expert activation pattern analysis.

Analyzes which experts activate for which inputs, enabling:
- Expert specialization discovery (code/language/domain)
- Activation frequency heatmaps
- SRP/SCH metrics for offloading suitability (arXiv:2505.16056)
- Expert similarity analysis for pruning decisions
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExpertActivation:
    """Record of expert activations for a single token."""

    token_idx: int
    token_text: str
    layer: int
    activated_experts: list[int]
    gate_scores: list[float]


@dataclass
class ExpertProfile:
    """Aggregated expert activation statistics."""

    model_id: str
    num_layers: int = 0
    num_experts: int = 0
    num_active_per_token: int = 0
    total_tokens_analyzed: int = 0

    # Per-expert stats
    activation_frequency: dict[int, int] = field(default_factory=dict)  # expert_id → count
    # layer → {expert → count}
    layer_expert_freq: dict[int, dict[int, int]] = field(default_factory=dict)

    # Specialization
    # expert → {token_type → count}
    expert_token_types: dict[int, Counter] = field(default_factory=dict)

    # SRP/SCH metrics (arXiv:2505.16056)
    srp_score: float = 0.0  # Segment Routing Performance
    sch_score: float = 0.0  # Segment Cache Hit Rate
    offload_suitable: bool = False

    # Expert similarity (for pruning)
    expert_similarity_pairs: list[tuple[int, int, float]] = field(default_factory=list)


def analyze_activations(
    activations: list[ExpertActivation],
    num_experts: int,
    num_active: int,
    cache_size: int | None = None,
) -> ExpertProfile:
    """Analyze expert activation patterns from recorded data.

    Args:
        activations: List of per-token expert activations
        num_experts: Total number of experts per layer
        num_active: Number of active experts per token (K)
        cache_size: Expert cache size for SCH calculation (default: 2*K)
    """
    if cache_size is None:
        cache_size = num_active * 2

    profile = ExpertProfile(
        model_id="",
        num_experts=num_experts,
        num_active_per_token=num_active,
        total_tokens_analyzed=len(activations),
    )

    # Aggregate per-expert activation frequency
    freq: Counter = Counter()
    layer_freq: dict[int, Counter] = defaultdict(Counter)

    for act in activations:
        for eid in act.activated_experts:
            freq[eid] += 1
            layer_freq[act.layer][eid] += 1

    profile.activation_frequency = dict(freq)
    profile.layer_expert_freq = {layer: dict(c) for layer, c in layer_freq.items()}

    # Detect layers
    layers = set(a.layer for a in activations)
    profile.num_layers = len(layers)

    # Calculate SRP (Segment Routing Performance)
    # SRP measures how well a fixed set of experts covers a segment of tokens
    profile.srp_score = _calculate_srp(activations, num_active)

    # Calculate SCH (Segment Cache Hit Rate)
    profile.sch_score = _calculate_sch(activations, cache_size)

    # Offloading suitability: high SCH → good for offloading
    profile.offload_suitable = profile.sch_score >= 0.6

    return profile


def _calculate_srp(activations: list[ExpertActivation], k: int, segment_size: int = 32) -> float:
    """Calculate Segment Routing Performance.

    For each segment of tokens, find the best fixed set of K experts
    and measure what fraction of token requests they cover.
    """
    if not activations:
        return 0.0

    segments = [activations[i:i + segment_size] for i in range(0, len(activations), segment_size)]
    total_coverage = 0.0

    for segment in segments:
        if not segment:
            continue
        # Count expert usage in this segment
        expert_counts: Counter = Counter()
        for act in segment:
            for eid in act.activated_experts:
                expert_counts[eid] += 1

        # Best K experts for this segment
        best_experts = set(eid for eid, _ in expert_counts.most_common(k))

        # Coverage: what fraction of activations are covered by best K
        covered = sum(
            1 for act in segment
            if any(eid in best_experts for eid in act.activated_experts)
        )
        total_coverage += covered / len(segment)

    return total_coverage / len(segments) if segments else 0.0


def _calculate_sch(activations: list[ExpertActivation], cache_size: int) -> float:
    """Calculate Segment Cache Hit Rate.

    Simulates an LRU cache of given size and measures hit rate
    across the activation sequence.
    """
    if not activations or cache_size <= 0:
        return 0.0

    cache: list[int] = []  # LRU order (most recent at end)
    hits = 0
    total = 0

    for act in activations:
        for eid in act.activated_experts:
            total += 1
            if eid in cache:
                hits += 1
                # Move to most recent
                cache.remove(eid)
                cache.append(eid)
            else:
                cache.append(eid)
                if len(cache) > cache_size:
                    cache.pop(0)  # Evict least recently used

    return hits / total if total > 0 else 0.0


def simulate_cache_policies(
    activations: list[ExpertActivation],
    cache_sizes: list[int] | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """Compare cache policies across different cache sizes.

    Returns: {policy_name: [(cache_size, hit_rate), ...]}
    """
    if cache_sizes is None:
        cache_sizes = [2, 4, 8, 16, 32, 64]

    results: dict[str, list[tuple[int, float]]] = {
        "LRU": [],
        "LFU": [],
        "FIFO": [],
    }

    for size in cache_sizes:
        # LRU
        lru_hits = _simulate_lru(activations, size)
        results["LRU"].append((size, lru_hits))

        # LFU
        lfu_hits = _simulate_lfu(activations, size)
        results["LFU"].append((size, lfu_hits))

        # FIFO
        fifo_hits = _simulate_fifo(activations, size)
        results["FIFO"].append((size, fifo_hits))

    return results


def _simulate_lru(activations: list[ExpertActivation], cache_size: int) -> float:
    cache: list[int] = []
    hits = total = 0
    for act in activations:
        for eid in act.activated_experts:
            total += 1
            if eid in cache:
                hits += 1
                cache.remove(eid)
                cache.append(eid)
            else:
                cache.append(eid)
                if len(cache) > cache_size:
                    cache.pop(0)
    return hits / total if total > 0 else 0.0


def _simulate_lfu(activations: list[ExpertActivation], cache_size: int) -> float:
    cache: dict[int, int] = {}  # expert → frequency
    hits = total = 0
    for act in activations:
        for eid in act.activated_experts:
            total += 1
            if eid in cache:
                hits += 1
                cache[eid] += 1
            else:
                if len(cache) >= cache_size:
                    # Evict least frequent
                    min_eid = min(cache, key=cache.get)  # type: ignore
                    del cache[min_eid]
                cache[eid] = 1
    return hits / total if total > 0 else 0.0


def _simulate_fifo(activations: list[ExpertActivation], cache_size: int) -> float:
    cache: list[int] = []
    cache_set: set[int] = set()
    hits = total = 0
    for act in activations:
        for eid in act.activated_experts:
            total += 1
            if eid in cache_set:
                hits += 1
            else:
                if len(cache) >= cache_size:
                    evicted = cache.pop(0)
                    cache_set.discard(evicted)
                cache.append(eid)
                cache_set.add(eid)
    return hits / total if total > 0 else 0.0


def format_expert_report(p: ExpertProfile) -> str:
    """Format expert analysis report."""
    lines = [
        "Expert Activation Analysis",
        "=" * 50,
        f"  Experts:        {p.num_experts} total, {p.num_active_per_token} active/token",
        f"  Layers:         {p.num_layers}",
        f"  Tokens:         {p.total_tokens_analyzed:,}",
        "",
        "  Offloading Metrics (arXiv:2505.16056):",
        f"    SRP Score:    {p.srp_score:.3f}",
        f"    SCH Score:    {p.sch_score:.3f}",
        f"    Suitable:     {'Yes' if p.offload_suitable else 'No'} (threshold: 0.6)",
        "",
    ]

    # Top activated experts
    if p.activation_frequency:
        sorted_experts = sorted(p.activation_frequency.items(), key=lambda x: -x[1])
        lines.append("  Top 10 Most Active Experts:")
        for eid, count in sorted_experts[:10]:
            pct = count / p.total_tokens_analyzed * 100 if p.total_tokens_analyzed else 0
            bar = "#" * int(pct / 2)
            lines.append(f"    Expert {eid:>3}: {count:>6} ({pct:>5.1f}%) {bar}")

        # Least active (pruning candidates)
        lines.append("")
        lines.append("  Bottom 10 Least Active (pruning candidates):")
        for eid, count in sorted_experts[-10:]:
            pct = count / p.total_tokens_analyzed * 100 if p.total_tokens_analyzed else 0
            lines.append(f"    Expert {eid:>3}: {count:>6} ({pct:>5.1f}%)")

    return "\n".join(lines)


def save_profile(profile: ExpertProfile, path: Path) -> None:
    """Save expert profile to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_id": profile.model_id,
        "num_experts": profile.num_experts,
        "num_active_per_token": profile.num_active_per_token,
        "total_tokens": profile.total_tokens_analyzed,
        "srp_score": profile.srp_score,
        "sch_score": profile.sch_score,
        "offload_suitable": profile.offload_suitable,
        "activation_frequency": profile.activation_frequency,
    }
    path.write_text(json.dumps(data, indent=2))
