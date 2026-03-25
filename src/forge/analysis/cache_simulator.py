"""Cache policy simulator for MoE expert weight caching.

Simulates LRU/LFU/ARC/W-TinyLFU/FIFO caching strategies on expert
activation traces and recommends optimal policy + cache size.

Based on: flash-moe (71% OS page cache), Hawkeye (Belady approx),
W-TinyLFU (Caffeine library, 8 bytes/entry).
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field


@dataclass
class CacheSimResult:
    """Result from a single cache simulation run."""

    policy: str
    cache_size: int
    hit_rate: float
    total_accesses: int
    hits: int
    misses: int


@dataclass
class CacheSimReport:
    """Complete cache simulation report comparing multiple policies."""

    num_experts: int = 0
    num_accesses: int = 0
    unique_experts_accessed: int = 0
    results: list[CacheSimResult] = field(default_factory=list)
    recommended_policy: str = ""
    recommended_size: int = 0
    recommended_hit_rate: float = 0.0


def simulate_all_policies(
    access_trace: list[int],
    cache_sizes: list[int] | None = None,
    num_experts: int = 0,
) -> CacheSimReport:
    """Run all cache policies on an access trace and compare.

    Args:
        access_trace: Sequence of expert IDs accessed
        cache_sizes: Cache sizes to test (default: [2, 4, 8, 16, 32, 64])
        num_experts: Total number of experts (for reporting)
    """
    if cache_sizes is None:
        cache_sizes = [2, 4, 8, 16, 32, 64]

    report = CacheSimReport(
        num_experts=num_experts,
        num_accesses=len(access_trace),
        unique_experts_accessed=len(set(access_trace)),
    )

    policies = {
        "LRU": _simulate_lru,
        "LFU": _simulate_lfu,
        "FIFO": _simulate_fifo,
        "ARC": _simulate_arc,
        "W-TinyLFU": _simulate_wtinylfu,
    }

    best_hit_rate = 0.0

    for size in cache_sizes:
        for name, func in policies.items():
            hits, misses = func(access_trace, size)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0

            result = CacheSimResult(
                policy=name, cache_size=size,
                hit_rate=hit_rate, total_accesses=total,
                hits=hits, misses=misses,
            )
            report.results.append(result)

            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                report.recommended_policy = name
                report.recommended_size = size
                report.recommended_hit_rate = hit_rate

    return report


def _simulate_lru(trace: list[int], size: int) -> tuple[int, int]:
    cache: OrderedDict[int, None] = OrderedDict()
    hits = misses = 0
    for eid in trace:
        if eid in cache:
            hits += 1
            cache.move_to_end(eid)
        else:
            misses += 1
            cache[eid] = None
            if len(cache) > size:
                cache.popitem(last=False)
    return hits, misses


def _simulate_lfu(trace: list[int], size: int) -> tuple[int, int]:
    cache: dict[int, int] = {}
    hits = misses = 0
    for eid in trace:
        if eid in cache:
            hits += 1
            cache[eid] += 1
        else:
            misses += 1
            if len(cache) >= size:
                min_key = min(cache, key=cache.get)  # type: ignore
                del cache[min_key]
            cache[eid] = 1
    return hits, misses


def _simulate_fifo(trace: list[int], size: int) -> tuple[int, int]:
    cache: list[int] = []
    cache_set: set[int] = set()
    hits = misses = 0
    for eid in trace:
        if eid in cache_set:
            hits += 1
        else:
            misses += 1
            if len(cache) >= size:
                evicted = cache.pop(0)
                cache_set.discard(evicted)
            cache.append(eid)
            cache_set.add(eid)
    return hits, misses


def _simulate_arc(trace: list[int], size: int) -> tuple[int, int]:
    """Simplified ARC (Adaptive Replacement Cache).

    Maintains two LRU lists: T1 (recency) and T2 (frequency).
    Adapts partition between them based on hit patterns.
    """
    t1: OrderedDict[int, None] = OrderedDict()  # Recent
    t2: OrderedDict[int, None] = OrderedDict()  # Frequent
    b1: OrderedDict[int, None] = OrderedDict()  # Ghost of T1
    b2: OrderedDict[int, None] = OrderedDict()  # Ghost of T2
    p = 0  # Target size for T1

    hits = misses = 0

    for eid in trace:
        if eid in t1:
            hits += 1
            del t1[eid]
            t2[eid] = None
        elif eid in t2:
            hits += 1
            t2.move_to_end(eid)
        else:
            misses += 1
            if eid in b1:
                p = min(size, p + max(1, len(b2) // max(len(b1), 1)))
                del b1[eid]
                _replace(t1, t2, p, size)
                t2[eid] = None
            elif eid in b2:
                p = max(0, p - max(1, len(b1) // max(len(b2), 1)))
                del b2[eid]
                _replace(t1, t2, p, size)
                t2[eid] = None
            else:
                if len(t1) + len(b1) >= size:
                    if len(t1) < size:
                        if b1:
                            b1.popitem(last=False)
                        _replace(t1, t2, p, size)
                    else:
                        if t1:
                            t1.popitem(last=False)
                elif len(t1) + len(b1) + len(t2) + len(b2) >= size:
                    if len(t1) + len(b1) + len(t2) + len(b2) >= 2 * size:
                        if b2:
                            b2.popitem(last=False)
                    _replace(t1, t2, p, size)
                t1[eid] = None

        # Trim ghost lists
        while len(b1) > size:
            b1.popitem(last=False)
        while len(b2) > size:
            b2.popitem(last=False)

    return hits, misses


def _replace(t1, t2, p, size):
    if t1 and (len(t1) > p or (len(t1) == p and len(t2) > 0)):
        old = next(iter(t1))
        del t1[old]
    elif t2:
        old = next(iter(t2))
        del t2[old]


def _simulate_wtinylfu(trace: list[int], size: int) -> tuple[int, int]:
    """Simplified W-TinyLFU (Window + TinyLFU admission).

    Window cache (1% of size) feeds into main cache via frequency filter.
    """
    window_size = max(1, size // 100) or 1
    main_size = size - window_size

    window: OrderedDict[int, None] = OrderedDict()
    main_cache: OrderedDict[int, None] = OrderedDict()
    freq: Counter = Counter()  # Approximate frequency sketch

    hits = misses = 0

    for eid in trace:
        freq[eid] += 1

        if eid in window or eid in main_cache:
            hits += 1
            if eid in window:
                window.move_to_end(eid)
            else:
                main_cache.move_to_end(eid)
        else:
            misses += 1
            # Add to window
            window[eid] = None
            if len(window) > window_size:
                victim_key = next(iter(window))
                del window[victim_key]

                # Try to admit to main via TinyLFU
                if main_cache and len(main_cache) >= main_size:
                    main_victim = next(iter(main_cache))
                    if freq[victim_key] > freq[main_victim]:
                        del main_cache[main_victim]
                        main_cache[victim_key] = None
                else:
                    main_cache[victim_key] = None

        # Periodic frequency decay (every 10*size accesses)
        if sum(freq.values()) > 10 * size:
            for k in list(freq):
                freq[k] //= 2
                if freq[k] == 0:
                    del freq[k]

    return hits, misses


def format_cache_report(report: CacheSimReport) -> str:
    """Format cache simulation comparison report."""
    lines = [
        "Cache Policy Simulation",
        "=" * 60,
        f"  Total Accesses:    {report.num_accesses:,}",
        f"  Unique Experts:    {report.unique_experts_accessed}",
        f"  Recommended:       {report.recommended_policy} (size={report.recommended_size}, "
        f"hit={report.recommended_hit_rate:.1%})",
        "",
    ]

    # Group by cache size
    sizes = sorted(set(r.cache_size for r in report.results))
    policies = sorted(set(r.policy for r in report.results))

    # Header
    header = f"  {'Size':>6}"
    for p in policies:
        header += f" {p:>10}"
    lines.append(header)
    lines.append("  " + "-" * (6 + 10 * len(policies)))

    for size in sizes:
        row = f"  {size:>6}"
        for p in policies:
            r = next((r for r in report.results if r.cache_size == size and r.policy == p), None)
            if r:
                row += f" {r.hit_rate:>9.1%}"
            else:
                row += f" {'N/A':>10}"
        lines.append(row)

    return "\n".join(lines)
