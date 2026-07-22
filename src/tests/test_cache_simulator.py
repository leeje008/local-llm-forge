"""Characterization tests for the MoE expert cache-policy simulator.

Pins the CURRENT behavior of ``forge.analysis.cache_simulator``. Pure
Python over lists of int "expert ids" — no mlx, network, or disk I/O.

KNOWN BUG pinned here (not fixed): ``cache_size=0`` crashes for any
non-empty access trace. ``_simulate_lfu`` raises ``ValueError`` from
``min(cache, key=cache.get)`` on an empty dict (it tries to evict before
the cache has ever gained an entry), and ``_simulate_fifo`` would
separately raise ``IndexError`` from ``cache.pop(0)`` on an empty list for
the same reason. Because ``simulate_all_policies`` iterates policies in
the fixed order LRU, LFU, FIFO, ARC, W-TinyLFU, the LFU ``ValueError`` is
what actually surfaces first. See ``test_cache_size_zero_raises_on_lfu``.
"""

from __future__ import annotations

import pytest

from forge.analysis.cache_simulator import (
    CacheSimReport,
    _simulate_arc,
    _simulate_fifo,
    _simulate_lfu,
    _simulate_lru,
    _simulate_wtinylfu,
    format_cache_report,
    simulate_all_policies,
)

# Access trace: 3 unique experts cycled 5 times = 15 accesses.
CYCLE_TRACE = [1, 2, 3] * 5


# --------------------------------------------------------------------------- #
# simulate_all_policies() — happy path
# --------------------------------------------------------------------------- #


def test_report_metadata_reflects_trace():
    report = simulate_all_policies(CYCLE_TRACE, cache_sizes=[3], num_experts=3)
    assert isinstance(report, CacheSimReport)
    assert report.num_accesses == 15
    assert report.unique_experts_accessed == 3
    assert report.num_experts == 3


def test_default_cache_sizes_used_when_not_specified():
    report = simulate_all_policies(CYCLE_TRACE)
    sizes = sorted({r.cache_size for r in report.results})
    assert sizes == [2, 4, 8, 16, 32, 64]


@pytest.mark.parametrize(
    "policy, expected_hits, expected_misses",
    [
        ("LRU", 12, 3),
        ("LFU", 12, 3),
        ("FIFO", 12, 3),
        ("ARC", 12, 3),
        ("W-TinyLFU", 12, 3),
    ],
)
def test_all_policies_hit_rate_when_cache_size_fits_working_set(
    policy, expected_hits, expected_misses
):
    """cache_size=3 exactly fits the 3 unique experts, so after the first
    cold pass (3 misses) every subsequent access across all 5 policies
    hits."""
    report = simulate_all_policies(CYCLE_TRACE, cache_sizes=[3], num_experts=3)
    result = next(r for r in report.results if r.policy == policy)
    assert result.hits == expected_hits
    assert result.misses == expected_misses
    assert result.hit_rate == pytest.approx(0.8)


def test_lru_thrashes_when_cache_smaller_than_working_set():
    """cache_size=2 < 3 unique experts on a strict round-robin trace ->
    every access evicts the item that will be needed next: 0% hit rate."""
    report = simulate_all_policies(CYCLE_TRACE, cache_sizes=[2], num_experts=3)
    lru = next(r for r in report.results if r.policy == "LRU")
    assert lru.hits == 0
    assert lru.misses == 15
    assert lru.hit_rate == 0.0


def test_wtinylfu_partially_admits_under_thrashing_cache_size():
    """Unlike LRU/LFU/FIFO/ARC (all 0% at size=2 on this trace), the
    window+main-cache admission policy in W-TinyLFU picks up some hits."""
    report = simulate_all_policies(CYCLE_TRACE, cache_sizes=[2], num_experts=3)
    wtlfu = next(r for r in report.results if r.policy == "W-TinyLFU")
    assert wtlfu.hits == 4
    assert wtlfu.misses == 11


def test_recommended_policy_picks_first_tie_at_smallest_winning_size():
    """At size=3 all five policies tie at hit_rate=0.8; the strictly-greater
    ('>') comparison means only the first-iterated policy (LRU, per the
    dict insertion order in simulate_all_policies) is recorded, and a later
    size (5) with the same hit_rate never overrides it."""
    report = simulate_all_policies(CYCLE_TRACE, cache_sizes=[2, 3, 5], num_experts=3)
    assert report.recommended_policy == "LRU"
    assert report.recommended_size == 3
    assert report.recommended_hit_rate == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_empty_trace_yields_zero_hit_rate_everywhere_and_no_recommendation():
    report = simulate_all_policies([], cache_sizes=[2, 4], num_experts=0)

    assert report.num_accesses == 0
    assert report.unique_experts_accessed == 0
    assert len(report.results) == 2 * 5  # 2 sizes x 5 policies
    assert all(r.hits == 0 and r.misses == 0 and r.hit_rate == 0.0 for r in report.results)
    assert report.recommended_policy == ""
    assert report.recommended_size == 0
    assert report.recommended_hit_rate == 0.0


def test_cache_size_zero_raises_on_lfu():
    """BUG: cache_size=0 with a non-empty trace crashes inside _simulate_lfu
    (ValueError: min() arg is an empty sequence) because the eviction check
    `len(cache) >= size` is true even when the cache has zero entries."""
    with pytest.raises(ValueError):
        simulate_all_policies([5], cache_sizes=[0], num_experts=1)


def test_cache_size_zero_empty_trace_does_not_crash():
    """With an empty trace the per-policy loop body never executes, so the
    cache_size=0 bug never triggers."""
    report = simulate_all_policies([], cache_sizes=[0], num_experts=0)
    assert all(r.hits == 0 and r.misses == 0 for r in report.results)


# --------------------------------------------------------------------------- #
# Individual policy functions (the pure hit/miss engines)
# --------------------------------------------------------------------------- #


def test_simulate_lru_direct_hit_miss_counts():
    assert _simulate_lru(CYCLE_TRACE, 3) == (12, 3)
    assert _simulate_lru(CYCLE_TRACE, 2) == (0, 15)


def test_simulate_fifo_direct_hit_miss_counts():
    assert _simulate_fifo(CYCLE_TRACE, 3) == (12, 3)


def test_simulate_lfu_direct_hit_miss_counts():
    assert _simulate_lfu(CYCLE_TRACE, 3) == (12, 3)


def test_simulate_arc_direct_hit_miss_counts():
    assert _simulate_arc(CYCLE_TRACE, 3) == (12, 3)


def test_simulate_wtinylfu_direct_hit_miss_counts():
    assert _simulate_wtinylfu(CYCLE_TRACE, 3) == (12, 3)


def test_simulate_lru_single_element_trace_is_a_cold_miss():
    assert _simulate_lru([42], 10) == (0, 1)


def test_simulate_lru_empty_trace_is_zero_zero():
    assert _simulate_lru([], 10) == (0, 0)


# --------------------------------------------------------------------------- #
# format_cache_report() — smoke test
# --------------------------------------------------------------------------- #


def test_format_cache_report_contains_recommendation_and_grid():
    report = simulate_all_policies(CYCLE_TRACE, cache_sizes=[2, 3], num_experts=3)
    text = format_cache_report(report)

    assert "Cache Policy Simulation" in text
    assert "Recommended:" in text
    assert "LRU" in text
    assert "W-TinyLFU" in text


def test_format_cache_report_on_empty_results():
    report = CacheSimReport()
    text = format_cache_report(report)
    assert "Cache Policy Simulation" in text
    assert "Total Accesses:    0" in text
