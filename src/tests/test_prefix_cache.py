"""Characterization tests for the radix-tree prefix cache.

Pins the CURRENT behavior of ``forge.engine.prefix_cache`` (SGLang-inspired
RadixAttention-style KV cache sharing). No mlx, network, or disk I/O — the
module operates purely on Python lists/tuples of int token ids and opaque
``kv_state`` placeholders.

Known quirk pinned here (not fixed): a lookup for a token sequence that
lands EXACTLY on an internal split node (a branch point created when two
inserted sequences diverge) returns a MISS even though that prefix is
structurally present in the tree, because cache entries are only stored at
insertion endpoints, never at split nodes. See
``test_lookup_on_internal_split_node_is_a_miss``.
"""

from __future__ import annotations

import pytest

from forge.engine.prefix_cache import (
    CacheEntry,
    PrefixMatch,
    RadixPrefixCache,
    format_prefix_cache_report,
)


def make_cache(**kwargs) -> RadixPrefixCache:
    return RadixPrefixCache(**kwargs)


# --------------------------------------------------------------------------- #
# insert()
# --------------------------------------------------------------------------- #


def test_insert_returns_true_and_updates_stats():
    cache = make_cache()
    ok = cache.insert([1, 2, 3], kv_state="kv-a", memory_bytes=100)
    assert ok is True
    assert cache.stats.total_entries == 1
    assert cache.stats.total_memory_bytes == 100


def test_insert_empty_sequence_returns_false_and_is_a_noop():
    cache = make_cache()
    ok = cache.insert([], kv_state="kv-empty")
    assert ok is False
    assert cache.stats.total_entries == 0
    assert cache.get_all_entries() == []


def test_insert_accepts_list_or_tuple():
    cache = make_cache()
    assert cache.insert((1, 2), kv_state="a") is True
    assert cache.insert([3, 4], kv_state="b") is True
    assert cache.stats.total_entries == 2


def test_insert_exact_duplicate_replaces_entry_without_growing_count():
    cache = make_cache()
    cache.insert([1, 2, 3], kv_state="v1", memory_bytes=100)
    cache.insert([1, 2, 3], kv_state="v2", memory_bytes=50)

    # Replacing at an exact-match node decrements then the outer insert()
    # re-increments — net effect: count stays at 1, memory reflects both
    # deltas (old bytes subtracted internally, new bytes added by insert()).
    assert cache.stats.total_entries == 1
    match = cache.find_longest_prefix([1, 2, 3])
    assert match is not None
    assert match.kv_state == "v2"


# --------------------------------------------------------------------------- #
# find_longest_prefix() — hit / miss
# --------------------------------------------------------------------------- #


def test_lookup_exact_match_is_a_hit():
    cache = make_cache()
    cache.insert([1, 2, 3], kv_state="kv-a")

    match = cache.find_longest_prefix([1, 2, 3])

    assert isinstance(match, PrefixMatch)
    assert match.kv_state == "kv-a"
    assert match.prefix_len == 3
    assert match.tokens_saved == 3
    assert cache.stats.cache_hits == 1
    assert cache.stats.cache_misses == 0
    assert cache.stats.total_tokens_saved == 3


def test_lookup_longer_sequence_matches_stored_prefix():
    cache = make_cache()
    cache.insert([1, 2, 3], kv_state="kv-a")

    match = cache.find_longest_prefix([1, 2, 3, 4, 5])

    assert match is not None
    assert match.prefix_len == 3
    assert match.kv_state == "kv-a"


def test_lookup_no_matching_first_token_is_a_miss():
    cache = make_cache()
    cache.insert([1, 2, 3], kv_state="kv-a")

    match = cache.find_longest_prefix([9, 9, 9])

    assert match is None
    assert cache.stats.cache_misses == 1
    assert cache.stats.cache_hits == 0


def test_lookup_on_empty_cache_is_a_miss():
    cache = make_cache()
    match = cache.find_longest_prefix([1, 2, 3])
    assert match is None
    assert cache.stats.total_lookups == 1
    assert cache.stats.cache_misses == 1


def test_lookup_empty_sequence_is_a_miss_even_with_entries():
    cache = make_cache()
    cache.insert([1, 2, 3], kv_state="kv-a")

    match = cache.find_longest_prefix([])

    assert match is None
    assert cache.stats.total_lookups == 1
    assert cache.stats.cache_misses == 1


def test_touch_updates_entry_access_count_on_hit():
    cache = make_cache()
    cache.insert([1, 2], kv_state="kv-a")
    cache.find_longest_prefix([1, 2])
    cache.find_longest_prefix([1, 2])

    entries = cache.get_all_entries()
    assert len(entries) == 1
    assert entries[0].access_count == 2


def test_lookup_on_internal_split_node_is_a_miss():
    """Two sequences sharing a [1, 2] prefix create a split node at [1, 2]
    with no cache_entry of its own (entries live only at the two leaves).
    A lookup for exactly [1, 2] therefore misses even though it is a valid
    structural prefix of both stored sequences — current (quirky) behavior."""
    cache = make_cache()
    cache.insert([1, 2, 3, 4], kv_state="kv-a")
    cache.insert([1, 2, 5, 6], kv_state="kv-b")

    match = cache.find_longest_prefix([1, 2])

    assert match is None
    assert cache.stats.cache_misses == 1


# --------------------------------------------------------------------------- #
# Shared-prefix branching
# --------------------------------------------------------------------------- #


def test_shared_prefix_branch_both_full_sequences_resolve_independently():
    cache = make_cache()
    cache.insert([1, 2, 3, 4], kv_state="kv-a")
    cache.insert([1, 2, 5, 6], kv_state="kv-b")

    match_a = cache.find_longest_prefix([1, 2, 3, 4])
    match_b = cache.find_longest_prefix([1, 2, 5, 6])

    assert match_a is not None and match_a.kv_state == "kv-a" and match_a.prefix_len == 4
    assert match_b is not None and match_b.kv_state == "kv-b" and match_b.prefix_len == 4
    assert cache.stats.total_entries == 2


def test_shared_prefix_branch_diverging_lookup_falls_back_to_no_match():
    cache = make_cache()
    cache.insert([1, 2, 3, 4], kv_state="kv-a")
    cache.insert([1, 2, 5, 6], kv_state="kv-b")

    # Diverges after [1, 2] into a third direction never inserted.
    match = cache.find_longest_prefix([1, 2, 9, 9])

    assert match is None


def test_insert_reversed_order_produces_same_split_result():
    """Splitting is symmetric regardless of insertion order."""
    cache_a = make_cache()
    cache_a.insert([1, 2, 3, 4], kv_state="kv-a")
    cache_a.insert([1, 2, 5, 6], kv_state="kv-b")

    cache_b = make_cache()
    cache_b.insert([1, 2, 5, 6], kv_state="kv-b")
    cache_b.insert([1, 2, 3, 4], kv_state="kv-a")

    for cache in (cache_a, cache_b):
        m1 = cache.find_longest_prefix([1, 2, 3, 4])
        m2 = cache.find_longest_prefix([1, 2, 5, 6])
        assert m1.kv_state == "kv-a"
        assert m2.kv_state == "kv-b"


def test_insert_shorter_sequence_after_longer_one_splits_correctly():
    cache = make_cache()
    cache.insert([1, 2, 3, 4], kv_state="kv-long")
    cache.insert([1, 2], kv_state="kv-short")

    match_short = cache.find_longest_prefix([1, 2])
    match_long = cache.find_longest_prefix([1, 2, 3, 4])

    assert match_short is not None and match_short.kv_state == "kv-short"
    assert match_short.prefix_len == 2
    assert match_long is not None and match_long.kv_state == "kv-long"
    assert match_long.prefix_len == 4


# --------------------------------------------------------------------------- #
# Eviction
# --------------------------------------------------------------------------- #


def test_eviction_triggered_by_max_entries_removes_lru(monkeypatch):
    # Deterministic clock so LRU ordering is never a race.
    clock = iter(range(1, 100))
    monkeypatch.setattr(
        "forge.engine.prefix_cache.time.monotonic", lambda: next(clock)
    )

    cache = make_cache(max_memory_mb=2048.0, max_entries=2)
    cache.insert([1], kv_state="a", memory_bytes=10)
    cache.insert([2], kv_state="b", memory_bytes=10)
    # Third insert exceeds max_entries=2 -> evicts entry [1] (oldest access).
    cache.insert([3], kv_state="c", memory_bytes=10)

    assert cache.stats.evictions == 1
    assert cache.stats.total_entries == 2
    assert cache.find_longest_prefix([1]) is None
    assert cache.find_longest_prefix([2]) is not None
    assert cache.find_longest_prefix([3]) is not None


def test_eviction_triggered_by_memory_budget(monkeypatch):
    clock = iter(range(1, 100))
    monkeypatch.setattr(
        "forge.engine.prefix_cache.time.monotonic", lambda: next(clock)
    )

    # Budget fits exactly two 400_000-byte entries (1 MiB == 1_048_576 B).
    cache = make_cache(max_memory_mb=1.0, max_entries=1000)
    cache.insert([1], kv_state="a", memory_bytes=400_000)
    cache.insert([2], kv_state="b", memory_bytes=400_000)
    cache.insert([3], kv_state="c", memory_bytes=400_000)

    assert cache.stats.evictions == 1
    assert cache.stats.total_entries == 2
    assert cache.stats.total_memory_bytes == 800_000
    assert cache.find_longest_prefix([1]) is None


def test_eviction_returns_false_when_cache_empty_and_single_entry_too_big():
    """First insert already exceeds budget and there is nothing to evict ->
    insert fails outright (current behavior: no partial admission)."""
    cache = make_cache(max_memory_mb=0.0001, max_entries=1000)
    ok = cache.insert([1, 2, 3], kv_state="a", memory_bytes=10_000)
    assert ok is False
    assert cache.stats.total_entries == 0


def test_zero_memory_bytes_insert_never_triggers_eviction_by_budget():
    """memory_bytes=0 (the default) skips the budget-check branch entirely,
    so inserts can exceed max_entries without eviction unless a nonzero
    memory_bytes insert later re-triggers the check."""
    cache = make_cache(max_memory_mb=2048.0, max_entries=1)
    cache.insert([1], kv_state="a")
    cache.insert([2], kv_state="b")
    assert cache.stats.evictions == 0
    assert cache.stats.total_entries == 2


# --------------------------------------------------------------------------- #
# clear() / get_all_entries()
# --------------------------------------------------------------------------- #


def test_clear_resets_tree_and_stats():
    cache = make_cache()
    cache.insert([1, 2], kv_state="a", memory_bytes=10)
    cache.find_longest_prefix([1, 2])

    cache.clear()

    assert cache.get_all_entries() == []
    assert cache.stats.total_entries == 0
    assert cache.stats.total_lookups == 0
    assert cache.stats.cache_hits == 0


def test_get_all_entries_returns_one_per_inserted_sequence():
    cache = make_cache()
    cache.insert([1, 2], kv_state="a")
    cache.insert([3, 4], kv_state="b")
    cache.insert([1, 2, 5], kv_state="c")

    entries = cache.get_all_entries()
    assert len(entries) == 3
    assert {e.kv_state for e in entries} == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# Stats properties
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "hits, lookups, expected",
    [
        (0, 0, 0.0),   # no lookups yet -> guarded by max(total_lookups, 1)
        (3, 4, 0.75),
        (4, 4, 1.0),
    ],
)
def test_hit_rate_property(hits, lookups, expected):
    cache = make_cache()
    cache.stats.cache_hits = hits
    cache.stats.total_lookups = lookups
    assert cache.stats.hit_rate == expected


def test_memory_mb_property_converts_bytes():
    cache = make_cache()
    cache.stats.total_memory_bytes = 2 * 1024 * 1024
    assert cache.stats.memory_mb == 2.0


def test_cache_entry_touch_updates_access_count_and_timestamp():
    entry = CacheEntry(tokens=(1, 2), kv_state="x")
    assert entry.access_count == 0
    entry.touch()
    entry.touch()
    assert entry.access_count == 2
    assert entry.last_accessed > 0.0


# --------------------------------------------------------------------------- #
# format_prefix_cache_report() — smoke test
# --------------------------------------------------------------------------- #


def test_format_prefix_cache_report_on_empty_cache():
    cache = make_cache()
    report = format_prefix_cache_report(cache)
    assert "Prefix Cache Statistics" in report
    assert "Entries:       0" in report


def test_format_prefix_cache_report_lists_entries_after_inserts():
    cache = make_cache()
    cache.insert([1, 2, 3], kv_state="a", memory_bytes=1024)
    cache.find_longest_prefix([1, 2, 3])

    report = format_prefix_cache_report(cache)

    assert "Entries:       1" in report
    assert "Cached Prefixes:" in report
    assert "1 hits" in report
