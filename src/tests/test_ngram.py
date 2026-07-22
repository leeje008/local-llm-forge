"""Characterization tests for forge.engine.speculative.ngram.NGramDrafter.

Pins CURRENT behavior of the pure-Python n-gram self-speculative drafter
(no mlx, no filesystem, no network — token-history based prediction only).

Noted-but-not-fixed behavior: ``table_size`` only blocks the *table* from
gaining new keys once at capacity (``if key not in self._table: ... continue``
skips the update entirely for a would-be-new key). An *existing* key keeps
absorbing new continuations without limit even after the table is "full" by
key count — see ``test_table_size_cap_blocks_new_keys_but_not_existing_key_growth``.
This suite pins that behavior rather than treating it as a bug to fix.
"""

from __future__ import annotations

from forge.engine.speculative.ngram import NGramDrafter

# A 3-token cycle repeated 3x, used across several tests to build a
# deterministic, non-trivial n-gram table.
CYCLE_OBSERVATIONS = [1, 2, 3, 1, 2, 3, 1, 2, 3]


def _observe_cycle(drafter: NGramDrafter, tokens: list[int] = CYCLE_OBSERVATIONS) -> None:
    for t in tokens:
        drafter.observe(t)


# --------------------------------------------------------------------------- #
# __init__ / reset
# --------------------------------------------------------------------------- #


def test_default_construction():
    d = NGramDrafter()
    assert d.n == 3
    assert d.max_draft == 5
    assert d.table_size == 65536
    assert d._table == {}
    assert d._history == []


def test_reset_clears_table_and_history():
    d = NGramDrafter(n=2, max_draft=3)
    _observe_cycle(d)
    assert d._table  # non-empty before reset
    assert d._history

    d.reset()

    assert d._table == {}
    assert d._history == []
    assert d.draft() == []
    assert d.get_stats() == {
        "ngram_order": 2,
        "table_keys": 0,
        "total_entries": 0,
        "history_len": 0,
    }


# --------------------------------------------------------------------------- #
# observe() — n-gram table construction
# --------------------------------------------------------------------------- #


def test_observe_builds_table_for_repeating_cycle():
    """Nine observes of the 3-token cycle [1,2,3] with n=2 produce this exact
    table (verified against the running implementation, not hand-derived)."""
    d = NGramDrafter(n=2, max_draft=3)
    _observe_cycle(d)

    assert d._table == {
        (1,): {2: 3},
        (2,): {3: 3},
        (1, 2): {3: 3},
        (3,): {1: 2},
        (2, 3): {1: 2},
        (3, 1): {2: 2},
    }
    assert d._history == CYCLE_OBSERVATIONS


def test_observe_single_token_does_not_populate_table():
    """The first observe() only appends to history; no n-gram window exists
    yet because ``len(history) > order`` is false for every order."""
    d = NGramDrafter()
    d.observe(5)
    assert d._table == {}
    assert d._history == [5]


def test_get_stats_reflects_table_and_history():
    d = NGramDrafter(n=2, max_draft=3)
    _observe_cycle(d)

    assert d.get_stats() == {
        "ngram_order": 2,
        "table_keys": 6,
        "total_entries": 6,
        "history_len": 9,
    }


# --------------------------------------------------------------------------- #
# draft() — happy path + no-match edge cases
# --------------------------------------------------------------------------- #


def test_draft_returns_empty_list_with_no_history():
    d = NGramDrafter()
    assert d.draft() == []
    assert d.draft(None) == []
    assert d.draft([]) == []


def test_draft_returns_empty_when_single_observation_has_no_continuation():
    d = NGramDrafter()
    d.observe(5)
    assert d.draft() == []


def test_draft_predicts_cycle_continuation_after_repeated_observation():
    """After the drafter has seen the 3-cycle [1,2,3] repeat 3x, drafting off
    the internal history predicts the next cycle steps: 1 -> 2 -> 3."""
    d = NGramDrafter(n=2, max_draft=3)
    _observe_cycle(d)

    assert d.draft() == [1, 2, 3]


def test_draft_with_explicit_context_overrides_internal_history():
    d = NGramDrafter(n=2, max_draft=3)
    _observe_cycle(d)

    assert d.draft([1, 2]) == [3, 1, 2]


def test_draft_returns_empty_for_unmatched_context():
    d = NGramDrafter(n=2, max_draft=3)
    _observe_cycle(d)

    assert d.draft([99]) == []


# --------------------------------------------------------------------------- #
# Edge: max_draft cap
# --------------------------------------------------------------------------- #


def test_draft_length_capped_at_max_draft():
    d = NGramDrafter(n=2, max_draft=2)
    _observe_cycle(d, [7, 8, 9] * 5)

    result = d.draft()
    assert result == [7, 8]
    assert len(result) == d.max_draft


# --------------------------------------------------------------------------- #
# Edge: table_size cap behavior
# --------------------------------------------------------------------------- #


def test_table_size_cap_blocks_new_keys_but_not_existing_key_growth():
    """With table_size=1 (n=1), only the first n-gram key observed is ever
    inserted. Later observes that would create a *new* key are silently
    dropped, but once a key exists it keeps accumulating new continuations
    even after the table is at capacity — see module docstring."""
    d = NGramDrafter(n=1, table_size=1)

    d.observe(1)
    assert d._table == {}

    d.observe(2)  # creates key (1,) -> {2: 1}; table now at capacity (1 key)
    assert d._table == {(1,): {2: 1}}

    d.observe(3)  # would create new key (2,); table full -> dropped
    assert d._table == {(1,): {2: 1}}

    d.observe(1)  # would create new key (3,); table full -> dropped
    assert d._table == {(1,): {2: 1}}

    d.observe(9)  # history ends ...,1,9 -> key (1,) already exists -> updates
    assert d._table == {(1,): {2: 1, 9: 1}}
