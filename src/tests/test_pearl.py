"""Characterization tests for forge.engine.speculative.pearl.

Pins CURRENT behavior of PEARLConfig / PEARLStats / PEARLScheduler /
DraftPlan (pure Python state machine for pre-verify / post-verify overlap
planning). No mlx, no filesystem, no network.

Noted-but-not-fixed: ``plan_draft_round``'s post-verify gate is a strict
``recent_acceptance_rate > 0.5`` (not ``>=``), so a rate of exactly 0.5 does
NOT enable post-verify extra drafting. Pinned as-is in
``test_plan_draft_round_post_verify_gate``.
"""

from __future__ import annotations

import pytest

from forge.engine.speculative.pearl import (
    DraftPlan,
    PEARLConfig,
    PEARLScheduler,
    PEARLStats,
)


def test_config_defaults():
    cfg = PEARLConfig()
    assert cfg.enable_pre_verify is True
    assert cfg.enable_post_verify is True
    assert cfg.post_verify_tokens == 2


def test_stats_defaults():
    stats = PEARLStats()
    assert stats.total_steps == 0
    assert stats.pre_verify_hits == 0
    assert stats.post_verify_extra_accepted == 0
    assert stats.total_draft_tokens == 0
    assert stats.total_accepted_tokens == 0


# --------------------------------------------------------------------------- #
# PEARLStats properties — divide-by-zero guards
# --------------------------------------------------------------------------- #


def test_acceptance_rate_zero_steps_guard():
    stats = PEARLStats()
    assert stats.acceptance_rate == pytest.approx(0.0)


def test_pre_verify_hit_rate_zero_steps_guard():
    stats = PEARLStats()
    assert stats.pre_verify_hit_rate == pytest.approx(0.0)


def test_acceptance_rate_computes_ratio():
    stats = PEARLStats(total_draft_tokens=4, total_accepted_tokens=3)
    assert stats.acceptance_rate == pytest.approx(0.75)


def test_pre_verify_hit_rate_computes_ratio():
    stats = PEARLStats(total_steps=4, pre_verify_hits=1)
    assert stats.pre_verify_hit_rate == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# plan_draft_round()
# --------------------------------------------------------------------------- #


def test_plan_draft_round_default_config_full_overlap():
    sched = PEARLScheduler()
    plan = sched.plan_draft_round(base_draft_len=4, recent_acceptance_rate=0.9)
    assert plan == DraftPlan(
        main_draft_len=4,
        pre_verify_first=True,
        post_verify_extra=2,
        total_candidates=6,
    )


def test_plan_draft_round_increments_total_steps():
    sched = PEARLScheduler()
    assert sched.stats.total_steps == 0
    sched.plan_draft_round(base_draft_len=4, recent_acceptance_rate=0.9)
    sched.plan_draft_round(base_draft_len=4, recent_acceptance_rate=0.9)
    assert sched.stats.total_steps == 2


def test_plan_draft_round_pre_verify_requires_base_draft_len_above_one():
    sched = PEARLScheduler()
    plan = sched.plan_draft_round(base_draft_len=1, recent_acceptance_rate=0.9)
    assert plan.pre_verify_first is False
    assert plan.main_draft_len == 1


@pytest.mark.parametrize(
    "recent_acceptance_rate, expected_post_extra",
    [
        (0.5, 0),  # exactly at the gate: NOT enabled (strict >, see module docstring)
        (0.51, 2),  # just above the gate: enabled
        (0.0, 0),
        (1.0, 2),
    ],
)
def test_plan_draft_round_post_verify_gate(recent_acceptance_rate, expected_post_extra):
    sched = PEARLScheduler()
    plan = sched.plan_draft_round(
        base_draft_len=4, recent_acceptance_rate=recent_acceptance_rate
    )
    assert plan.post_verify_extra == expected_post_extra
    assert plan.total_candidates == 4 + expected_post_extra


def test_plan_draft_round_pre_verify_disabled_by_config():
    sched = PEARLScheduler(PEARLConfig(enable_pre_verify=False))
    plan = sched.plan_draft_round(base_draft_len=4, recent_acceptance_rate=0.99)
    assert plan.pre_verify_first is False
    assert plan.post_verify_extra == 2  # post-verify unaffected


def test_plan_draft_round_post_verify_disabled_by_config():
    sched = PEARLScheduler(PEARLConfig(enable_post_verify=False))
    plan = sched.plan_draft_round(base_draft_len=4, recent_acceptance_rate=0.99)
    assert plan.post_verify_extra == 0
    assert plan.total_candidates == 4
    assert plan.pre_verify_first is True  # pre-verify unaffected


# --------------------------------------------------------------------------- #
# record_result()
# --------------------------------------------------------------------------- #


def test_record_result_accumulates_stats():
    sched = PEARLScheduler()
    sched.record_result(
        drafted=4, accepted=3, pre_verify_accepted=True, post_verify_accepted=2
    )
    assert sched.stats.total_draft_tokens == 4
    assert sched.stats.total_accepted_tokens == 3
    assert sched.stats.pre_verify_hits == 1
    assert sched.stats.post_verify_extra_accepted == 2
    assert sched.stats.acceptance_rate == pytest.approx(0.75)


def test_record_result_defaults_do_not_bump_pre_verify_or_post_verify():
    sched = PEARLScheduler()
    sched.record_result(drafted=4, accepted=4)
    assert sched.stats.pre_verify_hits == 0
    assert sched.stats.post_verify_extra_accepted == 0


def test_record_result_accumulates_across_multiple_calls():
    sched = PEARLScheduler()
    sched.plan_draft_round(base_draft_len=4, recent_acceptance_rate=0.9)  # total_steps=1
    sched.record_result(
        drafted=4, accepted=3, pre_verify_accepted=True, post_verify_accepted=2
    )
    sched.record_result(drafted=4, accepted=4)

    assert sched.stats.total_steps == 1
    assert sched.stats.total_draft_tokens == 8
    assert sched.stats.total_accepted_tokens == 7
    assert sched.stats.pre_verify_hits == 1
    assert sched.stats.post_verify_extra_accepted == 2
    assert sched.stats.acceptance_rate == pytest.approx(0.875)
    assert sched.stats.pre_verify_hit_rate == pytest.approx(1.0)  # 1 hit / 1 step


# --------------------------------------------------------------------------- #
# reset()
# --------------------------------------------------------------------------- #


def test_reset_replaces_stats_with_fresh_instance():
    sched = PEARLScheduler()
    sched.plan_draft_round(base_draft_len=4, recent_acceptance_rate=0.9)
    sched.record_result(drafted=4, accepted=3)
    assert sched.stats.total_steps != 0

    sched.reset()

    assert sched.stats == PEARLStats()
