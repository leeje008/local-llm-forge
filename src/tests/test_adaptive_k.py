"""Characterization tests for forge.engine.speculative.adaptive_k.

Pins CURRENT behavior of AdaptiveKConfig / AdaptiveKController: pure Python,
in-memory rolling-window acceptance-rate tracking used to grow/shrink the
speculative draft length K. No mlx, no filesystem, no network.
"""

from __future__ import annotations

import pytest

from forge.engine.speculative.adaptive_k import AdaptiveKConfig, AdaptiveKController


def test_config_defaults():
    cfg = AdaptiveKConfig()
    assert cfg.initial_k == 3
    assert cfg.min_k == 1
    assert cfg.max_k == 10
    assert cfg.target_acceptance == pytest.approx(0.7)
    assert cfg.increase_threshold == pytest.approx(0.8)
    assert cfg.decrease_threshold == pytest.approx(0.4)
    assert cfg.window_size == 10
    assert cfg.step_size == 1


def test_controller_uses_default_config_when_none_given():
    ctrl = AdaptiveKController()
    assert ctrl.current_k == 3
    assert ctrl.config == AdaptiveKConfig()


# --------------------------------------------------------------------------- #
# record_round() — increase / decrease / dead zone
# --------------------------------------------------------------------------- #


def test_record_round_increases_k_under_high_acceptance():
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=3))
    for expected_k in (4, 5, 6, 7, 8):
        ctrl.record_round(drafted=10, accepted=9)  # rate 0.9 >= 0.8 threshold
        assert ctrl.current_k == expected_k


def test_record_round_decreases_k_under_low_acceptance():
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=5))
    for expected_k in (4, 3, 2, 1, 1):
        ctrl.record_round(drafted=10, accepted=1)  # rate 0.1 <= 0.4 threshold
        assert ctrl.current_k == expected_k


@pytest.mark.parametrize(
    "accepted, expected_k",
    [
        (6, 4),  # rate 0.6: strictly between thresholds -> no change (dead zone)
        (8, 5),  # rate 0.8: exactly == increase_threshold -> increases
        (4, 3),  # rate 0.4: exactly == decrease_threshold -> decreases
    ],
)
def test_record_round_threshold_boundaries(accepted, expected_k):
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=4))
    ctrl.record_round(drafted=10, accepted=accepted)
    assert ctrl.current_k == expected_k


def test_record_round_clamps_at_max_k():
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=9, max_k=10, step_size=1))
    for _ in range(5):
        ctrl.record_round(drafted=10, accepted=10)
    assert ctrl.current_k == 10


def test_record_round_clamps_at_min_k():
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=2, min_k=1, step_size=1))
    for _ in range(5):
        ctrl.record_round(drafted=10, accepted=0)
    assert ctrl.current_k == 1


def test_record_round_ignores_zero_drafted():
    """drafted == 0 returns immediately: no history entry recorded, no K
    change, even though accepted == drafted (0/0) would otherwise divide by
    zero."""
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=3))
    ctrl.record_round(drafted=0, accepted=0)
    assert ctrl.current_k == 3
    assert ctrl._history == []
    assert ctrl.get_stats() == {
        "current_k": 3,
        "avg_acceptance_rate": 0.0,
        "window_size": 0,
    }


# --------------------------------------------------------------------------- #
# rolling window_size
# --------------------------------------------------------------------------- #


def test_history_rolls_off_beyond_window_size():
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=3, window_size=3))
    rounds = [(10, 10), (10, 10), (10, 10), (10, 0), (10, 0)]
    expected_k = [4, 5, 6, 6, 5]
    expected_history = [
        [1.0],
        [1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    for (drafted, accepted), k, hist in zip(rounds, expected_k, expected_history):
        ctrl.record_round(drafted, accepted)
        assert ctrl.current_k == k
        assert ctrl._history == pytest.approx(hist)


# --------------------------------------------------------------------------- #
# reset() / get_stats()
# --------------------------------------------------------------------------- #


def test_reset_restores_initial_k_and_clears_history():
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=3))
    for _ in range(5):
        ctrl.record_round(drafted=10, accepted=9)
    assert ctrl.current_k != 3

    ctrl.reset()

    assert ctrl.current_k == 3
    assert ctrl._history == []
    assert ctrl.get_stats() == {
        "current_k": 3,
        "avg_acceptance_rate": 0.0,
        "window_size": 0,
    }


def test_get_stats_reports_rounded_average_and_window_len():
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=3, window_size=3))
    ctrl.record_round(drafted=10, accepted=10)  # rate 1.0
    ctrl.record_round(drafted=10, accepted=10)  # rate 1.0
    ctrl.record_round(drafted=10, accepted=0)  # rate 0.0
    stats = ctrl.get_stats()
    assert stats["avg_acceptance_rate"] == pytest.approx(0.667, abs=1e-3)
    assert stats["window_size"] == 3
