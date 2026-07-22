"""Characterization tests for forge.engine.kv_cache.estimation.

These pin CURRENT behavior. ``estimate_max_context`` now clamps a negative
``available_memory_mb`` to 0 (fixed). One documented quirk remains pinned as-is:

- ``recommend_kv_optimization``'s percentage threshold checks are strict ``>``
  (not ``>=``), so a context length exactly at a threshold does not trigger
  the corresponding recommendation.
"""

from __future__ import annotations

import pytest

from forge.engine.kv_cache.base import KVCompressionMethod
from forge.engine.kv_cache.estimation import (
    estimate_kv_cache_memory,
    estimate_max_context,
    format_kv_report,
    recommend_kv_optimization,
)

# --------------------------------------------------------------------------- #
# estimate_kv_cache_memory()
# --------------------------------------------------------------------------- #


def test_estimate_kv_cache_memory_happy_path_fp16():
    mb = estimate_kv_cache_memory(num_layers=28, num_kv_heads=4, head_dim=128, seq_len=4096)
    assert mb == pytest.approx(224.0)


def test_estimate_kv_cache_memory_fp8_halves_fp16():
    fp16 = estimate_kv_cache_memory(28, 4, 128, 4096)
    fp8 = estimate_kv_cache_memory(28, 4, 128, 4096, compression=KVCompressionMethod.FP8)
    assert fp8 == pytest.approx(fp16 / 2)
    assert fp8 == pytest.approx(112.0)


def test_estimate_kv_cache_memory_turbo_compression():
    turbo = estimate_kv_cache_memory(28, 4, 128, 4096, compression=KVCompressionMethod.TURBO)
    assert turbo == pytest.approx(40.6)


def test_estimate_kv_cache_memory_zero_layers_is_zero():
    assert estimate_kv_cache_memory(0, 4, 128, 4096) == 0.0


def test_estimate_kv_cache_memory_zero_heads_is_zero():
    assert estimate_kv_cache_memory(28, 0, 128, 4096) == 0.0


# --------------------------------------------------------------------------- #
# estimate_max_context()
# --------------------------------------------------------------------------- #


def test_estimate_max_context_happy_path():
    tokens = estimate_max_context(28, 4, 128, available_memory_mb=1024)
    assert tokens == 18724


def test_estimate_max_context_turbo_compression_extends_context():
    tokens = estimate_max_context(
        28, 4, 128, available_memory_mb=1024, compression=KVCompressionMethod.TURBO
    )
    assert tokens == 103307


def test_estimate_max_context_zero_layers_short_circuits_to_zero():
    # per_token_bytes == 0 -> explicit early return, avoids a ZeroDivisionError.
    assert estimate_max_context(0, 4, 128, available_memory_mb=1024) == 0


def test_estimate_max_context_zero_available_memory_is_zero():
    assert estimate_max_context(28, 4, 128, available_memory_mb=0) == 0


def test_estimate_max_context_negative_available_memory_clamps_to_zero():
    # Fixed: a negative memory budget is clamped to 0 rather than producing a
    # nonsensical negative token count.
    assert estimate_max_context(28, 4, 128, available_memory_mb=-100) == 0


# --------------------------------------------------------------------------- #
# recommend_kv_optimization()
# --------------------------------------------------------------------------- #


def test_recommend_kv_optimization_small_kv_only_gqa_note():
    recs = recommend_kv_optimization(
        model_params_b=7.6, context_length=4096, available_memory_gb=38.0
    )
    assert set(recs.keys()) == {"gqa_note"}


def test_recommend_kv_optimization_context_exactly_at_h2o_threshold_is_excluded():
    # Threshold check is strict `context_length > 8192`, so exactly 8192 does
    # NOT trigger h2o_eviction (pinned boundary behavior).
    recs = recommend_kv_optimization(7.6, context_length=8192, available_memory_gb=38.0)
    assert "h2o_eviction" not in recs
    assert recs["turbo_kv"]["priority"] == "medium"


def test_recommend_kv_optimization_context_just_above_h2o_threshold_triggers_it():
    recs = recommend_kv_optimization(7.6, context_length=8193, available_memory_gb=38.0)
    assert recs["h2o_eviction"]["enabled"] is True
    assert recs["h2o_eviction"]["budget_ratio"] == 0.2
    assert "ada_kv" not in recs
    assert "sliding_window" not in recs
    assert recs["turbo_kv"]["priority"] == "medium"


def test_recommend_kv_optimization_long_context_triggers_ada_kv_and_sliding_window():
    recs = recommend_kv_optimization(7.6, context_length=16385, available_memory_gb=38.0)
    assert recs["ada_kv"]["enabled"] is True
    assert recs["sliding_window"] == {
        "enabled": True,
        "window_size": 8192,
        "reason": "Context 16,385 — sliding window limits KV growth",
    }


def test_recommend_kv_optimization_high_kv_pct_triggers_all_recommendations_with_high_priority():
    recs = recommend_kv_optimization(7.6, context_length=32768, available_memory_gb=6.0)
    assert set(recs.keys()) == {
        "turbo_kv", "fp8_kv", "h2o_eviction", "ada_kv", "sliding_window", "gqa_note",
    }
    assert recs["turbo_kv"]["priority"] == "high"
    assert recs["fp8_kv"]["savings_gb"] == pytest.approx(12.45184)


def test_recommend_kv_optimization_gqa_note_always_present():
    recs = recommend_kv_optimization(0.0, context_length=1, available_memory_gb=1.0)
    assert recs["gqa_note"] == {
        "enabled": True,
        "reason": "GQA reduces KV cache proportionally to head ratio",
    }


# --------------------------------------------------------------------------- #
# format_kv_report()
# --------------------------------------------------------------------------- #


def test_format_kv_report_contains_expected_markers():
    report = format_kv_report(28, 4, 128, context_lengths=[2048, 200000])
    assert isinstance(report, str)
    assert "KV Cache Analysis" in report
    assert "Layers: 28, KV Heads: 4, Head Dim: 128" in report
    assert "TurboQ = TurboQuant 3-bit VQ (~5.5x compression)" in report
    assert "112 MB" in report  # FP16 column for context=2048
    assert "10.7 GB" in report  # FP16 column for context=200000 (>= 1024 MB -> GB format)


def test_format_kv_report_defaults_to_seven_standard_context_lengths():
    report = format_kv_report(28, 4, 128)
    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        assert f"{ctx:,}" in report
