"""Characterization tests for forge.analyzer.memory_calculator.

These tests pin the CURRENT observed behavior of the memory budget math,
including edge cases that look questionable (see BUG REPORT in the PR/report
that accompanies this file). They intentionally do NOT assert "correct"
behavior where the implementation looks buggy -- they pin what it actually
does today so regressions are caught.
"""

from __future__ import annotations

import pytest

from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.memory_calculator import (
    calc_activation_memory,
    calc_kv_memory,
    calc_kv_per_token,
    calc_theoretical_tps,
    calc_weight_memory,
    calculate,
    format_report,
)
from forge.analyzer.model_inspector import ModelProfile

# --------------------------------------------------------------------------- #
# calc_weight_memory
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("quant", "expected_gb"),
    [
        ("fp16", 15.96),
        ("int8", 7.98),
        ("int4", 3.99),
        ("int3", 2.9925),
        ("int2", 1.995),
    ],
)
def test_calc_weight_memory_known_quants(dense_7b, quant, expected_gb):
    assert calc_weight_memory(dense_7b, quant) == pytest.approx(expected_gb)


def test_calc_weight_memory_unknown_quant_falls_back_to_0_5_bpp(dense_7b):
    """Unknown quant strings silently use BITS_TO_BYTES.get(quant, 0.5) --
    i.e. they're treated as int4-equivalent (0.5 bytes/param) with no error
    or warning. Same numeric result as int4."""
    assert calc_weight_memory(dense_7b, "q5_k_m") == pytest.approx(3.99)
    assert calc_weight_memory(dense_7b, "q5_k_m") == calc_weight_memory(dense_7b, "int4")


def test_calc_weight_memory_applies_5_percent_overhead(dense_7b):
    # fp16 without overhead would be 7.6 * 2.0 = 15.2 GB; with 5% overhead: 15.96
    raw = dense_7b.total_params_b * 2.0
    assert calc_weight_memory(dense_7b, "fp16") == pytest.approx(raw * 1.05)


def test_calc_weight_memory_zero_params_is_zero(tiny_model):
    assert calc_weight_memory(tiny_model, "fp16") == 0.0


# --------------------------------------------------------------------------- #
# calc_kv_per_token
# --------------------------------------------------------------------------- #


def test_calc_kv_per_token_gqa_model(dense_7b):
    # 2 (K+V) * 28 layers * 4 kv_heads * 128 head_dim * 2 bytes / 1e9
    assert calc_kv_per_token(dense_7b) == pytest.approx(5.7344e-05)


def test_calc_kv_per_token_dense_72b(dense_72b):
    assert calc_kv_per_token(dense_72b) == pytest.approx(0.00032768)


def test_calc_kv_per_token_falls_back_to_attention_heads_when_kv_heads_falsy():
    """num_kv_heads=0 is falsy, so `model.num_kv_heads or model.num_attention_heads`
    falls back to num_attention_heads -- effectively treating the model as MHA."""
    mha_like = ModelProfile(
        model_id="synthetic/mha-fallback",
        num_layers=12,
        num_attention_heads=16,
        num_kv_heads=0,
        head_dim=64,
    )
    assert calc_kv_per_token(mha_like) == pytest.approx(4.9152e-05)


def test_calc_kv_per_token_respects_custom_dtype_bytes(dense_7b):
    # fp32 kv cache (4 bytes) doubles the fp16-default (2 bytes) result
    default = calc_kv_per_token(dense_7b)
    fp32 = calc_kv_per_token(dense_7b, kv_dtype_bytes=4)
    assert fp32 == pytest.approx(default * 2)


def test_calc_kv_per_token_tiny_model_is_zero(tiny_model):
    assert calc_kv_per_token(tiny_model) == 0.0


# --------------------------------------------------------------------------- #
# calc_kv_memory
# --------------------------------------------------------------------------- #


def test_calc_kv_memory_scales_with_seq_len_and_batch(dense_7b):
    assert calc_kv_memory(dense_7b, seq_len=2048) == pytest.approx(0.117440512)
    assert calc_kv_memory(dense_7b, seq_len=100, batch=4) == pytest.approx(0.0229376)


def test_calc_kv_memory_zero_seq_len_is_zero(dense_7b):
    assert calc_kv_memory(dense_7b, seq_len=0) == 0.0


def test_calc_kv_memory_linear_in_batch(dense_7b):
    single = calc_kv_memory(dense_7b, seq_len=512, batch=1)
    quad = calc_kv_memory(dense_7b, seq_len=512, batch=4)
    assert quad == pytest.approx(single * 4)


# --------------------------------------------------------------------------- #
# calc_activation_memory
# --------------------------------------------------------------------------- #


def test_calc_activation_memory_decode_path_seq_len_1(dense_7b):
    # decode: base only, no quadratic attention term added (seq_len > 1 check excludes 1)
    assert calc_activation_memory(dense_7b, seq_len=1) == pytest.approx(0.000802816)


def test_calc_activation_memory_default_seq_len_matches_decode(dense_7b):
    assert calc_activation_memory(dense_7b) == pytest.approx(
        calc_activation_memory(dense_7b, seq_len=1)
    )


def test_calc_activation_memory_prefill_path_adds_quadratic_term(dense_7b):
    # prefill: base + seq_len^2 * num_attention_heads * 2 / 1e9
    assert calc_activation_memory(dense_7b, seq_len=2048) == pytest.approx(0.23568384)


def test_calc_activation_memory_tiny_model_is_zero(tiny_model):
    assert calc_activation_memory(tiny_model, seq_len=1) == 0.0
    assert calc_activation_memory(tiny_model, seq_len=2048) == 0.0


# --------------------------------------------------------------------------- #
# calculate
# --------------------------------------------------------------------------- #


def test_calculate_dense_7b_fits_on_m4_pro(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro)

    assert budget.usable_memory_gb == pytest.approx(38.0)
    assert budget.kv_per_token_gb == pytest.approx(5.7344e-05)
    assert budget.activation_base_gb == pytest.approx(0.000802816)
    assert budget.can_run is True
    assert budget.recommended_quant == "fp16"
    assert budget.recommended_context == 8192  # min(default_context, max_ctx)
    assert budget.recommended_memory_gb == pytest.approx(16.430564864)

    assert [e.quant for e in budget.estimates] == ["fp16", "int8", "int4", "int3", "int2"]
    # All quant levels fit on 38GB usable for a 7.6B model -- even int2's max_context
    # is capped at the model's own max_context (32768), not the much larger KV budget.
    for est in budget.estimates:
        assert est.fits_with_kv is True
        assert est.max_context == 32768

    fp16_est = budget.estimates[0]
    assert fp16_est.weight_memory_gb == pytest.approx(15.96)
    assert fp16_est.quality_pct == pytest.approx(100.0)
    assert fp16_est.total_memory_gb == pytest.approx(16.430564864)

    int2_est = budget.estimates[-1]
    assert int2_est.weight_memory_gb == pytest.approx(1.995)
    assert int2_est.quality_pct == pytest.approx(80.0)
    assert int2_est.total_memory_gb == pytest.approx(2.465564864)


def test_calculate_dense_72b_does_not_fit_on_small_hw(dense_72b, small_hw):
    budget = calculate(dense_72b, small_hw)

    assert budget.usable_memory_gb == pytest.approx(6.0)
    assert budget.can_run is False
    assert budget.recommended_quant == ""
    assert budget.recommended_context == 0
    assert budget.recommended_memory_gb == 0.0

    assert [e.quant for e in budget.estimates] == ["fp16", "int8", "int4", "int3", "int2"]
    # Weight memory alone exceeds the safe limit (6.0 * 0.85 = 5.1 GB) at every
    # quant level for a 72B model, so `remaining <= 0` for all of them.
    for est in budget.estimates:
        assert est.fits_with_kv is False
        assert est.max_context == 0

    int2_est = budget.estimates[-1]
    assert int2_est.weight_memory_gb == pytest.approx(19.08375)
    assert int2_est.total_memory_gb == pytest.approx(19.08899288)


def test_calculate_custom_default_context_and_safety_margin(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro, default_context=1024, safety_margin=0.5)

    assert budget.can_run is True
    assert budget.recommended_quant == "fp16"
    assert budget.recommended_context == 1024
    assert budget.recommended_memory_gb == pytest.approx(16.019523072)
    assert budget.estimates[0].total_memory_gb == pytest.approx(16.019523072)


def test_calculate_max_context_below_2048_does_not_fit_even_with_headroom():
    """A model whose KV cache is huge per-token (relative to weight size) can have
    remaining > 0 (so it isn't rejected outright) yet max_ctx < 2048, which the
    `fits` check treats as not fitting regardless of total_memory_gb <= safe_limit."""
    hw = HardwareProfile(chip="Apple M4 Pro", total_memory_gb=48.0, memory_bandwidth_gbs=273.0)
    tiny_weight_big_kv = ModelProfile(
        model_id="synthetic/tiny-weight-big-kv",
        total_params_b=0.001,
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=4,
        head_dim=1_000_000,
        max_context=0,  # 0 disables the max_context cap (`if model.max_context > 0`)
    )

    budget = calculate(tiny_weight_big_kv, hw)

    assert budget.can_run is False
    for est in budget.estimates:
        assert est.max_context == 504
        assert est.fits_with_kv is False


# --------------------------------------------------------------------------- #
# calc_theoretical_tps
# --------------------------------------------------------------------------- #


def test_calc_theoretical_tps_happy_path(dense_7b, m4_pro):
    # (273.0 / 15.96) * 0.37
    assert calc_theoretical_tps(dense_7b, m4_pro, "fp16") == pytest.approx(6.328947368421052)


def test_calc_theoretical_tps_custom_efficiency(dense_7b, m4_pro):
    default = calc_theoretical_tps(dense_7b, m4_pro, "fp16")
    doubled = calc_theoretical_tps(dense_7b, m4_pro, "fp16", efficiency=0.74)
    assert doubled == pytest.approx(default * 2)


def test_calc_theoretical_tps_zero_bandwidth_returns_zero_guard(dense_7b):
    hw = HardwareProfile(chip="Unknown", total_memory_gb=48.0, memory_bandwidth_gbs=0.0)
    assert calc_theoretical_tps(dense_7b, hw, "fp16") == 0.0


def test_calc_theoretical_tps_negative_bandwidth_returns_zero_guard(dense_7b, m4_pro):
    hw = HardwareProfile(chip="Unknown", total_memory_gb=48.0, memory_bandwidth_gbs=-10.0)
    assert calc_theoretical_tps(dense_7b, hw, "fp16") == 0.0


def test_calc_theoretical_tps_zero_weight_memory_returns_zero_guard(tiny_model, m4_pro):
    # tiny_model.total_params_b == 0.0 -> calc_weight_memory == 0.0 -> guarded to 0.0
    assert calc_theoretical_tps(tiny_model, m4_pro, "fp16") == 0.0


# --------------------------------------------------------------------------- #
# format_report
# --------------------------------------------------------------------------- #


def test_format_report_contains_expected_sections_when_can_run(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro)
    report = format_report(budget)

    assert isinstance(report, str)
    assert "Memory Budget" in report
    assert "=" * 50 in report
    assert f"Model:            {dense_7b.model_id}" in report
    assert "Usable Memory:    38.0 GB" in report
    assert "Quantization Options:" in report
    assert "Quant" in report and "Weights" in report and "Max Ctx" in report
    assert "Recommended: fp16 @ 8,192 context" in report
    assert "Est. Memory: 16.4 GB" in report
    # one report line per quant estimate
    for quant in ("fp16", "int8", "int4", "int3", "int2"):
        assert quant in report


def test_format_report_contains_cannot_fit_message_when_not_can_run(dense_72b, small_hw):
    budget = calculate(dense_72b, small_hw)
    report = format_report(budget)

    assert "Model cannot fit in available memory at any quantization level." in report
    assert "Recommended:" not in report
