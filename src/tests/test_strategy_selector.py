"""Characterization tests for forge.optimizer.strategy_selector.

These tests pin the CURRENT observable behavior of the pure selection logic
(`select`, `_select_quant_method`, `_select_runtime`, `_find_draft_model`,
`format_report`, `build_compound_pipeline`). They do not judge correctness —
where behavior looks surprising it is noted in a comment and pinned as-is.

Explicitly out of scope: `run_asvd_rank_reduction` / `run_layer_prune` (stubs
that "execute" a pipeline stage) are never invoked here.
"""

from __future__ import annotations

import pytest

from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.memory_calculator import MemoryBudget, calculate
from forge.analyzer.model_inspector import ModelProfile
from forge.optimizer.strategy_selector import (
    _find_draft_model,
    _select_quant_method,
    _select_runtime,
    build_compound_pipeline,
    format_report,
    select,
)

# --------------------------------------------------------------------------- #
# select()
# --------------------------------------------------------------------------- #


def test_select_auto_picks_highest_quality_that_fits(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro)
    strategy = select(dense_7b, m4_pro, budget)

    assert strategy.quantization == "fp16"
    assert strategy.quant_method == "mlx_native"
    assert strategy.format == "mlx"
    assert strategy.runtime == "mlx-lm"
    assert strategy.context_length == 8192
    assert strategy.batch_size == 1
    assert strategy.use_speculative is False
    assert strategy.draft_model is None
    assert strategy.use_prompt_cache is True
    assert strategy.use_paged_attention is True  # context >= 8192
    assert strategy.expert_cache_size is None  # dense model
    assert strategy.mixed_quant_recipe is None
    assert strategy.compound_pipeline == []
    assert strategy.estimated_tps == pytest.approx(6.328947368421052)
    assert strategy.estimated_memory_gb == pytest.approx(16.430564864)
    assert strategy.estimated_quality_pct == pytest.approx(100.0)
    assert strategy.warnings == []
    assert strategy.reasoning[0] == (
        "Selected fp16 quantization (highest quality that fits in 38GB)"
    )


def test_select_force_quant_overrides_budget_recommendation(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro)
    strategy = select(dense_7b, m4_pro, budget, force_quant="int8")

    assert strategy.quantization == "int8"
    # Context is recalculated from scratch for forced quant (not budget.recommended_context).
    assert strategy.context_length == 32768
    assert strategy.estimated_quality_pct == pytest.approx(99.5)
    assert "Quantization forced to int8" in strategy.reasoning


def test_select_enable_compound_builds_pipeline_with_chosen_quant_method(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro)
    strategy = select(dense_7b, m4_pro, budget, enable_compound=True)

    # dense_7b is < 13B and has 28 layers (< 32), so neither asvd nor
    # layer_prune stages are added — only rotation + final quantize.
    assert strategy.compound_pipeline == ["gsr_rotation", "quantize:mlx_native"]
    assert any("Compound pipeline enabled" in r for r in strategy.reasoning)


def test_select_enable_speculative_attaches_draft_model(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro)
    strategy = select(dense_7b, m4_pro, budget, enable_speculative=True)

    assert strategy.use_speculative is True
    assert strategy.draft_model == "Qwen/Qwen2.5-0.5B"
    assert any("Speculative decoding enabled" in r for r in strategy.reasoning)


def test_select_quant_method_override_replaces_selected_method(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro)
    strategy = select(dense_7b, m4_pro, budget, quant_method_override="aqlm")

    assert strategy.quant_method == "aqlm"
    assert "Quant method overridden: aqlm" in strategy.reasoning


def test_select_too_large_model_returns_default_strategy_with_warning(dense_72b, small_hw):
    budget = calculate(dense_72b, small_hw)
    assert budget.can_run is False  # sanity check on the fixture combination

    strategy = select(dense_72b, small_hw, budget)

    assert strategy.warnings == ["Model cannot fit in available memory at any quantization level"]
    assert strategy.reasoning == ["Model too large for available hardware"]
    # Early-return path: strategy is the untouched dataclass default.
    assert strategy.quantization == "int4"
    assert strategy.context_length == 4096
    assert strategy.estimated_tps == 0.0
    assert strategy.estimated_memory_gb == 0.0


def test_select_moe_model_sets_expert_cache(moe_8x7b, m4_pro):
    budget = calculate(moe_8x7b, m4_pro)
    strategy = select(moe_8x7b, m4_pro, budget)

    # num_active_experts=2 -> expert_cache_size = 2 * 2
    assert strategy.expert_cache_size == 4
    assert any("Expert cache: 4 entries" in r for r in strategy.reasoning)


# --------------------------------------------------------------------------- #
# _select_quant_method()
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("quant", "has_mlx", "expected"),
    [
        ("fp16", True, ("mlx_native", None)),
        ("fp16", False, ("mlx_native", None)),
        ("int8", True, ("mlx_native", None)),
        ("int8", False, ("mlx_native", None)),
        ("int4", True, ("mlx_native", None)),
        ("int4", False, ("mlx_native", None)),
        ("int3", True, ("mlx_native", "mixed_3_6")),
        ("int3", False, ("hqq", None)),
        ("int2", True, ("hqq", None)),
        ("int2", False, ("hqq", None)),
        # Unknown values fall through to the default branch.
        ("unknown", True, ("mlx_native", None)),
        ("unknown", False, ("mlx_native", None)),
    ],
)
def test_select_quant_method_matrix(quant, has_mlx, expected, m4_pro):
    hardware = m4_pro if has_mlx else HardwareProfile(chip="x", total_memory_gb=16.0, has_mlx=False)
    assert _select_quant_method(quant, hardware) == expected


# --------------------------------------------------------------------------- #
# _select_runtime()
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("format_", "has_mlx", "has_ollama", "expected"),
    [
        ("mlx", True, True, "mlx-lm"),
        ("mlx", False, True, "ollama"),
        ("mlx", False, False, "llama.cpp"),
        ("gguf", True, True, "ollama"),  # format != mlx, so has_mlx is irrelevant
        ("gguf", False, True, "ollama"),
        ("gguf", False, False, "llama.cpp"),
    ],
)
def test_select_runtime_matrix(format_, has_mlx, has_ollama, expected):
    hardware = HardwareProfile(
        chip="x", total_memory_gb=16.0, has_mlx=has_mlx, has_ollama=has_ollama
    )
    assert _select_runtime(hardware, format_) == expected


# --------------------------------------------------------------------------- #
# _find_draft_model()
# --------------------------------------------------------------------------- #


def test_find_draft_model_known_architecture(dense_7b):
    assert _find_draft_model(dense_7b) == "Qwen/Qwen2.5-0.5B"


def test_find_draft_model_matches_substring_in_hf_class_name():
    # Architecture matching is substring-based, not exact — a raw HF class
    # name like "Qwen2ForCausalLM" still matches the "qwen2" key.
    model = ModelProfile(architecture="Qwen2ForCausalLM")
    assert _find_draft_model(model) == "Qwen/Qwen2.5-0.5B"


def test_find_draft_model_unknown_architecture_falls_back():
    model = ModelProfile(architecture="totally-unknown-arch")
    assert _find_draft_model(model) == "HuggingFaceTB/SmolLM2-360M"


# --------------------------------------------------------------------------- #
# format_report()
# --------------------------------------------------------------------------- #


def test_format_report_contains_expected_markers(dense_7b, m4_pro):
    budget = calculate(dense_7b, m4_pro)
    strategy = select(dense_7b, m4_pro, budget)
    report = format_report(strategy)

    assert isinstance(report, str)
    assert "Optimization Strategy" in report
    assert "Quantization:     fp16 (mlx_native)" in report
    assert "Runtime:          mlx-lm" in report
    assert "Context Length:   8,192" in report
    assert "Speed:   ~6.3 tok/s" in report
    assert "Reasoning:" in report
    # No warnings on the happy path.
    assert "Warnings:" not in report


def test_format_report_includes_warnings_section_when_present():
    model = ModelProfile(
        architecture="qwen2", total_params_b=72.7, num_layers=80, hidden_size=8192,
        num_attention_heads=64, num_kv_heads=8, head_dim=128,
    )
    hardware = HardwareProfile(chip="x", total_memory_gb=16.0, has_mlx=True)
    budget = MemoryBudget(
        model_id="x", usable_memory_gb=6.0, kv_per_token_gb=0.0, activation_base_gb=0.0,
        can_run=False,
    )
    strategy = select(model, hardware, budget)
    report = format_report(strategy)
    assert "Warnings:" in report
    assert "! Model cannot fit in available memory at any quantization level" in report


# --------------------------------------------------------------------------- #
# build_compound_pipeline() — pure config assembly, never executed
# --------------------------------------------------------------------------- #


def test_build_compound_pipeline_small_dense_model_skips_asvd_and_prune(dense_7b, m4_pro):
    # total_params_b=7.6 (< 13.0) and num_layers=28 (< 32).
    assert build_compound_pipeline(dense_7b, m4_pro) == ["gsr_rotation", "quantize:mlx_native"]


def test_build_compound_pipeline_large_dense_model_includes_all_stages(dense_72b, m4_pro):
    # total_params_b=72.7 (>= 13.0) and num_layers=80 (>= 32).
    assert build_compound_pipeline(dense_72b, m4_pro, quant_method="hqq") == [
        "asvd_rank_reduction",
        "layer_prune",
        "gsr_rotation",
        "quantize:hqq",
    ]


def test_build_compound_pipeline_moe_skips_asvd_but_keeps_layer_prune(moe_8x7b, m4_pro):
    # MoE models never get asvd_rank_reduction regardless of size; num_layers=32 (>= 32).
    assert build_compound_pipeline(moe_8x7b, m4_pro) == [
        "layer_prune",
        "gsr_rotation",
        "quantize:mlx_native",
    ]


def test_build_compound_pipeline_tiny_model_still_gets_rotation_and_quantize(tiny_model, m4_pro):
    # Degenerate all-zero profile: model_type defaults to "dense" and
    # total_params_b/num_layers are both 0, so only the always-on stages run.
    assert build_compound_pipeline(tiny_model, m4_pro) == ["gsr_rotation", "quantize:mlx_native"]
