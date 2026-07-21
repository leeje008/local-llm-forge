"""Characterization tests for forge.router (feasibility, alternatives, offload).

These tests pin what the source ACTUALLY does today, not what "should" happen.
Where the behavior looks surprising it is pinned as-is with an explanatory
comment rather than "fixed" — see the accompanying bug report for a summary
of anything suspicious.
"""

from __future__ import annotations

import pytest

from forge.analyzer import memory_calculator as mc
from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.memory_calculator import MemoryBudget, QuantEstimate
from forge.router import alternatives, feasibility, offload
from forge.router.feasibility import Route

# --------------------------------------------------------------------------- #
# Local fixtures (hardware states not covered by conftest.py)
# --------------------------------------------------------------------------- #


@pytest.fixture
def zero_hw() -> HardwareProfile:
    """10GB machine -> usable_memory_gb == 0.0 (8GB OS + 2GB framework buffer)."""
    return HardwareProfile(
        chip="Apple M1",
        cpu_cores_physical=4,
        cpu_cores_logical=4,
        gpu_cores=4,
        ane_tops=5.0,
        total_memory_gb=10.0,
        memory_bandwidth_gbs=60.0,
        disk_available_gb=50.0,
        metal_version=3,
        os_version="26.0.0",
        has_mlx=True,
        mlx_version="0.31.0",
        has_ollama=False,
        python_version="3.14.0",
    )


# --------------------------------------------------------------------------- #
# feasibility.route — 5 stages
# --------------------------------------------------------------------------- #


def test_route_stage1_full_precision(dense_7b, m4_pro):
    """dense_7b fits fp16 easily on a 38GB-usable M4 Pro."""
    budget = mc.calculate(dense_7b, m4_pro)
    decision = feasibility.route(dense_7b, m4_pro, budget)

    assert decision.recommended.route == Route.FULL_PRECISION
    assert decision.recommended.quant == "fp16"
    assert decision.recommended.feasible is True
    assert decision.reasoning == ["Model fits in FP16 — best quality"]


def test_route_stage2_quantized_happy_path(dense_7b, small_hw):
    """16GB hw: fp16/int8 don't fit, int4 does -> QUANTIZED int4, and the
    loop breaks immediately so int3/int2 are never even evaluated."""
    budget = mc.calculate(dense_7b, small_hw)
    decision = feasibility.route(dense_7b, small_hw, budget)

    assert decision.recommended.route == Route.QUANTIZED
    assert decision.recommended.quant == "int4"
    assert decision.reasoning == ["Model fits with int4 quantization (97% quality)"]

    quant_routes = [r for r in decision.all_routes if r.route == Route.QUANTIZED]
    assert [(r.quant, r.feasible) for r in quant_routes] == [
        ("int8", False),
        ("int4", True),
    ]


def test_route_stage2_quantized_skips_higher_bits_when_infeasible(moe_8x7b, m4_pro):
    """Edge case: fp16 AND int8 are both too large for the MoE model even on
    the M4 Pro; the loop must fall through to int4 and mark fp16/int8
    infeasible in all_routes."""
    budget = mc.calculate(moe_8x7b, m4_pro)
    decision = feasibility.route(moe_8x7b, m4_pro, budget)

    assert decision.recommended.route == Route.QUANTIZED
    assert decision.recommended.quant == "int4"

    fp16_route = next(r for r in decision.all_routes if r.route == Route.FULL_PRECISION)
    assert fp16_route.feasible is False

    quant_routes = {r.quant: r.feasible for r in decision.all_routes if r.route == Route.QUANTIZED}
    assert quant_routes == {"int8": False, "int4": True}


def test_route_stage3_offloaded_forced_via_synthetic_budget(dense_72b, m4_pro):
    """No fixture combo naturally reaches OFFLOADED: dense_72b + m4_pro lands
    on QUANTIZED int3 through the real memory_calculator budget. estimate_offload()
    only depends on (model, hardware) — not the budget — so a synthetic budget
    where every quant estimate reports fits_with_kv=False isolates stage 3
    without touching forge/ source."""
    no_fit_budget = MemoryBudget(
        model_id=dense_72b.model_id,
        usable_memory_gb=m4_pro.usable_memory_gb,
        kv_per_token_gb=0.001,
        activation_base_gb=0.1,
        estimates=[
            QuantEstimate(
                quant=q,
                weight_memory_gb=999.0,
                fits_with_kv=False,
                max_context=0,
                quality_pct=0.0,
                total_memory_gb=999.0,
            )
            for q in ["fp16", "int8", "int4", "int3", "int2"]
        ],
        can_run=False,
    )
    decision = feasibility.route(dense_72b, m4_pro, no_fit_budget)

    assert decision.recommended.route == Route.OFFLOADED
    assert decision.recommended.quant == "int4"
    assert decision.recommended.label == "Disk Offload (59/80 GPU layers)"
    assert decision.reasoning == ["Model fits with partial offload (74% GPU)"]

    full_precision = next(
        r for r in decision.all_routes if r.route == Route.FULL_PRECISION
    )
    assert full_precision.feasible is False

    # Stage 2 only iterates the four quantized levels; fp16 is stage 1's job.
    quantized = [r for r in decision.all_routes if r.route == Route.QUANTIZED]
    assert [r.quant for r in quantized] == ["int8", "int4", "int3", "int2"]
    assert all(r.feasible is False for r in quantized)


def test_route_stage4_downsize(dense_72b, small_hw):
    """72.7B model on a 6GB-usable machine: nothing fits, offload doesn't
    clear the 30% GPU-layer minimum, but smaller-model alternatives do."""
    budget = mc.calculate(dense_72b, small_hw)
    decision = feasibility.route(dense_72b, small_hw, budget)

    assert decision.recommended.route == Route.DOWNSIZE
    assert decision.recommended.label == "Smaller Alternative: Llama-3.1-8B-Instruct"
    assert decision.reasoning == [
        "Original model too large — recommending Llama-3.1-8B-Instruct"
    ]
    assert len(decision.alternatives) == 2

    infeasible = [r for r in decision.all_routes if not r.feasible]
    assert len(infeasible) == 6  # fp16, int8, int4, int3, int2, offload
    assert {r.route for r in infeasible} == {Route.FULL_PRECISION, Route.QUANTIZED, Route.OFFLOADED}


def test_route_stage5_cloud_fallback(dense_7b, zero_hw):
    """zero_hw has usable_memory_gb == 0.0 -> nothing fits and no alternative
    clears the (usable * 0.85) threshold -> CLOUD_FALLBACK.

    Quirk pinned here: unlike every other stage, the DOWNSIZE stage only
    appends a RoutePlan when find_alternatives() returns a non-empty list —
    when there are zero alternatives, no DOWNSIZE entry (feasible or not)
    is added to all_routes at all.
    """
    assert zero_hw.usable_memory_gb == 0.0
    budget = mc.calculate(dense_7b, zero_hw)
    decision = feasibility.route(dense_7b, zero_hw, budget)

    assert decision.recommended.route == Route.CLOUD_FALLBACK
    assert decision.reasoning == ["Model cannot run locally — cloud API recommended"]
    assert decision.alternatives == []
    assert not any(r.route == Route.DOWNSIZE for r in decision.all_routes)
    assert all(
        r.feasible is False for r in decision.all_routes if r.route != Route.CLOUD_FALLBACK
    )


# --------------------------------------------------------------------------- #
# feasibility._find_estimate
# --------------------------------------------------------------------------- #


def test_find_estimate_found(dense_7b, m4_pro):
    budget = mc.calculate(dense_7b, m4_pro)
    est = feasibility._find_estimate(budget, "fp16")
    assert est is not None
    assert est.quant == "fp16"


def test_find_estimate_not_found_returns_none(dense_7b, m4_pro):
    budget = mc.calculate(dense_7b, m4_pro)
    assert feasibility._find_estimate(budget, "int9") is None


# --------------------------------------------------------------------------- #
# feasibility.format_route_report
# --------------------------------------------------------------------------- #


def test_format_route_report_full_precision(dense_7b, m4_pro):
    budget = mc.calculate(dense_7b, m4_pro)
    decision = feasibility.route(dense_7b, m4_pro, budget)
    report = feasibility.format_route_report(decision)

    assert isinstance(report, str)
    assert "Model Routing Analysis" in report
    assert "Recommended: Full Precision (FP16)" in report
    assert "Command:     forge optimize Qwen/Qwen2.5-7B-Instruct" in report
    assert "Reasoning:" in report
    assert "✓" in report
    assert "[1]" in report


def test_format_route_report_includes_alternatives_section(dense_72b, small_hw):
    budget = mc.calculate(dense_72b, small_hw)
    decision = feasibility.route(dense_72b, small_hw, budget)
    report = feasibility.format_route_report(decision)

    assert "Model Alternatives:" in report
    assert "Llama-3.1-8B-Instruct: 범용 8B, 4-bit로 ~4GB" in report
    assert "✗" in report  # infeasible routes are present in this scenario


# --------------------------------------------------------------------------- #
# alternatives.find_alternatives
# --------------------------------------------------------------------------- #


def test_find_alternatives_too_large_model_returns_candidates(dense_72b, m4_pro):
    results = alternatives.find_alternatives(dense_72b, m4_pro)

    assert [r["model_id"] for r in results] == [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen1.5-MoE-A2.7B",
    ]
    # results are sorted by params_b, largest (closest to original quality) first
    assert [r["params_b"] for r in results] == sorted(
        (r["params_b"] for r in results), reverse=True
    )


def test_find_alternatives_respects_max_results(dense_72b, m4_pro):
    results = alternatives.find_alternatives(dense_72b, m4_pro, max_results=1)
    assert len(results) == 1
    assert results[0]["model_id"] == "mistralai/Mixtral-8x7B-Instruct-v0.1"


@pytest.mark.parametrize("model_fixture", ["dense_7b", "tiny_model"])
def test_find_alternatives_no_alternatives_returns_empty(model_fixture, m4_pro, request):
    """dense_7b (7.6B, arch 'qwen2'): its own size tier is "" (below the 10B
    floor for tier lookup), so the code falls back to scanning ALL qwen2
    tiers — but every candidate in the DB is excluded by the `params_b >=
    model.total_params_b` filter since none is strictly smaller than 7.6B.
    tiny_model has no architecture match at all. Both land on an empty list."""
    model = request.getfixturevalue(model_fixture)
    assert alternatives.find_alternatives(model, m4_pro) == []


def test_find_alternatives_unknown_architecture_falls_back_to_universal(moe_8x7b, m4_pro):
    """'mixtral' has no entry in _ALTERNATIVES_DB, so family_alts stays empty
    and the universal fallback list is used for a moe_8x7b input model."""
    results = alternatives.find_alternatives(moe_8x7b, m4_pro)
    assert [r["model_id"] for r in results] == [
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]


# --------------------------------------------------------------------------- #
# alternatives._size_tier
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "params_b,expected_tier",
    [
        (500, "405b"),
        (350, "405b"),  # exact lower boundary of the 405b tier
        (349.99, "123b"),  # just under -> falls through to 123b
        (100, "123b"),  # exact lower boundary of the 123b tier
        (99.99, "72b"),
        (60, "72b"),  # exact lower boundary of the 72b tier
        (59.99, "32b"),
        (25, "32b"),  # exact lower boundary of the 32b tier
        (24.99, "14b"),
        (10, "14b"),  # exact lower boundary of the 14b tier
        (9.99, ""),
        (0, ""),
    ],
)
def test_size_tier_boundaries(params_b, expected_tier):
    assert alternatives._size_tier(params_b) == expected_tier


# --------------------------------------------------------------------------- #
# offload.estimate_offload
# --------------------------------------------------------------------------- #


def test_estimate_offload_feasible_ollama_runtime(dense_7b, m4_pro, monkeypatch):
    """dense_7b fits entirely on GPU (28/28 layers). Runtime detection is
    pinned deterministically via monkeypatch instead of depending on whether
    llama-cli happens to be on the host's PATH."""
    monkeypatch.setattr(offload.shutil, "which", lambda name: None)
    result = offload.estimate_offload(dense_7b, m4_pro)

    assert result["feasible"] is True
    assert result["gpu_layers"] == 28
    assert result["cpu_layers"] == 0
    assert result["total_layers"] == 28
    assert result["gpu_pct"] == pytest.approx(100.0)
    assert result["estimated_tps"] == pytest.approx(21.5, rel=1e-2)
    assert result["runtime"] == "ollama"
    assert result["command"] == (
        "ollama run Qwen/Qwen2.5-7B-Instruct  # Ollama auto-manages offloading"
    )


def test_estimate_offload_feasible_llamacpp_runtime(dense_7b, m4_pro, monkeypatch):
    """When llama-cli is on PATH, the command/runtime branch flips."""
    monkeypatch.setattr(offload.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    result = offload.estimate_offload(dense_7b, m4_pro)

    assert result["runtime"] == "llama.cpp"
    assert result["command"] == "llama-cli -m Qwen--Qwen2.5-7B-Instruct.gguf -ngl 28 -c 8192"


def test_estimate_offload_infeasible_reason(dense_72b, small_hw):
    """dense_72b on a 6GB-usable machine: only 11% of layers fit on GPU,
    below the default 30% minimum -> infeasible with a reason string."""
    result = offload.estimate_offload(dense_72b, small_hw)

    assert result["feasible"] is False
    assert result["reason"] == "Only 11% of layers fit on GPU (minimum 30%)"
    assert result["gpu_layers"] == 9
    assert result["gpu_pct"] == pytest.approx(11.25)


def test_estimate_offload_zero_layers_returns_infeasible_no_exception(tiny_model, m4_pro):
    """tiny_model has num_layers == 0. The source has an explicit early-return
    guard for this (`if num_layers == 0`), so it returns a clean infeasible
    dict rather than raising ZeroDivisionError — pinned as-is."""
    result = offload.estimate_offload(tiny_model, m4_pro)
    assert result == {"feasible": False, "reason": "Unknown number of layers"}
