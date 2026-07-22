"""Characterization tests for forge.engine.speculative.draft_models.

Pins current behavior. Two spots that previously produced suspicious results
were fixed (commit updating these tests): the ``FALLBACK_DRAFTS`` loop now
honors ``available_memory_gb`` (returns ``None`` when even the fallback does
not fit), and the unreachable ``prefer_mlx_community`` non-mlx inner loop was
removed. ``prefer_mlx_community=True`` and ``=False`` still produce identical
output for every known architecture, because every current ``DRAFT_MODELS``
entry that fits is an mlx-community candidate.
"""

from __future__ import annotations

import pytest

from forge.engine.speculative.draft_models import (
    DraftModelInfo,
    estimate_speedup,
    select_draft_model,
)

# --------------------------------------------------------------------------- #
# select_draft_model()
# --------------------------------------------------------------------------- #


def test_select_draft_model_architecture_match_qwen2():
    result = select_draft_model("qwen2")
    assert result == DraftModelInfo(
        model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        estimated_size_gb=0.3,
        architecture_match=True,
        source="architecture_match",
    )


def test_select_draft_model_unknown_architecture_falls_back():
    result = select_draft_model("totally-unknown-arch")
    assert result.source == "fallback"
    assert result.architecture_match is False
    assert result.model_id == "mlx-community/SmolLM2-360M-Instruct-4bit"
    assert result.estimated_size_gb == 0.3


def test_select_draft_model_memory_constrained_returns_none_when_nothing_fits():
    # qwen2's mlx candidate needs 0.3GB and the universal fallback also needs
    # 0.3GB; with only 0.1GB neither fits, so the function returns None.
    result = select_draft_model("qwen2", available_memory_gb=0.1)
    assert result is None


def test_select_draft_model_fallback_respects_available_memory_budget():
    # Fixed: the fallback loop now honors the budget. With 0.01GB available,
    # even the 0.3GB universal fallback does not fit, so the function returns
    # None instead of a model that cannot load.
    result = select_draft_model("qwen2", available_memory_gb=0.01)
    assert result is None


@pytest.mark.parametrize("prefer_mlx_community", [True, False])
def test_select_draft_model_prefer_mlx_community_both_code_paths_qwen2(prefer_mlx_community):
    # Both flag values currently produce the identical result for qwen2 —
    # see module docstring for why the `prefer_mlx_community=True`-only
    # "non-mlx fallback" loop never actually gets reached here.
    result = select_draft_model("qwen2", prefer_mlx_community=prefer_mlx_community)
    assert result.model_id == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert result.source == "architecture_match"


@pytest.mark.parametrize("prefer_mlx_community", [True, False])
def test_select_draft_model_prefer_mlx_community_both_code_paths_mistral(prefer_mlx_community):
    # mistral has only a single (mlx-community) candidate in DRAFT_MODELS,
    # so the flag has no observable effect here either.
    result = select_draft_model(
        "mistral", available_memory_gb=2.0, prefer_mlx_community=prefer_mlx_community
    )
    assert result.model_id == "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    assert result.estimated_size_gb == 0.8
    assert result.source == "architecture_match"


def test_select_draft_model_matches_by_substring_in_full_hf_class_name():
    result = select_draft_model("Qwen2ForCausalLM")
    assert result.model_id == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


# --------------------------------------------------------------------------- #
# estimate_speedup()
# --------------------------------------------------------------------------- #


def test_estimate_speedup_happy_path():
    speedup = estimate_speedup(
        target_params_b=7.6, draft_params_b=0.5, acceptance_rate=0.7, num_draft_tokens=3
    )
    assert speedup == pytest.approx(1.8626644736842106)


def test_estimate_speedup_zero_acceptance_rate_returns_one_without_computing_overhead():
    assert estimate_speedup(7.6, acceptance_rate=0) == 1.0


def test_estimate_speedup_negative_acceptance_rate_also_returns_one():
    # `<= 0` guard covers negative values too, not just exactly zero.
    assert estimate_speedup(7.6, acceptance_rate=-0.5) == 1.0


def test_estimate_speedup_full_acceptance_rate():
    speedup = estimate_speedup(7.6, draft_params_b=0.5, acceptance_rate=1.0, num_draft_tokens=3)
    assert speedup == pytest.approx(2.980263157894737)
