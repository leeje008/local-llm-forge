"""Unit tests for the N-gram self-speculative decode loop (Phase 7 integration).

These tests use a deterministic mock ``forward_fn`` (no model download) so the
speculative accept/reject logic, Adaptive-K adjustment, and MLXEngine opt-in
routing are all verified without loading a real model.
"""

from __future__ import annotations

import pytest

from forge.engine.speculative import (
    AdaptiveKConfig,
    AdaptiveKController,
    NGramDrafter,
    NGramSpeculativeDecoder,
    _argmax_row,
)

VOCAB = 256
EOS = 255


def _one_hot(token_id: int) -> list[float]:
    row = [0.0] * VOCAB
    row[token_id] = 10.0
    return row


def counter_forward(step: int = 1):
    """Target where next(token) = token + step (capped at EOS). No repetition,
    so the n-gram drafter never matches — exercises the progress guarantee."""

    def forward_fn(seq: list[int]):
        return [_one_hot(min(tok + step, EOS)) for tok in seq]

    return forward_fn


def cyclic_forward(cycle: list[int]):
    """Target that walks a fixed cycle: next(cycle[i]) = cycle[i+1 mod P].
    Repetition lets the n-gram drafter learn and get drafts accepted."""
    nxt = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}

    def forward_fn(seq: list[int]):
        return [_one_hot(nxt.get(tok, cycle[0])) for tok in seq]

    return forward_fn


def _make_decoder(initial_k: int = 3, use_adaptive_k: bool = True):
    return NGramSpeculativeDecoder(
        ngram=NGramDrafter(n=3, max_draft=5),
        adaptive_k=AdaptiveKController(AdaptiveKConfig(initial_k=initial_k)),
        use_adaptive_k=use_adaptive_k,
    )


# --------------------------------------------------------------------------- #
# Core loop correctness
# --------------------------------------------------------------------------- #


def test_argmax_row_plain_sequence():
    assert _argmax_row([0.1, 0.9, 0.3]) == 1
    assert _argmax_row(_one_hot(42)) == 42


def test_output_matches_greedy_on_counter():
    """Speculation must be output-identical to plain greedy decoding."""
    decoder = _make_decoder()
    res = decoder.generate([5], counter_forward(), max_tokens=10)
    assert res.tokens == list(range(6, 16))


def test_progress_guaranteed_without_ngram_matches():
    """Even when no draft ever matches, exactly max_tokens are produced."""
    decoder = _make_decoder()
    res = decoder.generate([5], counter_forward(), max_tokens=8)
    assert len(res.tokens) == 8
    assert res.total_accepted == 0  # counter never repeats an n-gram key


def test_empty_prompt_produces_nothing():
    decoder = _make_decoder()
    res = decoder.generate([], counter_forward(), max_tokens=8)
    assert res.tokens == []
    assert res.steps == 0


def test_eos_stops_generation():
    # Counter starting near EOS: 253 -> 254 -> 255(EOS) then stop.
    decoder = _make_decoder()
    res = decoder.generate([253], counter_forward(), max_tokens=50, eos_token_id=EOS)
    assert res.stopped_on_eos is True
    assert res.tokens[-1] == EOS
    assert EOS not in res.tokens[:-1]


# --------------------------------------------------------------------------- #
# Accept / reject + Adaptive-K behavior
# --------------------------------------------------------------------------- #


def test_drafts_accepted_on_repeating_pattern():
    cycle = [10, 11, 12, 13]
    decoder = _make_decoder()
    res = decoder.generate([13], cyclic_forward(cycle), max_tokens=40)
    # Output still exactly the greedy cycle walk.
    assert res.tokens == (cycle * 10)
    # The drafter learned the pattern and got tokens accepted.
    assert res.total_accepted > 0
    assert res.acceptance_rate > 0.5


def test_adaptive_k_increases_under_high_acceptance():
    cycle = [20, 21, 22, 23]
    decoder = _make_decoder(initial_k=3)
    decoder.generate([23], cyclic_forward(cycle), max_tokens=120)
    # Sustained high acceptance should push K above its initial value.
    assert decoder.adaptive_k.current_k > 3


def test_adaptive_k_disabled_keeps_fixed_length():
    decoder = _make_decoder(initial_k=3, use_adaptive_k=False)
    decoder.fixed_k = 2
    cycle = [30, 31, 32, 33]
    decoder.generate([33], cyclic_forward(cycle), max_tokens=60)
    # Controller must not have been driven when adaptive K is off.
    assert decoder.adaptive_k.current_k == 3


def test_adaptive_k_controller_direct():
    ctrl = AdaptiveKController(AdaptiveKConfig(initial_k=5))
    for _ in range(15):  # low acceptance -> K decreases
        ctrl.record_round(drafted=5, accepted=0)
    assert ctrl.current_k < 5

    ctrl2 = AdaptiveKController(AdaptiveKConfig(initial_k=3))
    for _ in range(15):  # high acceptance -> K increases
        ctrl2.record_round(drafted=5, accepted=5)
    assert ctrl2.current_k > 3


# --------------------------------------------------------------------------- #
# MLXEngine opt-in routing (no real model)
# --------------------------------------------------------------------------- #


def test_engine_config_has_ngram_fields():
    from forge.engine.mlx_engine import EngineConfig

    cfg = EngineConfig(model_path="x", ngram_spec=True)
    assert cfg.ngram_spec is True
    assert cfg.use_adaptive_k is True
    assert cfg.ngram_order == 3


def test_generate_routes_to_ngram_path(monkeypatch):
    from forge.engine.mlx_engine import EngineConfig, GenerationResult, MLXEngine

    eng = MLXEngine(EngineConfig(model_path="x", ngram_spec=True))
    eng._loaded = True  # skip real model load

    captured = {}

    def fake_ngram(prompt, max_tokens, temperature):
        captured["args"] = (prompt, max_tokens, temperature)
        return GenerationResult(text="ok", tokens_generated=3, speculative_used=True)

    monkeypatch.setattr(eng, "_generate_ngram_spec", fake_ngram)
    res = eng.generate("hello", max_tokens=7)

    assert captured["args"][0] == "hello"
    assert captured["args"][1] == 7
    assert res.text == "ok"
    assert res.speculative_used is True


def test_default_path_does_not_route_to_ngram(monkeypatch):
    from forge.engine.mlx_engine import EngineConfig, MLXEngine

    eng = MLXEngine(EngineConfig(model_path="x", ngram_spec=False))
    eng._loaded = True

    def boom(*args, **kwargs):
        raise AssertionError("ngram path must not run when ngram_spec is False")

    monkeypatch.setattr(eng, "_generate_ngram_spec", boom)
    # Default path tries to `import mlx_lm` and call stream_generate; we only
    # assert it did NOT take the ngram branch, so a failure there is fine.
    with pytest.raises(Exception) as exc:
        eng.generate("hello", max_tokens=3)
    assert "ngram path must not run" not in str(exc.value)


# --------------------------------------------------------------------------- #
# CLI integrity
# --------------------------------------------------------------------------- #


def test_cli_help_intact():
    from click.testing import CliRunner

    from forge.cli import main

    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0


def test_run_help_exposes_ngram_flag():
    from click.testing import CliRunner

    from forge.cli import main

    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--ngram-spec" in result.output
