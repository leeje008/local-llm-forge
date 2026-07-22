"""Characterization tests for the request-scheduling control plane.

Pins the CURRENT behavior of ``forge.engine.scheduler``'s pure control-plane
logic: ``ChunkedPrefillScheduler``, ``InterruptibleSession``, ``ModelRegistry``,
``configure_pmpd``, ``detect_vllm_mlx``, and ``build_context``. No mlx,
network, or disk writes:

- ``ModelRegistry`` is tested with an injected ``engine_factory`` returning a
  fake in-memory engine, so ``MLXEngine.load()`` (which does ``import
  mlx_lm``) is never reached.
- ``_estimate_model_memory_gb`` is exercised only against nonexistent paths
  (returns 0.0, a pure read-only ``Path.exists()`` check — no writes) or
  monkeypatched outright for deterministic budget-eviction tests.
- ``PagedAttentionAdapter.load()`` / ``.generate()`` are SKIPPED entirely —
  both paths ultimately construct a real ``MLXEngine`` and attempt to load a
  model. Only the pure constructor / ``backend`` property is covered.
- ``PMPDPlan`` / ``configure_pmpd`` build ``EngineConfig`` dataclasses only,
  no engine is instantiated.

KNOWN BUG pinned here (not fixed): ``ChunkedPrefillScheduler`` computes a
prefill work unit's ``chunk_index`` as ``req.current_chunk_index - 1``
*after* ``prefill_pos`` has already been advanced. When the prompt length
isn't a multiple of ``chunk_size``, the final (partial) chunk gets the same
``chunk_index`` as the second-to-last chunk instead of incrementing, e.g.
for a 7-token prompt with ``chunk_size=3`` the three emitted chunks report
indices ``0, 1, 1`` (not ``0, 1, 2``) even though ``total_chunks == 3``. See
``test_chunk_index_bug_on_non_divisible_prompt_length``.

KNOWN QUIRK pinned here: ``InterruptibleSession.finish()`` transitions to
``DONE`` unconditionally from ANY state (including ``READY``, before
``start()`` was ever called) — there is no state-machine guard on
``finish()`` the way there is on ``resume()``. See
``test_finish_from_ready_state_jumps_straight_to_done``.

KNOWN QUIRK pinned here: ``build_context(interruptible=True)`` sets
``ctx.sessions = {}`` but ``AdvancedServingContext.any_enabled()`` checks
``bool(self.sessions)`` — an empty dict is falsy, so requesting
``interruptible=True`` alone does NOT make ``any_enabled()`` return True
until a session is actually added. See
``test_build_context_interruptible_alone_does_not_flip_any_enabled``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.engine.mlx_engine import EngineConfig
from forge.engine.scheduler import (
    AdvancedServingContext,
    ChunkedPrefillScheduler,
    InterruptibleSession,
    ModelRegistry,
    PagedAttentionAdapter,
    PrefillRequest,
    ResumePlan,
    SessionState,
    WorkKind,
    build_context,
    configure_pmpd,
    detect_vllm_mlx,
)


class FakeEngine:
    """Stand-in for MLXEngine that never touches mlx_lm."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded = False
        self.unloaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.unloaded = True


# --------------------------------------------------------------------------- #
# PrefillRequest — pure dataclass properties
# --------------------------------------------------------------------------- #


def test_prefill_request_empty_prompt_is_immediately_done():
    req = PrefillRequest(request_id="r1", prompt_tokens=[], chunk_size=5)
    assert req.total_chunks == 0
    assert req.prefill_done is True
    assert req.current_chunk_index == 0


@pytest.mark.parametrize(
    "n_tokens, chunk_size, expected_total_chunks",
    [
        (7, 3, 3),   # 7 = 3+3+1 -> ceil(7/3) = 3
        (6, 3, 2),   # exact multiple
        (1, 3, 1),   # smaller than one chunk
    ],
)
def test_prefill_request_total_chunks_boundary(n_tokens, chunk_size, expected_total_chunks):
    req = PrefillRequest(request_id="r", prompt_tokens=list(range(n_tokens)), chunk_size=chunk_size)
    assert req.total_chunks == expected_total_chunks


def test_prefill_request_current_chunk_index_guards_zero_chunk_size():
    req = PrefillRequest(request_id="r", prompt_tokens=[1, 2, 3], chunk_size=0)
    assert req.current_chunk_index == 0


# --------------------------------------------------------------------------- #
# ChunkedPrefillScheduler — construction
# --------------------------------------------------------------------------- #


def test_scheduler_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError):
        ChunkedPrefillScheduler(chunk_size=0)
    with pytest.raises(ValueError):
        ChunkedPrefillScheduler(chunk_size=-1)


def test_schedule_step_on_empty_scheduler_yields_idle():
    sched = ChunkedPrefillScheduler()
    units = list(sched.schedule_step())
    assert len(units) == 1
    assert units[0].kind == WorkKind.IDLE
    assert units[0].request_id == ""


# --------------------------------------------------------------------------- #
# add_request() / pending()
# --------------------------------------------------------------------------- #


def test_add_request_increments_total_and_pending():
    sched = ChunkedPrefillScheduler()
    sched.add_request("r1", [1, 2, 3], max_tokens=10)
    assert sched.stats.total_requests == 1
    assert sched.pending() == 1


def test_max_active_requests_gates_queue_promotion():
    sched = ChunkedPrefillScheduler(chunk_size=10, max_active_requests=1)
    sched.add_request("r1", [1, 2, 3])
    sched.add_request("r2", [4, 5, 6])
    assert sched.pending() == 2

    # Only r1 is promoted to active and emits work this tick.
    units = list(sched.schedule_step())
    assert [u.request_id for u in units] == ["r1"]

    sched.finish("r1")
    units2 = list(sched.schedule_step())
    assert [u.request_id for u in units2] == ["r2"]


# --------------------------------------------------------------------------- #
# schedule_step() — chunk splitting
# --------------------------------------------------------------------------- #


def test_prefill_chunks_split_evenly_and_flag_final_chunk():
    sched = ChunkedPrefillScheduler(chunk_size=3, max_active_requests=8)
    sched.add_request("r1", [1, 2, 3, 4, 5, 6], max_tokens=10)  # divides evenly: 2 chunks

    step1 = list(sched.schedule_step())
    step2 = list(sched.schedule_step())

    assert step1[0].kind == WorkKind.PREFILL_CHUNK
    assert step1[0].tokens == [1, 2, 3]
    assert step1[0].chunk_index == 0
    assert step1[0].total_chunks == 2
    assert step1[0].is_final_chunk is False

    assert step2[0].tokens == [4, 5, 6]
    assert step2[0].chunk_index == 1
    assert step2[0].is_final_chunk is True


def test_chunk_index_bug_on_non_divisible_prompt_length():
    """BUG (pinned, not fixed): 7 tokens / chunk_size=3 -> chunks of
    [3, 3, 1] tokens, total_chunks=3, but chunk_index sequence comes out
    0, 1, 1 instead of 0, 1, 2 because current_chunk_index is computed
    from prefill_pos // chunk_size AFTER prefill_pos was already advanced,
    and 6 // 3 == 7 // 3 == 2 collide once the tail chunk is smaller than
    chunk_size."""
    sched = ChunkedPrefillScheduler(chunk_size=3, max_active_requests=8)
    sched.add_request("r1", list(range(7)), max_tokens=10)

    chunks = []
    for _ in range(3):
        chunks.extend(list(sched.schedule_step()))

    assert [c.tokens for c in chunks] == [[0, 1, 2], [3, 4, 5], [6]]
    assert [c.total_chunks for c in chunks] == [3, 3, 3]
    assert [c.chunk_index for c in chunks] == [0, 1, 1]  # <- bug: should be [0, 1, 2]
    assert [c.is_final_chunk for c in chunks] == [False, False, True]


def test_after_prefill_done_scheduler_emits_decode_steps():
    sched = ChunkedPrefillScheduler(chunk_size=10, max_active_requests=8)
    sched.add_request("r1", [1, 2, 3], max_tokens=10)

    prefill_units = list(sched.schedule_step())
    assert prefill_units[0].kind == WorkKind.PREFILL_CHUNK
    assert prefill_units[0].is_final_chunk is True

    decode_units = list(sched.schedule_step())
    assert decode_units[0].kind == WorkKind.DECODE
    assert decode_units[0].decode_step == 0
    assert sched.stats.decode_steps_emitted == 1


def test_mixed_batch_emits_prefill_and_decode_in_same_tick():
    sched = ChunkedPrefillScheduler(chunk_size=10, max_active_requests=8)
    sched.add_request("short", [1, 2], max_tokens=10)
    list(sched.schedule_step())  # short finishes prefill this tick

    sched.add_request("long", list(range(20)), max_tokens=10)
    units = list(sched.schedule_step())

    kinds = {u.request_id: u.kind for u in units}
    assert kinds["short"] == WorkKind.DECODE
    assert kinds["long"] == WorkKind.PREFILL_CHUNK


def test_zero_length_prompt_goes_straight_to_decode():
    sched = ChunkedPrefillScheduler(chunk_size=10, max_active_requests=8)
    sched.add_request("r1", [], max_tokens=5)

    units = list(sched.schedule_step())

    assert units[0].kind == WorkKind.DECODE


# --------------------------------------------------------------------------- #
# record_token() / finish() / preempt()
# --------------------------------------------------------------------------- #


def test_record_token_auto_finishes_at_max_tokens():
    sched = ChunkedPrefillScheduler(chunk_size=10, max_active_requests=8)
    sched.add_request("r1", [1], max_tokens=2)
    list(sched.schedule_step())  # prefill
    list(sched.schedule_step())  # decode

    sched.record_token("r1", 100)
    assert sched.stats.completed_requests == 0
    assert sched.pending() == 1

    sched.record_token("r1", 101)
    assert sched.stats.completed_requests == 1
    assert sched.pending() == 0


def test_record_token_on_unknown_request_id_is_a_noop():
    sched = ChunkedPrefillScheduler()
    sched.record_token("does-not-exist", 42)  # must not raise
    assert sched.stats.completed_requests == 0


def test_finish_removes_from_active_and_increments_completed():
    sched = ChunkedPrefillScheduler(chunk_size=10)
    sched.add_request("r1", [1, 2, 3])
    list(sched.schedule_step())  # promote to active

    sched.finish("r1")

    assert sched.stats.completed_requests == 1
    assert sched.pending() == 0


def test_preempt_removes_from_active_but_does_not_requeue():
    sched = ChunkedPrefillScheduler(chunk_size=10)
    sched.add_request("r1", [1, 2, 3])
    list(sched.schedule_step())  # promote to active

    req = sched.preempt("r1")

    assert req is not None
    assert req.request_id == "r1"
    assert sched.stats.preemptions == 1
    assert sched.pending() == 0  # neither active nor queued anymore


def test_preempt_unknown_request_returns_none():
    sched = ChunkedPrefillScheduler()
    assert sched.preempt("ghost") is None
    assert sched.stats.preemptions == 0


# --------------------------------------------------------------------------- #
# InterruptibleSession — state machine
# --------------------------------------------------------------------------- #


def test_session_starts_ready_then_running():
    session = InterruptibleSession(session_id="s1", prompt_text="hi", prompt_tokens=[1, 2])
    assert session.state == SessionState.READY
    session.start()
    assert session.state == SessionState.RUNNING


def test_pause_only_takes_effect_from_running():
    session = InterruptibleSession(session_id="s1", prompt_text="hi", prompt_tokens=[1])
    session.pause()  # no-op: state is READY, not RUNNING
    assert session.state == SessionState.READY
    assert session.pause_count == 0

    session.start()
    session.pause()
    assert session.state == SessionState.PAUSED
    assert session.pause_count == 1


def test_resume_produces_plan_and_returns_to_running():
    session = InterruptibleSession(
        session_id="s1",
        prompt_text="hi",
        prompt_tokens=[1, 2, 3],
        generated_tokens=[9, 8],
        max_tokens=5,
    )
    session.start()
    session.pause()

    plan = session.resume()

    assert isinstance(plan, ResumePlan)
    assert session.state == SessionState.RUNNING
    assert plan.prompt_tokens == [1, 2, 3]
    assert plan.generated_tokens == [9, 8]
    assert plan.remaining_tokens == 3  # max_tokens(5) - len(generated_tokens)(2)
    assert plan.reuse_prefix is False  # prefix_cache_ref not set
    assert plan.replay_tokens == [1, 2, 3, 9, 8]


def test_resume_reuse_prefix_reflects_prefix_cache_ref():
    session = InterruptibleSession(
        session_id="s1", prompt_text="hi", prompt_tokens=[1], prefix_cache_ref=object()
    )
    session.start()
    session.pause()
    plan = session.resume()
    assert plan.reuse_prefix is True


def test_resume_raises_when_not_paused():
    session = InterruptibleSession(session_id="s1", prompt_text="hi", prompt_tokens=[1])
    session.start()
    with pytest.raises(RuntimeError):
        session.resume()


def test_record_token_appends_and_marks_done_at_limit():
    session = InterruptibleSession(
        session_id="s1", prompt_text="hi", prompt_tokens=[1], max_tokens=2
    )
    session.start()
    session.record_token(10, "a")
    assert session.state == SessionState.RUNNING
    assert session.generated_text == "a"

    session.record_token(11, "b")
    assert session.state == SessionState.DONE
    assert session.generated_tokens == [10, 11]
    assert session.generated_text == "ab"


def test_finish_from_ready_state_jumps_straight_to_done():
    """QUIRK (pinned): finish() has no state guard, unlike resume()."""
    session = InterruptibleSession(session_id="s1", prompt_text="hi", prompt_tokens=[1])
    assert session.state == SessionState.READY
    session.finish()
    assert session.state == SessionState.DONE


def test_to_dict_from_dict_round_trip():
    session = InterruptibleSession(
        session_id="s1",
        prompt_text="hello",
        prompt_tokens=[1, 2],
        generated_tokens=[3],
        generated_text="x",
        max_tokens=7,
        prompt_cache_path="/some/path.safetensors",
    )
    session.start()
    session.pause()

    data = session.to_dict()
    restored = InterruptibleSession.from_dict(data)

    assert restored.session_id == "s1"
    assert restored.prompt_tokens == [1, 2]
    assert restored.generated_tokens == [3]
    assert restored.max_tokens == 7
    assert restored.state == SessionState.PAUSED
    assert restored.pause_count == 1


def test_from_dict_defaults_missing_fields():
    restored = InterruptibleSession.from_dict({"session_id": "bare"})
    assert restored.session_id == "bare"
    assert restored.prompt_text == ""
    assert restored.prompt_tokens == []
    assert restored.max_tokens == 256
    assert restored.state == SessionState.READY


# --------------------------------------------------------------------------- #
# ModelRegistry — LRU multi-model hot loading (fake engine, no mlx)
# --------------------------------------------------------------------------- #


def make_registry(**kwargs) -> ModelRegistry:
    kwargs.setdefault("engine_factory", lambda cfg: FakeEngine(cfg))
    return ModelRegistry(**kwargs)


def test_get_loads_once_and_reuses_on_second_call():
    reg = make_registry(memory_budget_gb=100.0, max_models=4)
    e1 = reg.get("/nonexistent/a")
    e2 = reg.get("/nonexistent/a")
    assert e1 is e2
    assert isinstance(e1, FakeEngine)
    assert e1.loaded is True
    assert reg.loaded_paths() == ["/nonexistent/a"]


def test_get_second_use_touches_use_count():
    reg = make_registry(memory_budget_gb=100.0, max_models=4)
    reg.get("/nonexistent/a")
    reg.get("/nonexistent/a")
    entry = reg._models[str(Path("/nonexistent/a"))]
    assert entry.use_count == 2


def test_lru_eviction_triggered_by_max_models():
    reg = make_registry(memory_budget_gb=100.0, max_models=2)
    reg.get("/nonexistent/a")
    reg.get("/nonexistent/b")
    reg.get("/nonexistent/c")  # evicts LRU ("a")

    assert reg.loaded_paths() == ["/nonexistent/b", "/nonexistent/c"]


def test_lru_eviction_respects_recent_access_order():
    reg = make_registry(memory_budget_gb=100.0, max_models=2)
    reg.get("/nonexistent/a")
    reg.get("/nonexistent/b")
    reg.get("/nonexistent/a")  # touch a -> b becomes LRU
    reg.get("/nonexistent/c")  # evicts "b", not "a"

    assert reg.loaded_paths() == ["/nonexistent/a", "/nonexistent/c"]


def test_budget_driven_eviction(monkeypatch):
    import forge.engine.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod, "_estimate_model_memory_gb", lambda p: 10.0)

    reg = make_registry(memory_budget_gb=15.0, max_models=10)
    reg.get("/nonexistent/a")
    reg.get("/nonexistent/b")  # resident would be 20 > 15 -> evicts "a" first

    assert reg.loaded_paths() == ["/nonexistent/b"]
    assert reg.resident_gb == 10.0


def test_evict_returns_false_for_unknown_path():
    reg = make_registry()
    assert reg.evict("/never/loaded") is False


def test_evict_calls_unload_on_fake_engine():
    reg = make_registry()
    reg.get("/nonexistent/a")
    assert reg.evict("/nonexistent/a") is True
    assert reg.loaded_paths() == []


def test_clear_evicts_everything():
    reg = make_registry(max_models=10)
    reg.get("/nonexistent/a")
    reg.get("/nonexistent/b")
    reg.clear()
    assert reg.loaded_paths() == []
    assert reg.resident_gb == 0.0


def test_format_report_smoke():
    reg = make_registry(memory_budget_gb=32.0)
    reg.get("/nonexistent/a")
    text = reg.format_report()
    assert "Model Registry" in text
    assert "Budget:   32.0 GB" in text


# --------------------------------------------------------------------------- #
# configure_pmpd() — pure EngineConfig orchestration
# --------------------------------------------------------------------------- #


def test_configure_pmpd_without_decode_path_is_single_precision():
    plan = configure_pmpd("modelA")
    assert plan.hot_swap_supported is False
    assert plan.prefill_config.model_path == "modelA"
    assert plan.decode_config.model_path == "modelA"
    assert "single-precision" in plan.notes


def test_configure_pmpd_with_decode_path_enables_hot_swap():
    plan = configure_pmpd("modelA", "modelB")
    assert plan.hot_swap_supported is True
    assert plan.prefill_config.model_path == "modelA"
    assert plan.decode_config.model_path == "modelB"
    assert "single-precision" not in plan.notes


def test_configure_pmpd_clones_base_config_fields_into_both_phases():
    base = EngineConfig(max_tokens=99, kv_bits=8, temperature=0.3)
    plan = configure_pmpd("modelA", "modelB", base_config=base)
    assert plan.prefill_config.max_tokens == 99
    assert plan.decode_config.kv_bits == 8
    assert plan.prefill_config.temperature == 0.3
    assert plan.prefill_config.pmpd_mode is True
    assert plan.decode_config.pmpd_mode is True


# --------------------------------------------------------------------------- #
# detect_vllm_mlx() / PagedAttentionAdapter (construction only — no load())
# --------------------------------------------------------------------------- #


def test_detect_vllm_mlx_matches_importlib_find_spec():
    import importlib.util

    assert detect_vllm_mlx() == (importlib.util.find_spec("vllm_mlx") is not None)


def test_paged_attention_adapter_backend_reflects_vllm_availability():
    adapter = PagedAttentionAdapter(EngineConfig(model_path="m"), prefer_vllm=True)
    expected = "vllm_mlx" if detect_vllm_mlx() else "mlx_engine"
    assert adapter.backend == expected


def test_paged_attention_adapter_prefer_vllm_false_always_uses_fallback():
    adapter = PagedAttentionAdapter(EngineConfig(model_path="m"), prefer_vllm=False)
    assert adapter.backend == "mlx_engine"


# NOTE: PagedAttentionAdapter.load() / .generate() are SKIPPED — both paths
# ultimately construct a real MLXEngine and call .load(), which does
# `import mlx_lm` and attempts to read model weights. Not pure control-plane
# logic, out of scope for this characterization suite.


# --------------------------------------------------------------------------- #
# AdvancedServingContext / build_context() — factory wiring
# --------------------------------------------------------------------------- #


def test_default_context_has_nothing_enabled():
    ctx = AdvancedServingContext()
    assert ctx.any_enabled() is False
    assert ctx.sessions == {}


def test_build_context_no_flags_enables_nothing():
    ctx = build_context("model_path")
    assert ctx.scheduler is None
    assert ctx.registry is None
    assert ctx.pmpd_plan is None
    assert ctx.paged_adapter is None
    assert ctx.any_enabled() is False


def test_build_context_chunked_prefill_wires_scheduler():
    ctx = build_context("model_path", chunked_prefill=True, chunk_size=128)
    assert isinstance(ctx.scheduler, ChunkedPrefillScheduler)
    assert ctx.scheduler.chunk_size == 128
    assert ctx.any_enabled() is True


def test_build_context_multi_model_wires_registry():
    ctx = build_context("model_path", multi_model=True, memory_budget_gb=8.0)
    assert isinstance(ctx.registry, ModelRegistry)
    assert ctx.registry.memory_budget_gb == 8.0
    assert ctx.any_enabled() is True


def test_build_context_interruptible_alone_does_not_flip_any_enabled():
    """QUIRK (pinned): sessions starts as an empty dict, which is falsy."""
    ctx = build_context("model_path", interruptible=True)
    assert ctx.sessions == {}
    assert ctx.any_enabled() is False

    # Adding an actual session is what flips any_enabled() to True.
    ctx.sessions["s1"] = InterruptibleSession(
        session_id="s1", prompt_text="hi", prompt_tokens=[1]
    )
    assert ctx.any_enabled() is True


def test_build_context_pmpd_wires_plan():
    ctx = build_context("model_path", pmpd=True)
    assert ctx.pmpd_plan is not None
    assert ctx.pmpd_plan.prefill_config.model_path == "model_path"
    assert ctx.any_enabled() is True


def test_build_context_use_vllm_mlx_wires_adapter_without_loading():
    ctx = build_context("model_path", use_vllm_mlx=True)
    assert isinstance(ctx.paged_adapter, PagedAttentionAdapter)
    assert ctx.any_enabled() is True
