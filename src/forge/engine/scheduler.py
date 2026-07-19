"""Phase 10: Request-level scheduling and serving enhancements.

Implements control-plane primitives for advanced serving on top of the
existing :class:`MLXEngine` + ``mlx_lm.server`` stack:

10.1 ChunkedPrefillScheduler — Sarathi (2403.02310) chunked prefill.
     Splits long prompts into fixed-size chunks and interleaves decode
     steps so TTFT of short requests isn't blocked by long prefills.

10.2 InterruptibleSession — FastServe-style preemption. Tracks prompt +
     generated tokens so far and can be paused/resumed. On resume the
     prefix is rebuilt through the radix prefix cache (or re-prefilled
     from prompt-cache files) rather than from a live KV tensor, since
     mlx-lm's internal KVCache doesn't expose a clean save/restore API.

10.3 ModelRegistry — multi-model hot loading with LRU eviction driven
     by a unified-memory budget.

10.4 configure_pmpd() — control plane for PMPD (ICLR 2025 2410.13461):
     FP16 prefill, INT4 decode. Experimental — the actual switch today
     requires pointing at two differently-quantized model copies, so
     this helper just orchestrates ``EngineConfig`` instances and flags
     whether hot-swap is possible on the current runtime.

10.5 PagedAttentionAdapter / detect_vllm_mlx — thin adapter that
     delegates to ``vllm_mlx`` when importable and transparently falls
     back to :class:`MLXEngine` otherwise.

All heavy libs (mlx_lm, vllm_mlx, psutil) are imported lazily so that
``from forge.engine import scheduler`` is cheap at CLI startup.
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator

from forge.engine.mlx_engine import EngineConfig, GenerationResult, MLXEngine

# ---------------------------------------------------------------------------
# 10.1 Chunked Prefill (Sarathi 2403.02310)
# ---------------------------------------------------------------------------


class WorkKind(str, Enum):
    """Kind of work a scheduler step performs."""

    PREFILL_CHUNK = "prefill_chunk"
    DECODE = "decode"
    IDLE = "idle"


@dataclass
class WorkUnit:
    """A single unit of work produced by the scheduler."""

    kind: WorkKind
    request_id: str
    tokens: list[int] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 0
    is_final_chunk: bool = False
    decode_step: int = 0


@dataclass
class PrefillRequest:
    """A request being processed through chunked prefill."""

    request_id: str
    prompt_tokens: list[int]
    max_tokens: int = 256
    chunk_size: int = 512
    # Runtime state
    prefill_pos: int = 0          # How many prompt tokens have been prefilled
    generated_tokens: list[int] = field(default_factory=list)
    finished: bool = False
    created_at: float = field(default_factory=time.monotonic)

    @property
    def prefill_done(self) -> bool:
        return self.prefill_pos >= len(self.prompt_tokens)

    @property
    def total_chunks(self) -> int:
        if not self.prompt_tokens:
            return 0
        return (len(self.prompt_tokens) + self.chunk_size - 1) // self.chunk_size

    @property
    def current_chunk_index(self) -> int:
        if self.chunk_size <= 0:
            return 0
        return self.prefill_pos // self.chunk_size


@dataclass
class SchedulerStats:
    total_requests: int = 0
    completed_requests: int = 0
    prefill_chunks_emitted: int = 0
    decode_steps_emitted: int = 0
    preemptions: int = 0


class ChunkedPrefillScheduler:
    """Sarathi-style chunked prefill scheduler.

    Long prompts are broken into ``chunk_size`` token chunks. Each step the
    scheduler emits at most one prefill chunk *and* one decode step per
    active request, ensuring that already-decoding requests continue to
    produce tokens while a large prompt is being ingested.

    The scheduler is purely a control-plane object: it yields
    :class:`WorkUnit` items that a runner (typically something that wraps
    ``mlx_lm.stream_generate``) consumes. This keeps the implementation
    independent of any particular MLX internal KV-cache API.
    """

    def __init__(self, chunk_size: int = 512, max_active_requests: int = 8):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.max_active_requests = max_active_requests
        self._queue: list[PrefillRequest] = []
        self._active: list[PrefillRequest] = []
        self.stats = SchedulerStats()

    def add_request(
        self,
        request_id: str,
        prompt_tokens: list[int],
        max_tokens: int = 256,
    ) -> PrefillRequest:
        req = PrefillRequest(
            request_id=request_id,
            prompt_tokens=list(prompt_tokens),
            max_tokens=max_tokens,
            chunk_size=self.chunk_size,
        )
        self._queue.append(req)
        self.stats.total_requests += 1
        return req

    def _promote_queue(self) -> None:
        while self._queue and len(self._active) < self.max_active_requests:
            self._active.append(self._queue.pop(0))

    def pending(self) -> int:
        return len(self._queue) + len(self._active)

    def schedule_step(self) -> Generator[WorkUnit, None, None]:
        """Yield work units for one scheduling tick.

        On each tick we (a) admit queued requests up to ``max_active_requests``,
        (b) emit one prefill chunk for every request that hasn't finished
        prefilling, and (c) emit one decode step for every request that has
        already finished prefilling and hasn't completed.
        """
        self._promote_queue()

        if not self._active:
            yield WorkUnit(kind=WorkKind.IDLE, request_id="")
            return

        for req in list(self._active):
            if req.finished:
                continue

            if not req.prefill_done:
                start = req.prefill_pos
                end = min(start + self.chunk_size, len(req.prompt_tokens))
                chunk = req.prompt_tokens[start:end]
                req.prefill_pos = end
                self.stats.prefill_chunks_emitted += 1
                yield WorkUnit(
                    kind=WorkKind.PREFILL_CHUNK,
                    request_id=req.request_id,
                    tokens=chunk,
                    chunk_index=req.current_chunk_index - 1,
                    total_chunks=req.total_chunks,
                    is_final_chunk=req.prefill_done,
                )
            else:
                self.stats.decode_steps_emitted += 1
                yield WorkUnit(
                    kind=WorkKind.DECODE,
                    request_id=req.request_id,
                    decode_step=len(req.generated_tokens),
                )

    def record_token(self, request_id: str, token: int) -> None:
        """Runner callback: record a newly-decoded token for a request."""
        for req in self._active:
            if req.request_id == request_id:
                req.generated_tokens.append(token)
                if len(req.generated_tokens) >= req.max_tokens:
                    self.finish(request_id)
                return

    def finish(self, request_id: str) -> None:
        for i, req in enumerate(self._active):
            if req.request_id == request_id:
                req.finished = True
                self._active.pop(i)
                self.stats.completed_requests += 1
                return

    def preempt(self, request_id: str) -> PrefillRequest | None:
        """Remove a request from the active set and return it for later resume."""
        for i, req in enumerate(self._active):
            if req.request_id == request_id:
                self.stats.preemptions += 1
                return self._active.pop(i)
        return None


# ---------------------------------------------------------------------------
# 10.2 Interruptible generation
# ---------------------------------------------------------------------------


class SessionState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"


@dataclass
class InterruptibleSession:
    """FastServe-style preemptable generation session.

    Because mlx-lm's internal ``KVCache`` does not expose a cheap
    serialize/deserialize API, we implement save/restore indirectly: the
    session tracks the original prompt and all tokens generated so far.
    When ``resume()`` is called after a ``pause()``, the runner rebuilds
    KV state by running the concatenated prefix back through the engine
    — ideally hitting the radix prefix cache so only the suffix actually
    gets recomputed.

    ``prompt_cache_path`` may point at an mlx-lm prompt-cache safetensors
    file (see :mod:`forge.engine.prompt_cache`) that already contains the
    system prefix. The runner is responsible for loading it when
    ``resume()`` emits a ``RESUME`` event — the session itself stays
    transport-agnostic.
    """

    session_id: str
    prompt_text: str
    prompt_tokens: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    generated_text: str = ""
    max_tokens: int = 256
    prompt_cache_path: str | None = None
    prefix_cache_ref: Any = None  # Opaque handle into a RadixPrefixCache entry
    state: SessionState = SessionState.READY
    paused_at: float = 0.0
    resumed_at: float = 0.0
    pause_count: int = 0

    # ---- control-plane API ----

    def start(self) -> None:
        self.state = SessionState.RUNNING

    def pause(self) -> None:
        """Mark the session as paused. The runner should flush KV + stop."""
        if self.state != SessionState.RUNNING:
            return
        self.state = SessionState.PAUSED
        self.paused_at = time.monotonic()
        self.pause_count += 1

    def resume(self) -> "ResumePlan":
        """Produce a plan for the runner to re-establish state.

        Returns a :class:`ResumePlan` describing what to re-prefill.
        """
        if self.state != SessionState.PAUSED:
            raise RuntimeError(f"cannot resume session in state {self.state}")
        self.state = SessionState.RUNNING
        self.resumed_at = time.monotonic()
        return ResumePlan(
            session_id=self.session_id,
            prompt_tokens=list(self.prompt_tokens),
            generated_tokens=list(self.generated_tokens),
            prompt_cache_path=self.prompt_cache_path,
            remaining_tokens=max(0, self.max_tokens - len(self.generated_tokens)),
            reuse_prefix=self.prefix_cache_ref is not None,
        )

    def record_token(self, token_id: int, token_text: str = "") -> None:
        self.generated_tokens.append(token_id)
        if token_text:
            self.generated_text += token_text
        if len(self.generated_tokens) >= self.max_tokens:
            self.state = SessionState.DONE

    def finish(self) -> None:
        self.state = SessionState.DONE

    # ---- serialization (for external checkpointing) ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt_text": self.prompt_text,
            "prompt_tokens": list(self.prompt_tokens),
            "generated_tokens": list(self.generated_tokens),
            "generated_text": self.generated_text,
            "max_tokens": self.max_tokens,
            "prompt_cache_path": self.prompt_cache_path,
            "state": self.state.value,
            "pause_count": self.pause_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterruptibleSession":
        return cls(
            session_id=data["session_id"],
            prompt_text=data.get("prompt_text", ""),
            prompt_tokens=list(data.get("prompt_tokens", [])),
            generated_tokens=list(data.get("generated_tokens", [])),
            generated_text=data.get("generated_text", ""),
            max_tokens=data.get("max_tokens", 256),
            prompt_cache_path=data.get("prompt_cache_path"),
            state=SessionState(data.get("state", "ready")),
            pause_count=data.get("pause_count", 0),
        )


@dataclass
class ResumePlan:
    """Instructions for a runner re-establishing a paused session."""

    session_id: str
    prompt_tokens: list[int]
    generated_tokens: list[int]
    prompt_cache_path: str | None
    remaining_tokens: int
    reuse_prefix: bool

    @property
    def replay_tokens(self) -> list[int]:
        """Full token sequence that needs a live KV state after resume."""
        return self.prompt_tokens + self.generated_tokens


# ---------------------------------------------------------------------------
# 10.3 Multi-model hot loading with LRU eviction
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    model_path: str
    engine: MLXEngine
    estimated_memory_gb: float
    loaded_at: float
    last_used: float
    use_count: int = 0

    def touch(self) -> None:
        self.last_used = time.monotonic()
        self.use_count += 1


def _estimate_model_memory_gb(model_path: str) -> float:
    """Rough estimate of resident memory for a loaded model.

    Uses the on-disk footprint of the safetensors shards as a proxy. Far
    from perfect — activations, KV, scratch all live on top — but good
    enough for LRU budgeting on a single unified-memory box.
    """
    p = Path(model_path)
    if not p.exists():
        return 0.0
    total = 0
    if p.is_file():
        total = p.stat().st_size
    else:
        for f in p.rglob("*.safetensors"):
            try:
                total += f.stat().st_size
            except OSError:
                pass
        if total == 0:
            for f in p.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
    return total / (1024 ** 3)


class ModelRegistry:
    """LRU registry of hot-loaded MLX models.

    Tracks an approximate resident-memory budget (in GB of unified memory)
    and evicts least-recently-used models when a new load would exceed it.

    Example::

        reg = ModelRegistry(memory_budget_gb=32.0)
        engine = reg.get("./optimized/qwen-7b-q4")
        result = engine.generate("hello")
    """

    def __init__(
        self,
        memory_budget_gb: float = 32.0,
        max_models: int = 4,
        engine_factory: Callable[[EngineConfig], MLXEngine] | None = None,
    ):
        self.memory_budget_gb = memory_budget_gb
        self.max_models = max_models
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._engine_factory = engine_factory or (lambda cfg: MLXEngine(cfg))

    @property
    def resident_gb(self) -> float:
        return sum(m.estimated_memory_gb for m in self._models.values())

    def loaded_paths(self) -> list[str]:
        return list(self._models.keys())

    def get(
        self,
        model_path: str,
        config: EngineConfig | None = None,
    ) -> MLXEngine:
        """Return an engine for ``model_path``, loading it if needed."""
        key = str(Path(model_path))
        if key in self._models:
            entry = self._models[key]
            entry.touch()
            self._models.move_to_end(key)
            return entry.engine

        estimated = _estimate_model_memory_gb(key)
        self._make_room(estimated)

        cfg = config or EngineConfig(model_path=key)
        cfg.model_path = key
        engine = self._engine_factory(cfg)
        engine.load()

        now = time.monotonic()
        entry = LoadedModel(
            model_path=key,
            engine=engine,
            estimated_memory_gb=estimated,
            loaded_at=now,
            last_used=now,
            use_count=1,
        )
        self._models[key] = entry
        return engine

    def evict(self, model_path: str) -> bool:
        key = str(Path(model_path))
        entry = self._models.pop(key, None)
        if entry is None:
            return False
        try:
            entry.engine.unload()
        except Exception:  # noqa: BLE001
            pass
        return True

    def clear(self) -> None:
        for key in list(self._models.keys()):
            self.evict(key)

    def _make_room(self, needed_gb: float) -> None:
        while (
            len(self._models) >= self.max_models
            or (self._models and self.resident_gb + needed_gb > self.memory_budget_gb)
        ):
            if not self._models:
                break
            # OrderedDict front = least-recently-used
            lru_key, _ = next(iter(self._models.items()))
            self.evict(lru_key)

    def format_report(self) -> str:
        lines = [
            "Model Registry",
            "=" * 50,
            f"  Budget:   {self.memory_budget_gb:.1f} GB",
            f"  Resident: {self.resident_gb:.1f} GB across {len(self._models)} model(s)",
        ]
        for m in self._models.values():
            age = time.monotonic() - m.loaded_at
            lines.append(
                f"  - {Path(m.model_path).name:<30} "
                f"{m.estimated_memory_gb:5.1f} GB  "
                f"uses={m.use_count}  age={age:.0f}s"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10.4 PMPD (Prefill-Max, Prefill-Decode) — FP16 prefill, INT4 decode
# ---------------------------------------------------------------------------


@dataclass
class PMPDPlan:
    """Control-plane description of a PMPD runtime switch.

    Experimental: the current MLX-LM runtime cannot change quantization on
    an already-loaded model. This helper therefore prepares *two* engine
    configs — one for the FP16 prefill model, one for the INT4 decode
    model — and documents whether runtime hot-swap is available. A
    downstream scheduler can act on ``hot_swap_supported`` to decide
    whether to fall back to a single-precision path.
    """

    prefill_config: EngineConfig
    decode_config: EngineConfig
    hot_swap_supported: bool
    notes: str = ""


def configure_pmpd(
    prefill_model_path: str,
    decode_model_path: str | None = None,
    base_config: EngineConfig | None = None,
) -> PMPDPlan:
    """Build a PMPD plan for a pair of model checkpoints.

    Args:
        prefill_model_path: Path to the higher-precision (FP16/BF16)
            checkpoint used for prefill.
        decode_model_path: Path to the INT4-quantized checkpoint used for
            decode. If omitted we set ``hot_swap_supported=False`` and both
            phases share ``prefill_model_path`` (pure control-plane stub).
        base_config: Optional template :class:`EngineConfig` whose
            sampler / KV settings are copied into both phases.
    """
    base = base_config or EngineConfig()

    def _clone(path: str) -> EngineConfig:
        return EngineConfig(
            model_path=path,
            max_tokens=base.max_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
            repetition_penalty=base.repetition_penalty,
            draft_model_path=base.draft_model_path,
            num_draft_tokens=base.num_draft_tokens,
            kv_bits=base.kv_bits,
            kv_group_size=base.kv_group_size,
            max_kv_size=base.max_kv_size,
            kv_compression=base.kv_compression,
            kv_eviction=base.kv_eviction,
            kv_budget_ratio=base.kv_budget_ratio,
            enable_prefix_cache=base.enable_prefix_cache,
            prefix_cache_memory_mb=base.prefix_cache_memory_mb,
            prompt_cache_path=base.prompt_cache_path,
            pmpd_mode=True,
        )

    prefill_cfg = _clone(prefill_model_path)
    decode_cfg = _clone(decode_model_path or prefill_model_path)

    hot_swap = decode_model_path is not None and decode_model_path != prefill_model_path
    notes = (
        "PMPD control plane only — mlx_lm does not yet support mid-generation "
        "requantization, so prefill and decode must be separately loaded."
    )
    if not hot_swap:
        notes += " No decode_model_path supplied: running single-precision."

    return PMPDPlan(
        prefill_config=prefill_cfg,
        decode_config=decode_cfg,
        hot_swap_supported=hot_swap,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 10.5 vllm-mlx PagedAttention adapter
# ---------------------------------------------------------------------------


def detect_vllm_mlx() -> bool:
    """Return True if the ``vllm_mlx`` package is importable."""
    try:
        return importlib.util.find_spec("vllm_mlx") is not None
    except (ImportError, ValueError):
        return False


class PagedAttentionAdapter:
    """Thin adapter that prefers ``vllm_mlx`` when available.

    When ``vllm_mlx`` is importable, :meth:`generate` delegates to its
    ``LLM`` class (PagedAttention-style block manager). Otherwise it
    transparently falls back to :class:`MLXEngine`.
    """

    def __init__(self, config: EngineConfig, prefer_vllm: bool = True):
        self.config = config
        self._use_vllm = prefer_vllm and detect_vllm_mlx()
        self._vllm_llm: Any = None
        self._fallback: MLXEngine | None = None

    @property
    def backend(self) -> str:
        return "vllm_mlx" if self._use_vllm else "mlx_engine"

    def load(self) -> None:
        if self._use_vllm:
            try:
                vllm_mlx = importlib.import_module("vllm_mlx")
                LLM = getattr(vllm_mlx, "LLM", None)
                if LLM is None:
                    raise ImportError("vllm_mlx has no LLM class")
                self._vllm_llm = LLM(model=self.config.model_path)
                return
            except Exception:  # noqa: BLE001
                # vllm_mlx present but incompatible — fall back.
                self._use_vllm = False
                self._vllm_llm = None

        self._fallback = MLXEngine(self.config)
        self._fallback.load()

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        if self._use_vllm and self._vllm_llm is not None:
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            try:
                SamplingParams = importlib.import_module("vllm_mlx").SamplingParams
                params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=kwargs.get("top_p", self.config.top_p),
                )
                start = time.monotonic()
                outputs = self._vllm_llm.generate([prompt], params)
                elapsed = time.monotonic() - start
                out = outputs[0] if outputs else None
                text = ""
                n_tokens = 0
                if out is not None and getattr(out, "outputs", None):
                    first = out.outputs[0]
                    text = getattr(first, "text", "")
                    n_tokens = len(getattr(first, "token_ids", []) or [])
                return GenerationResult(
                    text=text,
                    tokens_generated=n_tokens,
                    total_seconds=elapsed,
                    tps=(n_tokens / elapsed) if elapsed > 0 else 0.0,
                )
            except Exception:  # noqa: BLE001
                # Adapter error — demote to fallback for subsequent calls.
                self._use_vllm = False
                if self._fallback is None:
                    self._fallback = MLXEngine(self.config)
                    self._fallback.load()

        if self._fallback is None:
            self._fallback = MLXEngine(self.config)
            self._fallback.load()
        return self._fallback.generate(prompt, **kwargs)

    def unload(self) -> None:
        if self._fallback is not None:
            self._fallback.unload()
            self._fallback = None
        self._vllm_llm = None


# ---------------------------------------------------------------------------
# Convenience: bundle everything a server would need.
# ---------------------------------------------------------------------------


@dataclass
class AdvancedServingContext:
    """All the scheduler-level objects a serving runner may need."""

    scheduler: ChunkedPrefillScheduler | None = None
    registry: ModelRegistry | None = None
    sessions: dict[str, InterruptibleSession] = field(default_factory=dict)
    pmpd_plan: PMPDPlan | None = None
    paged_adapter: PagedAttentionAdapter | None = None

    def any_enabled(self) -> bool:
        return any(
            x is not None
            for x in (
                self.scheduler,
                self.registry,
                self.pmpd_plan,
                self.paged_adapter,
            )
        ) or bool(self.sessions)


def build_context(
    model_path: str,
    *,
    chunked_prefill: bool = False,
    chunk_size: int = 512,
    interruptible: bool = False,
    multi_model: bool = False,
    pmpd: bool = False,
    use_vllm_mlx: bool = False,
    memory_budget_gb: float = 32.0,
    base_config: EngineConfig | None = None,
) -> AdvancedServingContext:
    """Factory used by :func:`forge.pipeline.deployer.serve_advanced`."""
    ctx = AdvancedServingContext()

    if chunked_prefill:
        ctx.scheduler = ChunkedPrefillScheduler(chunk_size=chunk_size)

    if multi_model:
        ctx.registry = ModelRegistry(memory_budget_gb=memory_budget_gb)

    if interruptible:
        # The dict stays empty until real sessions get created — but its
        # presence is how the runner knows to wire up pause/resume handlers.
        ctx.sessions = {}

    if pmpd:
        ctx.pmpd_plan = configure_pmpd(
            prefill_model_path=model_path,
            decode_model_path=None,
            base_config=base_config,
        )

    if use_vllm_mlx:
        ctx.paged_adapter = PagedAttentionAdapter(
            config=base_config or EngineConfig(model_path=model_path),
            prefer_vllm=True,
        )

    return ctx
