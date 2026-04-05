"""Structured output via grammar-constrained decoding (XGrammar-2 / llguidance).

Provides a backend-agnostic logits mask producer that constrains LLM generation
to a JSON Schema or context-free grammar. Designed to plug into forge's MLX
engine as a logits post-processor, and to compose with speculative decoding by
applying the same mask to both drafter and target.

Backends supported (first available wins):
  1. xgrammar (Python package: xgrammar) -- preferred, arXiv 2601.04426.
  2. llguidance (Python package: llguidance) -- fallback, Microsoft Guidance AI.
  3. outlines (Python package: outlines) -- second fallback, mature ecosystem.
  4. "dummy"  -- pass-through no-op when nothing is installed (dev-friendly).

This module is Phase 12 Tier S Item 5. It deliberately avoids touching any
existing forge module: integration into MLXEngine / speculative drafter is
performed by the parent task via the public hooks exposed here
(``GrammarMasker.mask``, ``GrammarMasker.step``, ``ToolPreamblePin``).

All heavy backend imports are deferred to method bodies so this file imports
cleanly on a stock environment with none of the optional dependencies.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GrammarSpec",
    "StructuredDecodingConfig",
    "StructuredDecodingStats",
    "CompiledGrammar",
    "GrammarCompiler",
    "GrammarMasker",
    "ToolPreamblePin",
    "detect_structured_backends",
    "select_backend",
    "load_grammar_from_file",
    "format_structured_decoding_report",
    "example_json_schema",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GrammarSpec:
    """Description of a grammar to constrain decoding.

    Attributes:
        kind: One of ``"json_schema"``, ``"regex"``, ``"ebnf"``, ``"json"``.
            ``"json"`` means "any valid JSON value"; ``"json_schema"`` means
            the ``source`` field is a JSON Schema document.
        source: The schema / regex / grammar text itself.
        root_rule: Optional root rule name -- only meaningful for
            ``kind == "ebnf"``.
    """

    kind: str
    source: str
    root_rule: str | None = None

    def preview(self, n: int = 200) -> str:
        """Return a truncated preview of the grammar source for reports."""
        src = self.source.strip()
        if len(src) <= n:
            return src
        return src[:n] + "..."


@dataclass
class StructuredDecodingConfig:
    """User-facing configuration for structured decoding.

    Attributes:
        grammar: The grammar to apply; ``None`` disables structured decoding.
        backend: Either ``"auto"`` or an explicit backend name
            (``"xgrammar"``, ``"llguidance"``, ``"outlines"``, ``"dummy"``).
        apply_to_drafter: Phase 12 Tier S insight -- when composing with
            speculative decoding, the drafter should share the same mask as
            the target model. Otherwise the target rejects nearly every draft
            token in a constrained region, destroying the speedup.
        temperature_in_constrained_regions: Suggested sampling temperature in
            regions where the grammar allows only a handful of tokens. The
            default ``0.0`` means "be greedy where the grammar is tight".
        cache_compiled_grammars: Reuse compiled grammars across requests.
        tool_preamble_tokens: Tokens that should be pinned at the start of
            every request (tool definitions, system prompts). The prefix cache
            uses this to guarantee these blocks are never evicted.
    """

    grammar: GrammarSpec | None = None
    backend: str = "auto"
    apply_to_drafter: bool = True
    temperature_in_constrained_regions: float = 0.0
    cache_compiled_grammars: bool = True
    tool_preamble_tokens: list[int] | None = None


@dataclass
class StructuredDecodingStats:
    """Runtime stats collected while a ``GrammarMasker`` is live."""

    tokens_masked: int = 0
    mask_compute_time_us: float = 0.0
    backend_used: str = "dummy"
    grammar_compile_time_ms: float = 0.0

    def record_mask(self, elapsed_us: float) -> None:
        self.tokens_masked += 1
        self.mask_compute_time_us += elapsed_us

    @property
    def avg_mask_us(self) -> float:
        if self.tokens_masked == 0:
            return 0.0
        return self.mask_compute_time_us / self.tokens_masked


@dataclass
class CompiledGrammar:
    """An opaque, backend-specific compiled grammar object."""

    backend: str
    impl: Any
    compile_time_ms: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


_BACKEND_PRIORITY = ("xgrammar", "llguidance", "outlines")


def detect_structured_backends() -> dict[str, bool]:
    """Probe the environment for installed structured-decoding backends.

    Returns a mapping ``{backend_name: is_installed}``. Uses
    :func:`importlib.util.find_spec` so no side-effect imports happen.
    """
    result: dict[str, bool] = {}
    for name in _BACKEND_PRIORITY:
        try:
            result[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            result[name] = False
    return result


def select_backend(config: StructuredDecodingConfig) -> str:
    """Resolve ``config.backend`` to a concrete backend name.

    ``"auto"`` picks the first installed backend in priority order, falling
    back to ``"dummy"`` if nothing is available. An explicit backend is
    returned as-is if installed (or if it is ``"dummy"``), otherwise this
    logs a warning and returns ``"dummy"`` so the pipeline never crashes on
    a missing optional dependency.
    """
    available = detect_structured_backends()
    requested = (config.backend or "auto").lower()

    if requested == "auto":
        for name in _BACKEND_PRIORITY:
            if available.get(name):
                return name
        return "dummy"

    if requested == "dummy":
        return "dummy"

    if requested in available and available[requested]:
        return requested

    logger.warning(
        "Structured decoding backend %r requested but not installed; "
        "falling back to dummy.",
        requested,
    )
    return "dummy"


# ---------------------------------------------------------------------------
# Grammar compiler
# ---------------------------------------------------------------------------


class GrammarCompiler:
    """Compile :class:`GrammarSpec` objects into backend-specific matchers.

    The compiler is backend-specific (constructed with a resolved backend
    name), but hides all backend differences behind :meth:`compile`. Compiled
    grammars are cached by ``(kind, source)`` so repeated requests with the
    same schema pay the compile cost only once.
    """

    def __init__(
        self,
        backend: str,
        vocab_size: int,
        tokenizer: Any | None = None,
    ) -> None:
        self.backend = backend
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self._cache: dict[tuple[str, str], CompiledGrammar] = {}

    # -- public API -------------------------------------------------------

    def compile(self, spec: GrammarSpec) -> CompiledGrammar:
        """Compile ``spec`` using this compiler's backend.

        If the chosen backend raises at import or compile time, this method
        logs the failure and transparently falls back to the dummy backend
        so the caller always receives a usable :class:`CompiledGrammar`.
        """
        key = (spec.kind, spec.source)
        if key in self._cache:
            return self._cache[key]

        start = time.perf_counter()
        compiled: CompiledGrammar | None = None
        try:
            if self.backend == "xgrammar":
                compiled = self._compile_xgrammar(spec)
            elif self.backend == "llguidance":
                compiled = self._compile_llguidance(spec)
            elif self.backend == "outlines":
                compiled = self._compile_outlines(spec)
            else:
                compiled = self._compile_dummy(spec)
        except Exception as exc:  # noqa: BLE001 -- we want to swallow all
            logger.warning(
                "%s grammar compile failed (%s); falling back to dummy.",
                self.backend,
                exc,
            )
            compiled = self._compile_dummy(spec)

        if compiled is None:
            compiled = self._compile_dummy(spec)

        compiled.compile_time_ms = (time.perf_counter() - start) * 1000.0
        self._cache[key] = compiled
        return compiled

    # -- backend implementations -----------------------------------------

    def _compile_xgrammar(self, spec: GrammarSpec) -> CompiledGrammar:
        """Compile via the ``xgrammar`` package.

        ``xgrammar``'s public API (as of the arXiv 2601.04426 reference
        release) exposes ``GrammarCompiler`` and ``TokenizerInfo``. We call
        those conservatively and wrap any signature change in a try/except
        that demotes the compiler to the dummy backend.
        """
        try:
            import xgrammar as xgr  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("xgrammar not installed") from exc

        tokenizer_info = None
        if self.tokenizer is not None and hasattr(xgr, "TokenizerInfo"):
            try:
                tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                    self.tokenizer, vocab_size=self.vocab_size
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("xgrammar TokenizerInfo fallback: %s", exc)
                tokenizer_info = None

        compiler_cls = getattr(xgr, "GrammarCompiler", None)
        if compiler_cls is None:
            raise RuntimeError("xgrammar.GrammarCompiler not found")

        xgr_compiler = compiler_cls(tokenizer_info) if tokenizer_info else compiler_cls()

        if spec.kind in ("json_schema",):
            impl = xgr_compiler.compile_json_schema(spec.source)
        elif spec.kind == "json":
            impl = xgr_compiler.compile_builtin_json_grammar()
        elif spec.kind == "regex":
            # xgrammar exposes compile_regex in newer releases.
            fn = getattr(xgr_compiler, "compile_regex", None)
            if fn is None:
                raise RuntimeError("xgrammar has no compile_regex in this version")
            impl = fn(spec.source)
        elif spec.kind == "ebnf":
            impl = xgr_compiler.compile_grammar(spec.source, root_rule=spec.root_rule)
        else:
            raise ValueError(f"Unsupported grammar kind for xgrammar: {spec.kind!r}")

        return CompiledGrammar(
            backend="xgrammar",
            impl=impl,
            compile_time_ms=0.0,
            metadata={"vocab_size": self.vocab_size, "kind": spec.kind},
        )

    def _compile_llguidance(self, spec: GrammarSpec) -> CompiledGrammar:
        """Compile via the ``llguidance`` package.

        llguidance accepts a "llguidance grammar" dict with a ``grammars``
        list. JSON Schema gets wrapped under the ``json_schema`` key; regex
        goes through the ``regex`` key; raw EBNF through ``lark_grammar``.
        """
        try:
            import llguidance  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("llguidance not installed") from exc

        if spec.kind == "json_schema":
            grammar = {"grammars": [{"json_schema": json.loads(spec.source)}]}
        elif spec.kind == "json":
            grammar = {"grammars": [{"json_schema": {"type": "object"}}]}
        elif spec.kind == "regex":
            grammar = {"grammars": [{"regex": spec.source}]}
        elif spec.kind == "ebnf":
            grammar = {"grammars": [{"lark_grammar": spec.source}]}
        else:
            raise ValueError(f"Unsupported grammar kind for llguidance: {spec.kind!r}")

        tok = None
        if self.tokenizer is not None and hasattr(llguidance, "LLTokenizer"):
            try:
                tok = llguidance.LLTokenizer(self.tokenizer)
            except Exception as exc:  # noqa: BLE001
                logger.debug("llguidance LLTokenizer fallback: %s", exc)

        matcher_cls = getattr(llguidance, "LLMatcher", None) or getattr(
            llguidance, "LLInterpreter", None
        )
        if matcher_cls is None:
            raise RuntimeError("llguidance matcher class not found")

        impl = matcher_cls(tok, json.dumps(grammar)) if tok else matcher_cls(json.dumps(grammar))

        return CompiledGrammar(
            backend="llguidance",
            impl=impl,
            compile_time_ms=0.0,
            metadata={"vocab_size": self.vocab_size, "kind": spec.kind},
        )

    def _compile_outlines(self, spec: GrammarSpec) -> CompiledGrammar:
        """Compile via the ``outlines`` package (outlines.fsm)."""
        try:
            import outlines  # type: ignore[import-not-found]  # noqa: F401
            from outlines.fsm import guide as _guide  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("outlines not installed") from exc

        if self.tokenizer is None:
            raise RuntimeError("outlines backend requires a tokenizer")

        if spec.kind == "json_schema":
            try:
                from outlines.fsm.json_schema import build_regex_from_schema  # type: ignore
            except ImportError as exc:
                raise RuntimeError("outlines.fsm.json_schema unavailable") from exc
            regex = build_regex_from_schema(spec.source)
            impl = _guide.RegexGuide(regex, self.tokenizer)
        elif spec.kind == "regex":
            impl = _guide.RegexGuide(spec.source, self.tokenizer)
        elif spec.kind == "json":
            impl = _guide.RegexGuide(r".*", self.tokenizer)
        elif spec.kind == "ebnf":
            impl = _guide.CFGGuide(spec.source, self.tokenizer)
        else:
            raise ValueError(f"Unsupported grammar kind for outlines: {spec.kind!r}")

        return CompiledGrammar(
            backend="outlines",
            impl=impl,
            compile_time_ms=0.0,
            metadata={"vocab_size": self.vocab_size, "kind": spec.kind},
        )

    def _compile_dummy(self, spec: GrammarSpec) -> CompiledGrammar:
        """Pass-through compiler. Always succeeds, does no masking at runtime."""
        return CompiledGrammar(
            backend="dummy",
            impl={"kind": spec.kind, "source": spec.source},
            compile_time_ms=0.0,
            metadata={"vocab_size": self.vocab_size, "kind": spec.kind, "noop": True},
        )


# ---------------------------------------------------------------------------
# Runtime masker
# ---------------------------------------------------------------------------


class GrammarMasker:
    """Per-request grammar state machine + logits masker.

    A :class:`GrammarMasker` is constructed once per request from a
    :class:`CompiledGrammar` and driven token-by-token by the inference
    engine. The core contract is:

    1. Call :meth:`mask` with the current logits to get a masked copy that
       forbids tokens not allowed by the grammar.
    2. Sample a token from the masked logits.
    3. Call :meth:`step` with the sampled token to advance the grammar FSM.

    For speculative decoding, apply the same masker to both drafter and
    target logits, then commit the accepted tokens with :meth:`step`.

    MLX is imported lazily inside methods so the module itself stays cheap
    to import on systems without MLX (e.g., during unit tests in CI).
    """

    def __init__(self, compiled: CompiledGrammar, vocab_size: int) -> None:
        self.compiled = compiled
        self.vocab_size = vocab_size
        self.backend = compiled.backend
        self.stats = StructuredDecodingStats(
            backend_used=compiled.backend,
            grammar_compile_time_ms=compiled.compile_time_ms,
        )
        self._matcher: Any = None
        self._initialized = False
        self.reset()

    # -- public API -------------------------------------------------------

    def reset(self) -> None:
        """Rewind the grammar FSM to its initial state."""
        try:
            self._matcher = self._build_matcher()
            self._initialized = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to build %s matcher (%s); using dummy passthrough.",
                self.backend,
                exc,
            )
            self._matcher = None
            self.backend = "dummy"
            self.stats.backend_used = "dummy"
            self._initialized = True

    def mask(self, token_history: list[int], logits: Any) -> Any:
        """Return a masked copy of ``logits`` allowed by the grammar.

        Args:
            token_history: Tokens already generated for this request. Not
                strictly needed by matcher-based backends (they track state
                internally), but supplied for dummy/future backends that may
                want to rebuild FSM state lazily.
            logits: An ``mx.array`` of shape ``(vocab_size,)`` or
                ``(batch, vocab_size)``. For the dummy backend this is
                returned unchanged.
        """
        start = time.perf_counter()
        try:
            if self.backend == "dummy" or self._matcher is None:
                out = logits
            elif self.backend == "xgrammar":
                out = self._mask_xgrammar(logits)
            elif self.backend == "llguidance":
                out = self._mask_llguidance(logits)
            elif self.backend == "outlines":
                out = self._mask_outlines(logits, token_history)
            else:
                out = logits
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s mask step failed (%s); returning unmasked logits.",
                self.backend,
                exc,
            )
            out = logits
        elapsed_us = (time.perf_counter() - start) * 1e6
        self.stats.record_mask(elapsed_us)
        return out

    def step(self, token: int) -> bool:
        """Advance the grammar state after ``token`` has been committed.

        Returns ``False`` if the backend reports the token was not allowed
        -- which should never happen if :meth:`mask` was applied correctly
        and is therefore logged as a programming error.
        """
        if self.backend == "dummy" or self._matcher is None:
            return True
        try:
            if self.backend == "xgrammar":
                ok = bool(self._matcher.accept_token(int(token)))
            elif self.backend == "llguidance":
                fn = getattr(self._matcher, "consume_token", None) or getattr(
                    self._matcher, "advance", None
                )
                ok = bool(fn(int(token))) if fn else True
            elif self.backend == "outlines":
                # outlines guides are stateless; we track cursor externally.
                self._outlines_state = self._matcher.get_next_state(
                    getattr(self, "_outlines_state", 0), int(token)
                )
                ok = True
            else:
                ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s step failed for token %d: %s", self.backend, token, exc)
            ok = True
        if not ok:
            logger.error(
                "GrammarMasker.step rejected token %d -- mask was probably "
                "not applied before sampling.",
                token,
            )
        return ok

    # -- matcher construction --------------------------------------------

    def _build_matcher(self) -> Any:
        """Instantiate a per-request matcher from the compiled grammar."""
        if self.backend == "xgrammar":
            try:
                import xgrammar as xgr  # type: ignore[import-not-found]
            except ImportError:
                return None
            matcher_cls = getattr(xgr, "GrammarMatcher", None)
            if matcher_cls is None:
                return None
            return matcher_cls(self.compiled.impl)
        if self.backend == "llguidance":
            # llguidance matchers are single-use; rebuild per request.
            try:
                return self.compiled.impl.clone()
            except Exception:  # noqa: BLE001
                return self.compiled.impl
        if self.backend == "outlines":
            self._outlines_state = 0
            return self.compiled.impl
        return None

    # -- per-backend mask paths ------------------------------------------

    def _mask_xgrammar(self, logits: Any) -> Any:
        import mlx.core as mx  # noqa: PLC0415

        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            torch = None  # type: ignore[assignment]

        # xgrammar writes a bit-packed int32 bitmask of shape (vocab/32,).
        bitmask_len = (self.vocab_size + 31) // 32
        if torch is not None:
            bitmask = torch.zeros(bitmask_len, dtype=torch.int32)
            self._matcher.fill_next_token_bitmask(bitmask)
            # Unpack bitmask to a boolean array of vocab_size.
            import numpy as np  # noqa: PLC0415

            bits = np.unpackbits(
                bitmask.numpy().view("uint8"), bitorder="little"
            )[: self.vocab_size].astype(bool)
        else:
            # Fallback: ask for a plain list of allowed token ids.
            allowed = getattr(self._matcher, "find_next_token_bitmask", None)
            if allowed is None:
                return logits
            import numpy as np  # noqa: PLC0415

            bits = np.zeros(self.vocab_size, dtype=bool)
            for tok in allowed():
                bits[tok] = True

        mask = mx.array(bits)
        neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        return mx.where(mask, logits, neg_inf)

    def _mask_llguidance(self, logits: Any) -> Any:
        import mlx.core as mx  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        # llguidance exposes compute_mask / get_mask returning bytes or list.
        fn = (
            getattr(self._matcher, "compute_mask", None)
            or getattr(self._matcher, "get_mask", None)
            or getattr(self._matcher, "mask", None)
        )
        if fn is None:
            return logits
        raw = fn()
        if isinstance(raw, (bytes, bytearray)):
            bits = np.unpackbits(
                np.frombuffer(raw, dtype=np.uint8), bitorder="little"
            )[: self.vocab_size].astype(bool)
        else:
            bits = np.zeros(self.vocab_size, dtype=bool)
            for tok in raw:
                bits[int(tok)] = True
        mask = mx.array(bits)
        neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        return mx.where(mask, logits, neg_inf)

    def _mask_outlines(self, logits: Any, token_history: list[int]) -> Any:
        import mlx.core as mx  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        state = getattr(self, "_outlines_state", 0)
        try:
            allowed = self._matcher.get_next_instruction(state).tokens
        except Exception as exc:  # noqa: BLE001
            logger.debug("outlines get_next_instruction failed: %s", exc)
            return logits
        bits = np.zeros(self.vocab_size, dtype=bool)
        for tok in allowed:
            if 0 <= int(tok) < self.vocab_size:
                bits[int(tok)] = True
        mask = mx.array(bits)
        neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        return mx.where(mask, logits, neg_inf)


# ---------------------------------------------------------------------------
# Tool preamble pinning
# ---------------------------------------------------------------------------


class ToolPreamblePin:
    """A pinned prefix of tokens representing tool definitions / system prompt.

    Phase 12 Tier S research emphasizes that tool-calling systems frequently
    re-send the same (long) tool-definition block at the start of every
    request. Caching that block in the prefix cache and **guaranteeing** it
    is never evicted dramatically reduces TTFT for agentic workloads.

    This class is a tiny descriptor -- actual pinning is enforced by
    :mod:`forge.engine.prefix_cache` via the ``matches_prefix`` hook.
    """

    def __init__(self, preamble_tokens: list[int], hint: str = "") -> None:
        self.preamble_tokens: list[int] = list(preamble_tokens)
        self.hint = hint

    def __len__(self) -> int:
        return len(self.preamble_tokens)

    def matches_prefix(self, token_ids: list[int]) -> int:
        """Return the length of the pinned prefix that matches ``token_ids``.

        Returns ``0`` if the request does not start with the pinned block.
        A non-zero return is a signal to the prefix cache to treat those
        first tokens as pinned (never evicted).
        """
        n = 0
        for a, b in zip(self.preamble_tokens, token_ids):
            if a != b:
                break
            n += 1
        # Only count as a "match" if the *entire* preamble is present.
        return n if n == len(self.preamble_tokens) else 0

    def to_dict(self) -> dict:
        return {"preamble_tokens": list(self.preamble_tokens), "hint": self.hint}

    @classmethod
    def from_dict(cls, data: dict) -> "ToolPreamblePin":
        return cls(
            preamble_tokens=list(data.get("preamble_tokens", [])),
            hint=str(data.get("hint", "")),
        )


# ---------------------------------------------------------------------------
# File loader and reporting helpers
# ---------------------------------------------------------------------------


def load_grammar_from_file(path: str | Path) -> GrammarSpec:
    """Load a grammar from disk, inferring ``kind`` from the extension.

    Supported extensions:
      * ``.json``   -- treated as a JSON Schema document (``kind="json_schema"``).
      * ``.ebnf``, ``.lark`` -- EBNF / Lark context-free grammar.
      * ``.re``, ``.regex`` -- regular expression.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Grammar file not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix == ".json":
        # Validate it parses as JSON, but keep the original text for backends
        # that want to re-parse it themselves (e.g., xgrammar).
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} is not valid JSON: {exc}") from exc
        return GrammarSpec(kind="json_schema", source=text)
    if suffix in (".ebnf", ".lark"):
        return GrammarSpec(kind="ebnf", source=text)
    if suffix in (".re", ".regex"):
        return GrammarSpec(kind="regex", source=text.strip())

    raise ValueError(
        f"Cannot infer grammar kind from extension {suffix!r}; "
        "use .json, .ebnf, .lark, .re, or .regex."
    )


def format_structured_decoding_report(
    config: StructuredDecodingConfig,
    stats: StructuredDecodingStats | None = None,
) -> str:
    """Return a human-readable, multi-line summary of structured decoding."""
    available = detect_structured_backends()
    selected = select_backend(config)

    lines: list[str] = []
    lines.append("Structured Decoding Report")
    lines.append("=" * 40)
    lines.append("Backends detected:")
    for name in _BACKEND_PRIORITY:
        mark = "yes" if available.get(name) else "no "
        lines.append(f"  [{mark}] {name}")
    lines.append(f"  [yes] dummy  (always available)")
    lines.append(f"Backend requested: {config.backend}")
    lines.append(f"Backend selected:  {selected}")
    lines.append("")
    lines.append("Config:")
    lines.append(f"  apply_to_drafter            = {config.apply_to_drafter}")
    lines.append(
        f"  temperature_in_constrained  = {config.temperature_in_constrained_regions}"
    )
    lines.append(f"  cache_compiled_grammars     = {config.cache_compiled_grammars}")
    if config.tool_preamble_tokens is not None:
        lines.append(
            f"  tool_preamble_tokens        = <{len(config.tool_preamble_tokens)} tokens>"
        )
    else:
        lines.append("  tool_preamble_tokens        = (none)")
    lines.append("")

    if config.grammar is not None:
        lines.append(f"Grammar kind: {config.grammar.kind}")
        if config.grammar.root_rule:
            lines.append(f"Root rule:    {config.grammar.root_rule}")
        lines.append("Grammar preview (first 200 chars):")
        for row in config.grammar.preview(200).splitlines() or [""]:
            lines.append(f"  | {row}")
    else:
        lines.append("Grammar: (none)")

    if stats is not None:
        lines.append("")
        lines.append("Runtime stats:")
        lines.append(f"  backend_used             = {stats.backend_used}")
        lines.append(f"  tokens_masked            = {stats.tokens_masked}")
        lines.append(f"  avg mask time (us/token) = {stats.avg_mask_us:.2f}")
        lines.append(
            f"  total mask time (ms)     = {stats.mask_compute_time_us / 1000.0:.2f}"
        )
        lines.append(
            f"  grammar compile time (ms)= {stats.grammar_compile_time_ms:.2f}"
        )

    return "\n".join(lines)


def example_json_schema() -> GrammarSpec:
    """Return a tiny reference JSON Schema for docs, tests, and smoke checks.

    Shape: ``{"name": str, "age": int}`` with both fields required.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    return GrammarSpec(kind="json_schema", source=json.dumps(schema))
