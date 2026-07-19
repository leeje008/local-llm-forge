"""MLX-based inference engine with speculative decoding and KV cache optimization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator


@dataclass
class GenerationResult:
    """Result from a generation call."""

    text: str = ""
    tokens_generated: int = 0
    tps: float = 0.0
    ttft_seconds: float = 0.0
    total_seconds: float = 0.0
    speculative_used: bool = False
    prompt_cache_used: bool = False
    prefix_cache_hit: bool = False
    prefix_tokens_saved: int = 0
    kv_compression: str = "none"
    kv_eviction: str = "none"
    attention_backend: str = "default"
    ngram_acceptance_rate: float = 0.0
    ngram_tokens_per_step: float = 0.0


@dataclass
class EngineConfig:
    """Configuration for the MLX inference engine."""

    model_path: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    # Speculative decoding
    draft_model_path: str | None = None
    num_draft_tokens: int = 3
    # KV cache optimization
    kv_bits: int | None = None  # 8 for FP8 KV cache quantization
    kv_group_size: int = 64
    max_kv_size: int | None = None  # Sliding window limit
    # KV cache compression (Phase 6)
    kv_compression: str = "none"   # "none", "turbo", "fp8"
    kv_eviction: str = "none"      # "none", "sliding", "h2o", "ada_kv"
    kv_budget_ratio: float = 0.2   # For H2O/Ada-KV: fraction of tokens to keep
    # Prefix cache (Phase 6)
    enable_prefix_cache: bool = False
    prefix_cache_memory_mb: float = 2048.0
    # Prompt cache
    prompt_cache_path: str | None = None
    # PMPD — prefill FP16, decode INT4 (Phase 10.4, experimental).
    # Control-plane flag only; the actual quant switch is orchestrated by
    # forge.engine.scheduler.configure_pmpd().
    pmpd_mode: bool = False
    # Phase 12: Attention backend selection
    attention_backend: str = "default"  # "default" | "mfa" | "auto"
    mfa_min_seq_len: int = 2048  # auto mode: use MFA when seq_len >= this
    # Phase 7: N-gram self-speculative decoding (opt-in path; default unchanged)
    ngram_spec: bool = False
    ngram_order: int = 3
    ngram_max_draft: int = 5
    use_adaptive_k: bool = True
    adaptive_initial_k: int = 3


class MLXEngine:
    """High-level MLX inference engine wrapping mlx_lm with optimizations."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._draft_model = None
        self._loaded = False
        # Phase 6: KV cache managers
        self._prefix_cache = None
        self._h2o_manager = None
        self._ada_kv_manager = None
        self._turbo_compressor = None
        # Phase 12: Attention backend state
        self._mfa_available: bool | None = None
        self._attention_backend_active: str = "default"

    def load(self) -> None:
        """Load model (and optional draft model) into memory."""
        import mlx_lm

        self._model, self._tokenizer = mlx_lm.load(self.config.model_path)

        if self.config.draft_model_path:
            self._draft_model, _ = mlx_lm.load(self.config.draft_model_path)

        # Phase 6: Initialize KV cache managers
        self._init_kv_managers()

        self._loaded = True

        # Phase 12: Resolve attention backend after model is loaded
        self._attention_backend_active = self._init_attention_backend()

    def _init_attention_backend(self) -> str:
        """Resolve the active attention backend based on config and availability.

        In ``auto`` mode, the returned value indicates whether MFA *may* be used;
        the final per-call decision is additionally gated by sequence length via
        :meth:`_should_use_mfa` (see ``mfa_min_seq_len``).
        """
        requested = self.config.attention_backend
        if requested == "default":
            return "default"

        try:
            import mlx_mfa  # noqa: F401
            self._mfa_available = True
        except Exception:
            self._mfa_available = False

        if requested == "mfa":
            if self._mfa_available:
                print("[forge.engine] attention backend: mlx-mfa (metal-flash-attention) enabled")
                return "mfa"
            return "default"  # silent fallback
        if requested == "auto":
            return "mfa" if self._mfa_available else "default"
        return "default"

    def _should_use_mfa(self, seq_len: int) -> bool:
        """Per-call decision on whether to dispatch MFA kernel."""
        if self._attention_backend_active == "default" or not self._mfa_available:
            return False
        if self.config.attention_backend == "mfa":
            return True
        if self.config.attention_backend == "auto":
            return seq_len >= self.config.mfa_min_seq_len
        return False

    def _init_kv_managers(self) -> None:
        """Initialize KV cache compression and eviction managers."""
        cfg = self.config

        # TurboQuant compressor
        if cfg.kv_compression == "turbo":
            from forge.engine.kv_cache import TurboQuantCompressor, TurboQuantConfig
            self._turbo_compressor = TurboQuantCompressor(TurboQuantConfig(bits=3))

        # H2O eviction
        if cfg.kv_eviction == "h2o":
            from forge.engine.kv_cache import H2OConfig, H2OEvictionManager
            self._h2o_manager = H2OEvictionManager(
                H2OConfig(budget_ratio=cfg.kv_budget_ratio)
            )

        # Ada-KV (per-head adaptive H2O)
        if cfg.kv_eviction == "ada_kv":
            from forge.engine.kv_cache import AdaKVConfig, AdaKVManager
            self._ada_kv_manager = AdaKVManager(
                AdaKVConfig(total_budget_ratio=cfg.kv_budget_ratio)
            )

        # Prefix cache
        if cfg.enable_prefix_cache:
            from forge.engine.prefix_cache import RadixPrefixCache
            self._prefix_cache = RadixPrefixCache(
                max_memory_mb=cfg.prefix_cache_memory_mb
            )

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._tokenizer = None
        self._draft_model = None
        self._prefix_cache = None
        self._h2o_manager = None
        self._ada_kv_manager = None
        self._turbo_compressor = None
        self._loaded = False
        import gc
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from a prompt. Returns complete result."""
        if not self._loaded:
            self.load()

        # Phase 7: opt-in N-gram self-speculative path. The default generation
        # path below is left untouched when ngram_spec is disabled.
        if self.config.ngram_spec:
            return self._generate_ngram_spec(
                prompt,
                kwargs.get("max_tokens", self.config.max_tokens),
                kwargs.get("temperature", self.config.temperature),
            )

        import mlx_lm

        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)

        result = GenerationResult()
        result.speculative_used = self._draft_model is not None

        # Build sampler from mlx_lm
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)

        start = time.monotonic()
        first_token_time = None
        tokens = []

        # Build extra kwargs for KV cache optimization
        extra_kwargs = {}
        if self.config.kv_bits:
            extra_kwargs["kv_bits"] = self.config.kv_bits
            extra_kwargs["kv_group_size"] = self.config.kv_group_size
        if self.config.max_kv_size:
            extra_kwargs["max_kv_size"] = self.config.max_kv_size

        for response in mlx_lm.stream_generate(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            draft_model=self._draft_model,
            sampler=sampler,
            **extra_kwargs,
        ):
            if first_token_time is None:
                first_token_time = time.monotonic()
            tokens.append(response.text)

        elapsed = time.monotonic() - start
        result.text = "".join(tokens)
        result.tokens_generated = len(tokens)
        result.total_seconds = elapsed
        result.ttft_seconds = (first_token_time - start) if first_token_time else 0
        result.tps = result.tokens_generated / elapsed if elapsed > 0 else 0
        result.attention_backend = self._attention_backend_active

        return result

    def _generate_ngram_spec(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> GenerationResult:
        """N-gram self-speculative greedy decode (opt-in ``--ngram-spec`` path).

        Runs a custom decode loop driven by :class:`NGramSpeculativeDecoder`
        (N-gram drafting + Adaptive-K) against the real MLX model. Verification
        is greedy, so the output matches plain greedy decoding while accepting
        matched draft tokens. ``temperature`` is accepted for interface parity
        but this path is greedy by construction.
        """
        import mlx.core as mx

        from forge.engine.speculative import (
            AdaptiveKConfig,
            AdaptiveKController,
            NGramDrafter,
            NGramSpeculativeDecoder,
        )

        cfg = self.config
        _ = temperature  # greedy path; kept for signature parity

        prompt_ids = list(self._tokenizer.encode(prompt))

        def forward_fn(seq_tokens: list[int]):
            logits = self._model(mx.array([seq_tokens]))  # [1, L, vocab]
            return logits[0]

        decoder = NGramSpeculativeDecoder(
            ngram=NGramDrafter(n=cfg.ngram_order, max_draft=cfg.ngram_max_draft),
            adaptive_k=AdaptiveKController(
                AdaptiveKConfig(initial_k=cfg.adaptive_initial_k)
            ),
            use_adaptive_k=cfg.use_adaptive_k,
            fixed_k=cfg.num_draft_tokens,
        )

        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        start = time.monotonic()
        spec = decoder.generate(
            prompt_ids, forward_fn, max_tokens=max_tokens, eos_token_id=eos_id
        )
        elapsed = time.monotonic() - start

        return GenerationResult(
            text=self._tokenizer.decode(spec.tokens),
            tokens_generated=len(spec.tokens),
            total_seconds=elapsed,
            tps=len(spec.tokens) / elapsed if elapsed > 0 else 0.0,
            speculative_used=True,
            attention_backend=self._attention_backend_active,
            ngram_acceptance_rate=round(spec.acceptance_rate, 3),
            ngram_tokens_per_step=round(spec.avg_tokens_per_step, 3),
        )

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, GenerationResult]:
        """Stream tokens as they're generated. Yields text chunks."""
        if not self._loaded:
            self.load()

        # Phase 7: opt-in N-gram self-speculative path. Streams the completed
        # speculative result as a single chunk; the default path is unchanged.
        if self.config.ngram_spec:
            result = self._generate_ngram_spec(
                prompt,
                kwargs.get("max_tokens", self.config.max_tokens),
                kwargs.get("temperature", self.config.temperature),
            )
            yield result.text
            return result

        import mlx_lm

        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)

        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)

        extra_kwargs = {}
        if self.config.kv_bits:
            extra_kwargs["kv_bits"] = self.config.kv_bits
            extra_kwargs["kv_group_size"] = self.config.kv_group_size
        if self.config.max_kv_size:
            extra_kwargs["max_kv_size"] = self.config.max_kv_size

        start = time.monotonic()
        count = 0

        for response in mlx_lm.stream_generate(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            draft_model=self._draft_model,
            sampler=sampler,
            **extra_kwargs,
        ):
            count += 1
            yield response.text

        elapsed = time.monotonic() - start
        return GenerationResult(
            tokens_generated=count,
            total_seconds=elapsed,
            tps=count / elapsed if elapsed > 0 else 0,
            speculative_used=self._draft_model is not None,
            attention_backend=self._attention_backend_active,
        )

    def benchmark(
        self,
        prompts: list[str] | None = None,
        max_tokens: int = 100,
    ) -> dict:
        """Run quick internal benchmark."""
        if prompts is None:
            prompts = [
                "What is 2+2?",
                "Explain recursion briefly.",
                "Write a Python hello world.",
            ]

        results = []
        for prompt in prompts:
            r = self.generate(prompt, max_tokens=max_tokens)
            results.append({
                "prompt": prompt[:50],
                "tps": round(r.tps, 1),
                "ttft": round(r.ttft_seconds, 3),
                "tokens": r.tokens_generated,
            })

        tps_values = [r["tps"] for r in results]
        return {
            "avg_tps": round(sum(tps_values) / len(tps_values), 1),
            "min_tps": round(min(tps_values), 1),
            "max_tps": round(max(tps_values), 1),
            "speculative": self._draft_model is not None,
            "results": results,
        }
