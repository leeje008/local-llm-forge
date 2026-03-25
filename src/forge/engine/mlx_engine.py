"""MLX-based inference engine with speculative decoding and KV cache optimization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
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
    # Prompt cache
    prompt_cache_path: str | None = None


class MLXEngine:
    """High-level MLX inference engine wrapping mlx_lm with optimizations."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._draft_model = None
        self._loaded = False

    def load(self) -> None:
        """Load model (and optional draft model) into memory."""
        import mlx_lm

        self._model, self._tokenizer = mlx_lm.load(self.config.model_path)

        if self.config.draft_model_path:
            self._draft_model, _ = mlx_lm.load(self.config.draft_model_path)

        self._loaded = True

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._tokenizer = None
        self._draft_model = None
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

        return result

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, GenerationResult]:
        """Stream tokens as they're generated. Yields text chunks."""
        if not self._loaded:
            self.load()

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
