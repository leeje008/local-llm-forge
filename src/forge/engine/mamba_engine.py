"""Hybrid Mamba + Transformer and RWKV-7 inference engines (Phase 11.1, 11.2).

This module provides *reference-quality* scaffolding for state-space and
linear-recurrence model families that are not yet natively supported by
``mlx_lm`` as of MLX 0.31.0:

- **Hybrid Mamba/Transformer** (IBM Granite 4.0-H, 2025): interleaves
  selective SSM (Mamba-2) blocks with standard Transformer blocks.
- **RWKV-7** ("Goose", 2024-2025): a pure linear-recurrence architecture
  with a time-mixing state that subsumes attention.

Because MLX does not yet ship a production-grade selective-scan (SSM)
kernel, the :class:`HybridMambaEngine` and :class:`RWKV7Engine` execution
paths defer to ``transformers`` via a lazy import for actual token
generation, while still exposing the same ``load`` / ``generate`` /
``stream`` API surface as :class:`forge.engine.mlx_engine.MLXEngine`.

The small ``MambaBlock`` ``mlx.nn.Module`` below is provided as a
structural reference (what a native MLX port would look like) and can be
exercised for shape checks, but is **not** wired into the execution path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Generator

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# 11.1  Hybrid Mamba + Transformer
# ---------------------------------------------------------------------------


@dataclass
class MambaConfig:
    """Configuration for a Mamba / hybrid-Mamba model.

    Attributes
    ----------
    d_model:
        Residual-stream hidden size.
    d_state:
        SSM state dimension ``N`` (typically 16 or 128 in Mamba-2).
    d_conv:
        1-D causal convolution kernel width (typically 4).
    expand:
        FFN-style expansion factor for the inner dimension ``d_inner = expand * d_model``.
    num_layers:
        Total number of blocks in the stack.
    hybrid_pattern:
        Optional list describing which layers are Mamba (``"mamba"``) vs
        attention (``"attn"``). ``None`` means "all mamba". Granite 4.0-H
        uses a 9:1 mamba:attention ratio.
    vocab_size:
        Output vocabulary size.
    """

    d_model: int = 4096
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    num_layers: int = 32
    hybrid_pattern: list[str] | None = None
    vocab_size: int = 32000

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model


class MambaBlock(nn.Module):
    """Reference selective-SSM block (Mamba-2 style).

    This is **structural scaffolding**: it lays out the parameter shapes
    and a naive recurrence loop so the rest of the codebase can reason
    about Mamba layers. A production implementation would:

    1. Use a fused selective-scan kernel (currently requires custom Metal
       shaders; see ``src/forge/metal/kernels.metal``).
    2. Parallelize the scan via the associative-scan trick used by the
       reference CUDA implementation.
    3. Batch the discretization of ``A`` and ``B`` with the input
       projection.
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_inner

        # Input projection: x -> (x_proj, z_gate)
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=False)
        # Depthwise causal 1-D conv, implemented naively as a Linear over
        # a sliding window (a real impl would use mx.conv1d).
        self.conv1d_weight = mx.zeros((d_inner, cfg.d_conv))
        # SSM parameters: B, C are input-dependent; A is a stable diagonal
        # state transition; dt is the discretization step.
        self.x_proj = nn.Linear(d_inner, 2 * cfg.d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        # Log-parameterized ``A`` keeps the recurrence stable.
        self.A_log = mx.zeros((d_inner, cfg.d_state))
        self.D = mx.ones((d_inner,))
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def __call__(self, x: mx.array, state: mx.array | None = None) -> tuple[mx.array, mx.array]:
        """Run the selective scan.

        Parameters
        ----------
        x:
            ``(B, T, d_model)`` input.
        state:
            Optional previous SSM state ``(B, d_inner, d_state)`` for
            incremental decoding.

        Returns
        -------
        (y, new_state) where ``y`` has shape ``(B, T, d_model)``.
        """
        B, T, _ = x.shape
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_in, z = mx.split(xz, 2, axis=-1)

        # Naive reference recurrence (elementwise, for clarity).
        d_inner = self.cfg.d_inner
        d_state = self.cfg.d_state
        if state is None:
            state = mx.zeros((B, d_inner, d_state))

        A = -mx.exp(self.A_log)  # (d_inner, d_state), negative for stability
        outputs = []
        h = state
        for t in range(T):
            xt = x_in[:, t, :]  # (B, d_inner)
            # Input-dependent B, C, dt
            proj = self.x_proj(xt)  # (B, 2*d_state + 1)
            Bt = proj[:, :d_state]
            Ct = proj[:, d_state : 2 * d_state]
            dt = nn.softplus(self.dt_proj(proj[:, -1:]))  # (B, d_inner)

            # Discretize: h = exp(dt*A) * h + dt*B * x
            # Shapes: dt (B, d_inner), A (d_inner, d_state)
            dtA = dt[:, :, None] * A[None, :, :]  # (B, d_inner, d_state)
            h = mx.exp(dtA) * h + (dt[:, :, None] * Bt[:, None, :]) * xt[:, :, None]
            # y = C * h + D * x
            yt = mx.sum(h * Ct[:, None, :], axis=-1) + self.D * xt
            outputs.append(yt)

        y = mx.stack(outputs, axis=1)  # (B, T, d_inner)
        y = y * nn.silu(z)  # gating
        return self.out_proj(y), h


@dataclass
class HybridEngineConfig:
    """Runtime config for :class:`HybridMambaEngine` / :class:`RWKV7Engine`."""

    model_path: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    trust_remote_code: bool = True
    device: str = "mps"  # or "cpu"


@dataclass
class HybridGenerationResult:
    text: str = ""
    tokens_generated: int = 0
    tps: float = 0.0
    ttft_seconds: float = 0.0
    total_seconds: float = 0.0
    backend: str = "transformers"
    architecture_family: str = ""


class HybridMambaEngine:
    """Inference engine for hybrid Mamba+Transformer models (e.g. Granite 4.0-H).

    Mirrors the :class:`forge.engine.mlx_engine.MLXEngine` API so it can be
    swapped in transparently. Because ``mlx_lm`` does not yet ship native
    Mamba kernels, execution is performed through ``transformers`` with a
    deferred import. The class still exposes Mamba-specific knobs
    (``MambaConfig``) for downstream optimizers to reason about.
    """

    def __init__(
        self,
        config: HybridEngineConfig,
        mamba_config: MambaConfig | None = None,
    ) -> None:
        self.config = config
        self.mamba_config = mamba_config
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._architecture_family = "hybrid-mamba"

    def load(self) -> None:
        """Load model + tokenizer via ``transformers``.

        Detects mamba vs attention layers from the HF config and, if the
        caller did not supply a :class:`MambaConfig`, synthesizes one.
        """
        from transformers import (  # type: ignore[import-untyped]
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        hf_cfg = AutoConfig.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        raw = hf_cfg.to_dict()

        # Detect hybrid layer pattern. Granite 4.0-H exposes this via
        # ``layer_types`` or similar; fall back to "all mamba" otherwise.
        layer_types = raw.get("layer_types") or raw.get("block_types")
        if self.mamba_config is None:
            self.mamba_config = MambaConfig(
                d_model=raw.get("hidden_size", 4096),
                d_state=raw.get("mamba_d_state", raw.get("state_size", 128)),
                d_conv=raw.get("mamba_d_conv", raw.get("conv_kernel", 4)),
                expand=raw.get("mamba_expand", 2),
                num_layers=raw.get("num_hidden_layers", 32),
                hybrid_pattern=layer_types,
                vocab_size=raw.get("vocab_size", 32000),
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype="auto",
        )
        try:
            self._model = self._model.to(self.config.device)
        except Exception:  # pragma: no cover - device may be unavailable
            pass
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        import gc

        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, **kwargs: Any) -> HybridGenerationResult:
        """Generate a full completion synchronously."""
        if not self._loaded:
            self.load()

        import torch  # type: ignore[import-untyped]

        max_tokens = int(kwargs.get("max_tokens", self.config.max_tokens))
        temperature = float(kwargs.get("temperature", self.config.temperature))
        top_p = float(kwargs.get("top_p", self.config.top_p))

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        start = time.monotonic()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
            )
        elapsed = time.monotonic() - start
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        n = int(gen_ids.shape[0])
        return HybridGenerationResult(
            text=text,
            tokens_generated=n,
            tps=n / elapsed if elapsed > 0 else 0.0,
            total_seconds=elapsed,
            ttft_seconds=elapsed,  # no streaming granularity in this fallback
            backend="transformers",
            architecture_family=self._architecture_family,
        )

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, HybridGenerationResult]:
        """Token-level streaming using ``transformers.TextIteratorStreamer``."""
        if not self._loaded:
            self.load()

        from threading import Thread

        from transformers import TextIteratorStreamer  # type: ignore[import-untyped]

        max_tokens = int(kwargs.get("max_tokens", self.config.max_tokens))
        temperature = float(kwargs.get("temperature", self.config.temperature))
        top_p = float(kwargs.get("top_p", self.config.top_p))

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            streamer=streamer,
        )
        start = time.monotonic()
        first: float | None = None
        count = 0
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        for chunk in streamer:
            if first is None:
                first = time.monotonic()
            count += 1
            yield chunk
        thread.join()
        elapsed = time.monotonic() - start
        return HybridGenerationResult(
            tokens_generated=count,
            tps=count / elapsed if elapsed > 0 else 0.0,
            total_seconds=elapsed,
            ttft_seconds=(first - start) if first else 0.0,
            backend="transformers",
            architecture_family=self._architecture_family,
        )


# ---------------------------------------------------------------------------
# 11.2  RWKV-7 linear recurrence
# ---------------------------------------------------------------------------


@dataclass
class RWKV7Config:
    """Configuration for an RWKV-7 ("Goose") model."""

    d_model: int = 2048
    num_layers: int = 24
    vocab_size: int = 65536
    head_size: int = 64  # time-mix head dimension


class RWKV7State:
    """Per-layer linear-recurrence state container.

    RWKV-7 keeps (for each layer) a small fixed-size state tensor of shape
    ``(num_heads, head_size, head_size)`` — fundamentally different from a
    Transformer KV cache (which grows with sequence length). This class is
    a thin bookkeeping wrapper so the rest of forge can reason about state
    lifecycle (reset, save, restore) uniformly with the other engines.
    """

    def __init__(self, cfg: RWKV7Config):
        self.cfg = cfg
        num_heads = max(cfg.d_model // cfg.head_size, 1)
        self.layer_states: list[mx.array] = [
            mx.zeros((num_heads, cfg.head_size, cfg.head_size))
            for _ in range(cfg.num_layers)
        ]

    def reset(self) -> None:
        for i in range(len(self.layer_states)):
            self.layer_states[i] = mx.zeros_like(self.layer_states[i])


class RWKV7Engine:
    """Inference engine for RWKV-7 models.

    Like :class:`HybridMambaEngine`, this is a control-plane wrapper around
    ``transformers`` (which has an RWKV reference implementation) exposing
    the same API surface as :class:`MLXEngine`. A future native MLX port
    would replace the ``transformers`` call with a custom kernel driven
    off :class:`RWKV7State`.
    """

    def __init__(
        self,
        config: HybridEngineConfig,
        rwkv_config: RWKV7Config | None = None,
    ) -> None:
        self.config = config
        self.rwkv_config = rwkv_config
        self._model: Any = None
        self._tokenizer: Any = None
        self._state: RWKV7State | None = None
        self._loaded = False
        self._architecture_family = "rwkv7"

    def load(self) -> None:
        from transformers import (  # type: ignore[import-untyped]
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        hf_cfg = AutoConfig.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        raw = hf_cfg.to_dict()
        if self.rwkv_config is None:
            self.rwkv_config = RWKV7Config(
                d_model=raw.get("hidden_size", 2048),
                num_layers=raw.get("num_hidden_layers", 24),
                vocab_size=raw.get("vocab_size", 65536),
                head_size=raw.get("head_size", 64),
            )
        self._state = RWKV7State(self.rwkv_config)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype="auto",
        )
        try:
            self._model = self._model.to(self.config.device)
        except Exception:  # pragma: no cover
            pass
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._state = None
        self._loaded = False
        import gc

        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, **kwargs: Any) -> HybridGenerationResult:
        if not self._loaded:
            self.load()
        import torch  # type: ignore[import-untyped]

        max_tokens = int(kwargs.get("max_tokens", self.config.max_tokens))
        temperature = float(kwargs.get("temperature", self.config.temperature))
        top_p = float(kwargs.get("top_p", self.config.top_p))

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        start = time.monotonic()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
            )
        elapsed = time.monotonic() - start
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        n = int(gen_ids.shape[0])
        return HybridGenerationResult(
            text=text,
            tokens_generated=n,
            tps=n / elapsed if elapsed > 0 else 0.0,
            total_seconds=elapsed,
            ttft_seconds=elapsed,
            backend="transformers",
            architecture_family=self._architecture_family,
        )

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, HybridGenerationResult]:
        if not self._loaded:
            self.load()
        from threading import Thread

        from transformers import TextIteratorStreamer  # type: ignore[import-untyped]

        max_tokens = int(kwargs.get("max_tokens", self.config.max_tokens))
        temperature = float(kwargs.get("temperature", self.config.temperature))
        top_p = float(kwargs.get("top_p", self.config.top_p))
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            streamer=streamer,
        )
        start = time.monotonic()
        first: float | None = None
        count = 0
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        for chunk in streamer:
            if first is None:
                first = time.monotonic()
            count += 1
            yield chunk
        thread.join()
        elapsed = time.monotonic() - start
        return HybridGenerationResult(
            tokens_generated=count,
            tps=count / elapsed if elapsed > 0 else 0.0,
            total_seconds=elapsed,
            ttft_seconds=(first - start) if first else 0.0,
            backend="transformers",
            architecture_family=self._architecture_family,
        )


__all__ = [
    "MambaConfig",
    "MambaBlock",
    "HybridEngineConfig",
    "HybridGenerationResult",
    "HybridMambaEngine",
    "RWKV7Config",
    "RWKV7State",
    "RWKV7Engine",
]
