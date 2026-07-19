"""BitNet / ternary weight quantization engine (Phase 11.4).

BitNet b1.58 (*The Era of 1-bit LLMs*, 2402.17764) represents every
weight with a ternary value in ``{-1, 0, +1}`` plus a per-tensor (or
per-channel) scalar, yielding an effective bit-width of
``log2(3) ≈ 1.58`` bits.

Actual BitNet *inference* (where the memory savings pay off) requires
specialized matmul kernels that fuse ternary unpacking with the multiply
— Microsoft ships these in the ``bitnet.cpp`` project. This module
therefore provides two things:

1. **An MLX-native ``ternary_quantize`` routine** that can be used
   off-line to compress weights into ``{-1, 0, +1} * scale`` form,
   producing a packed artifact on disk.
2. **A ``BitNetEngine`` control-plane** that exposes the same API as
   :class:`forge.engine.mlx_engine.MLXEngine` and, at ``generate`` time,
   invokes a locally installed ``bitnet.cpp`` server binary via
   ``subprocess``. If the binary is not found it raises a clear
   ``RuntimeError`` explaining the dependency.

Pure-MLX ternary matmul kernels are a potential future enhancement (they
would live in ``src/forge/metal/kernels.metal``).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import mlx.core as mx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BitNetConfig:
    """BitNet runtime and quantization config.

    Attributes
    ----------
    model_path:
        Path to a BitNet-quantized artifact (e.g. a ``bitnet.cpp`` GGUF).
    ternary_mode:
        If True, :func:`ternary_quantize` will snap values to
        ``{-1, 0, +1}``. Disabling falls back to pure scalar scaling,
        which is only useful for debugging.
    bits:
        Effective bit-width. 1.58 is the canonical BitNet b1.58 value.
    bitnet_cpp_binary:
        Path to a ``llama-cli``-compatible binary from the
        ``bitnet.cpp`` project. Defaults to searching ``$PATH``.
    max_tokens:
        Default max new tokens.
    temperature:
        Default sampling temperature.
    top_p:
        Default top-p.
    quant_threshold:
        Values below ``quant_threshold * mean_abs_weight`` are mapped to
        0 during ternary quantization. Defaults to 0.7 (empirically
        recommended by the BitNet authors).
    per_channel:
        If True, compute one scale per output channel; otherwise one
        scale per tensor.
    """

    model_path: str = ""
    ternary_mode: bool = True
    bits: float = 1.58
    bitnet_cpp_binary: str | None = None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    quant_threshold: float = 0.7
    per_channel: bool = True


@dataclass
class TernaryTensor:
    """Packed ternary weight tensor.

    ``codes`` holds values in ``{-1, 0, +1}`` (stored as int8 for
    simplicity; a production packer would bit-pack five ternary values
    into each byte to hit ~1.58 bits/weight). ``scale`` is broadcast-
    compatible with ``codes``.
    """

    codes: mx.array
    scale: mx.array
    shape: tuple[int, ...]
    dtype: str = "int8"

    def dequantize(self) -> mx.array:
        """Reconstruct the full-precision tensor (lossy)."""
        return self.codes.astype(mx.float32) * self.scale


# ---------------------------------------------------------------------------
# Ternary quantization
# ---------------------------------------------------------------------------


def ternary_quantize(
    weights: mx.array,
    cfg: BitNetConfig | None = None,
) -> TernaryTensor:
    """Quantize a weight tensor to ternary ``{-1, 0, +1} * scale``.

    Follows the BitNet b1.58 recipe: compute ``gamma = mean(|W|)`` (per
    channel or per tensor), pick a threshold ``delta = quant_threshold *
    gamma``, snap values below ``delta`` in magnitude to 0, and sign
    everything else.

    Parameters
    ----------
    weights:
        A 2-D linear layer weight ``(out, in)`` typically, but any shape
        is accepted.
    cfg:
        BitNet config. If None, defaults are used.

    Returns
    -------
    :class:`TernaryTensor` wrapping the codes and per-channel scale.
    """
    cfg = cfg or BitNetConfig()
    w = weights.astype(mx.float32)

    if cfg.per_channel and w.ndim >= 2:
        # Per-output-channel scale along axis 0.
        abs_w = mx.abs(w)
        gamma = mx.mean(abs_w, axis=tuple(range(1, w.ndim)), keepdims=True)
    else:
        gamma = mx.mean(mx.abs(w))

    if not cfg.ternary_mode:
        # Pure scalar scaling fallback.
        codes = mx.sign(w)
        return TernaryTensor(codes=codes, scale=gamma, shape=tuple(w.shape))

    delta = cfg.quant_threshold * gamma
    # Zero-out small weights, sign the rest.
    zero_mask = mx.abs(w) < delta
    signed = mx.sign(w)
    codes = mx.where(zero_mask, mx.zeros_like(signed), signed)
    # Cast codes to int8 for compact storage.
    codes_i8 = codes.astype(mx.int8)
    return TernaryTensor(
        codes=codes_i8,
        scale=gamma.astype(mx.float32),
        shape=tuple(w.shape),
    )


def ternary_quantize_module(
    state_dict: dict[str, mx.array], cfg: BitNetConfig | None = None
) -> dict[str, TernaryTensor | mx.array]:
    """Quantize every 2-D weight in a state dict; leave embeddings/norms alone.

    This mirrors the BitNet recipe where only linear projections inside
    transformer blocks are ternarized; embeddings, layernorms, and the
    LM head are typically kept at higher precision.
    """
    out: dict[str, TernaryTensor | mx.array] = {}
    for name, tensor in state_dict.items():
        skip = any(
            k in name.lower()
            for k in ("embed", "norm", "layernorm", "rmsnorm", "lm_head", "bias")
        )
        if skip or tensor.ndim < 2:
            out[name] = tensor
        else:
            out[name] = ternary_quantize(tensor, cfg)
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class BitNetGenerationResult:
    text: str = ""
    tokens_generated: int = 0
    tps: float = 0.0
    total_seconds: float = 0.0
    backend: str = "bitnet.cpp"
    raw_stdout: str = ""


class BitNetEngine:
    """Control-plane wrapper around ``bitnet.cpp``.

    The engine does not run inference itself — the compute kernels live
    in Microsoft's ``bitnet.cpp`` repository. This class simply:

    1. Locates the ``bitnet.cpp`` CLI binary (``llama-cli`` built from
       that repo, or whatever path is supplied in :class:`BitNetConfig`).
    2. Exposes ``load``/``generate``/``stream``/``unload`` with the same
       signature as :class:`MLXEngine`, so the higher-level forge CLI
       can swap it in.
    3. Falls back to a clear error explaining what to install if the
       binary is missing.
    """

    def __init__(self, config: BitNetConfig):
        self.config = config
        self._binary: str | None = None
        self._loaded = False

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        bin_path = self.config.bitnet_cpp_binary or shutil.which("llama-cli")
        if bin_path is None or not Path(bin_path).exists():
            raise RuntimeError(
                "BitNetEngine requires bitnet.cpp's llama-cli binary. "
                "Build it from https://github.com/microsoft/BitNet and either "
                "put it on $PATH or set BitNetConfig.bitnet_cpp_binary."
            )
        if not self.config.model_path or not Path(self.config.model_path).exists():
            raise FileNotFoundError(
                f"BitNet model artifact not found: {self.config.model_path!r}"
            )
        self._binary = bin_path
        self._loaded = True

    def unload(self) -> None:
        self._binary = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # -- generation -------------------------------------------------------

    def _build_cmd(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> list[str]:
        assert self._binary is not None
        return [
            self._binary,
            "-m",
            self.config.model_path,
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "--temp",
            str(temperature),
            "--top-p",
            str(top_p),
            "-no-cnv",
        ]

    def generate(self, prompt: str, **kwargs: Any) -> BitNetGenerationResult:
        """Invoke ``bitnet.cpp`` synchronously.

        Note: for a real deployment you would keep a persistent server
        process (``llama-server``) alive rather than re-spawning per
        request; that optimization is left to the deployer layer.
        """
        if not self._loaded:
            self.load()

        max_tokens = int(kwargs.get("max_tokens", self.config.max_tokens))
        temperature = float(kwargs.get("temperature", self.config.temperature))
        top_p = float(kwargs.get("top_p", self.config.top_p))

        cmd = self._build_cmd(prompt, max_tokens, temperature, top_p)
        start = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.monotonic() - start
        if proc.returncode != 0:
            raise RuntimeError(
                f"bitnet.cpp failed (exit {proc.returncode}): {proc.stderr[:500]}"
            )
        text = proc.stdout
        # bitnet.cpp echoes the prompt; strip it when present.
        if text.startswith(prompt):
            text = text[len(prompt):]
        # Rough token count (whitespace-split) — the deployer can wire a
        # real tokenizer in later.
        n = max(len(text.split()), 1)
        return BitNetGenerationResult(
            text=text,
            tokens_generated=n,
            tps=n / elapsed if elapsed > 0 else 0.0,
            total_seconds=elapsed,
            raw_stdout=proc.stdout,
        )

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, BitNetGenerationResult]:
        """Stream output from ``bitnet.cpp`` line-by-line."""
        if not self._loaded:
            self.load()

        max_tokens = int(kwargs.get("max_tokens", self.config.max_tokens))
        temperature = float(kwargs.get("temperature", self.config.temperature))
        top_p = float(kwargs.get("top_p", self.config.top_p))

        cmd = self._build_cmd(prompt, max_tokens, temperature, top_p)
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        chunks: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            yield line
        proc.wait()
        elapsed = time.monotonic() - start
        text = "".join(chunks)
        n = max(len(text.split()), 1)
        return BitNetGenerationResult(
            text=text,
            tokens_generated=n,
            tps=n / elapsed if elapsed > 0 else 0.0,
            total_seconds=elapsed,
            raw_stdout=text,
        )


__all__ = [
    "BitNetConfig",
    "TernaryTensor",
    "ternary_quantize",
    "ternary_quantize_module",
    "BitNetEngine",
    "BitNetGenerationResult",
]
