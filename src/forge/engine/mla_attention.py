"""Multi-head Latent Attention (MLA) — DeepSeek-V2 / V3 (Phase 11.3).

Reference implementation of the compressed KV cache described in
*DeepSeek-V2* (arXiv:2405.04434) and reused in DeepSeek-V3. MLA replaces
the per-head ``(K, V)`` pair with two smaller tensors:

1. ``compressed_kv``: a shared low-rank projection of shape
   ``(B, T, d_kv_lora)``, recovered at attention time via the absorbed
   up-projection ``W^{UK}``/``W^{UV}``.
2. ``k_rope``: a small rotary-embedded component of shape
   ``(B, T, rope_head_dim)`` that preserves position information that
   low-rank compression cannot faithfully represent.

For a DeepSeek-V2-style config with ``num_heads=128``, ``head_dim=128``
this reduces KV cache memory by roughly::

    2 * num_heads * head_dim    2 * 128 * 128     32768
    ------------------------- = ------------- =  -------  ≈ 60x
    d_kv_lora + rope_head_dim     512 + 64         576

which matches the ~60x figure quoted in the paper.

This module provides:

- :class:`MLAConfig` — structural metadata.
- :class:`MLACompressedKVCache` — the actual reference container.
- :func:`decompress_kv` — expand the compressed cache into full ``K``/``V``
  tensors for an attention step, given the absorbed up-projection
  matrices.

An integration point for :class:`forge.engine.mlx_engine.MLXEngine` to
use this cache when serving DeepSeek-V2/V3 models is documented at the
bottom of the file; wiring is intentionally deferred so that this module
can stand alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MLAConfig:
    """Structural parameters of Multi-head Latent Attention.

    Attributes
    ----------
    num_heads:
        Number of attention heads.
    head_dim:
        Per-head dimension for the *non*-rope component
        (``W^{UK}`` / ``W^{UV}`` output).
    d_kv_lora:
        Rank of the shared KV compression (``d_c`` in the paper). Typical
        DeepSeek-V2: 512.
    d_q_lora:
        Rank of the query compression. Typical DeepSeek-V2: 1536.
    rope_head_dim:
        Per-head dimension of the small rope-carrying component. Typical
        DeepSeek-V2: 64.
    softmax_scale:
        Attention softmax scale. If 0, computed as
        ``1 / sqrt(head_dim + rope_head_dim)``.
    """

    num_heads: int = 128
    head_dim: int = 128
    d_kv_lora: int = 512
    d_q_lora: int = 1536
    rope_head_dim: int = 64
    softmax_scale: float = 0.0

    @property
    def qk_head_dim(self) -> int:
        return self.head_dim + self.rope_head_dim

    @property
    def effective_scale(self) -> float:
        if self.softmax_scale > 0:
            return self.softmax_scale
        return float(self.qk_head_dim) ** -0.5

    def bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Cache bytes per token (for a single layer)."""
        return dtype_bytes * (self.d_kv_lora + self.rope_head_dim)

    def compression_ratio(self) -> float:
        """Ratio vs storing full ``K`` and ``V`` in uncompressed form."""
        uncompressed = 2 * self.num_heads * self.head_dim
        compressed = self.d_kv_lora + self.rope_head_dim
        return uncompressed / compressed


# ---------------------------------------------------------------------------
# Compressed KV container
# ---------------------------------------------------------------------------


class MLACompressedKVCache:
    """Growing compressed KV cache for MLA.

    Stores only the low-rank ``compressed_kv`` and rotary-carrying
    ``k_rope`` tensors. Full ``K`` and ``V`` are reconstructed on the fly
    via :func:`decompress_kv` using the absorbed up-projections.

    The cache grows along the time axis (axis 1) as tokens are appended.
    """

    def __init__(self, cfg: MLAConfig, max_length: int | None = None):
        self.cfg = cfg
        self.max_length = max_length
        self._compressed_kv: mx.array | None = None  # (B, T, d_kv_lora)
        self._k_rope: mx.array | None = None  # (B, T, rope_head_dim)
        self._length = 0

    # -- state inspection -------------------------------------------------

    @property
    def length(self) -> int:
        return self._length

    @property
    def compressed_kv(self) -> mx.array | None:
        return self._compressed_kv

    @property
    def k_rope(self) -> mx.array | None:
        return self._k_rope

    def memory_bytes(self, dtype_bytes: int = 2) -> int:
        return self._length * self.cfg.bytes_per_token(dtype_bytes)

    # -- mutation ---------------------------------------------------------

    def append(self, new_compressed_kv: mx.array, new_k_rope: mx.array) -> None:
        """Append newly computed tokens to the cache.

        Parameters
        ----------
        new_compressed_kv:
            ``(B, T_new, d_kv_lora)``.
        new_k_rope:
            ``(B, T_new, rope_head_dim)``.
        """
        if self._compressed_kv is None:
            self._compressed_kv = new_compressed_kv
            self._k_rope = new_k_rope
        else:
            self._compressed_kv = mx.concatenate(
                [self._compressed_kv, new_compressed_kv], axis=1
            )
            self._k_rope = mx.concatenate([self._k_rope, new_k_rope], axis=1)
        self._length = int(self._compressed_kv.shape[1])
        if self.max_length is not None and self._length > self.max_length:
            # Sliding-window eviction of oldest tokens.
            drop = self._length - self.max_length
            self._compressed_kv = self._compressed_kv[:, drop:, :]
            self._k_rope = self._k_rope[:, drop:, :]
            self._length = self.max_length

    def reset(self) -> None:
        self._compressed_kv = None
        self._k_rope = None
        self._length = 0


# ---------------------------------------------------------------------------
# Decompression / attention helpers
# ---------------------------------------------------------------------------


def decompress_kv(
    cache: MLACompressedKVCache,
    W_UK: mx.array,
    W_UV: mx.array,
) -> tuple[mx.array, mx.array]:
    """Reconstruct per-head ``K`` and ``V`` from a compressed cache.

    Parameters
    ----------
    cache:
        Populated :class:`MLACompressedKVCache`.
    W_UK:
        Up-projection for keys, shape ``(d_kv_lora, num_heads * head_dim)``.
    W_UV:
        Up-projection for values, shape ``(d_kv_lora, num_heads * head_dim)``.

    Returns
    -------
    (K, V) each with shape ``(B, T, num_heads, qk_head_dim_or_head_dim)``.
    ``K`` concatenates the decompressed key with the broadcast
    ``k_rope`` component along the last axis, matching the DeepSeek-V2
    attention layout.
    """
    if cache.compressed_kv is None or cache.k_rope is None:
        raise ValueError("MLACompressedKVCache is empty")

    cfg = cache.cfg
    ckv = cache.compressed_kv  # (B, T, d_kv_lora)
    k_rope = cache.k_rope  # (B, T, rope_head_dim)
    B, T, _ = ckv.shape

    k_nope = ckv @ W_UK  # (B, T, num_heads*head_dim)
    v = ckv @ W_UV
    k_nope = k_nope.reshape(B, T, cfg.num_heads, cfg.head_dim)
    v = v.reshape(B, T, cfg.num_heads, cfg.head_dim)
    # Broadcast k_rope across heads (shared rope component).
    k_rope_b = mx.broadcast_to(
        k_rope[:, :, None, :], (B, T, cfg.num_heads, cfg.rope_head_dim)
    )
    k = mx.concatenate([k_nope, k_rope_b], axis=-1)
    return k, v


def mla_attention(
    q: mx.array,
    cache: MLACompressedKVCache,
    W_UK: mx.array,
    W_UV: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """Reference dense MLA attention step.

    Parameters
    ----------
    q:
        Query tensor ``(B, T_q, num_heads, qk_head_dim)``.
    cache:
        Populated :class:`MLACompressedKVCache`.
    W_UK, W_UV:
        Absorbed up-projection matrices.
    mask:
        Optional additive mask broadcastable to
        ``(B, num_heads, T_q, T_kv)``.
    """
    cfg = cache.cfg
    k, v = decompress_kv(cache, W_UK, W_UV)
    # (B, T_q, H, D) @ (B, T_kv, H, D) -> (B, H, T_q, T_kv)
    q_h = q.transpose(0, 2, 1, 3)
    k_h = k.transpose(0, 2, 1, 3)
    v_h = v.transpose(0, 2, 1, 3)
    scores = (q_h @ k_h.swapaxes(-1, -2)) * cfg.effective_scale
    if mask is not None:
        scores = scores + mask
    probs = mx.softmax(scores, axis=-1)
    out = probs @ v_h  # (B, H, T_q, head_dim)
    return out.transpose(0, 2, 1, 3)  # (B, T_q, H, head_dim)


# ---------------------------------------------------------------------------
# Integration note
# ---------------------------------------------------------------------------
# :class:`forge.engine.mlx_engine.MLXEngine` can opt in to MLA caches by
# inspecting ``ModelProfile.architecture`` / ``architecture_family``. When
# the value indicates DeepSeek-V2 or V3, the engine should:
#
#   1. Allocate one :class:`MLACompressedKVCache` per layer at load time,
#      sized from the model config (``d_kv_lora`` = ``kv_lora_rank``,
#      ``rope_head_dim`` = ``qk_rope_head_dim``).
#   2. During prefill, run the model's own MLA forward to obtain
#      ``compressed_kv`` + ``k_rope`` per layer and feed them via
#      :meth:`MLACompressedKVCache.append`.
#   3. During decode, decompress on demand with :func:`decompress_kv` (or
#      the fused :func:`mla_attention`).
#
# Full wiring is left to a follow-up that also hooks H2O/Ada-KV eviction
# through the compressed representation, which requires a cache-manager
# adapter.


__all__ = [
    "MLAConfig",
    "MLACompressedKVCache",
    "decompress_kv",
    "mla_attention",
]
