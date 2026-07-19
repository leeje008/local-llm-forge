"""Star Attention prefill for long-context Transformers (Phase 11.5).

Reference implementation of *Star Attention* (arXiv:2411.17076), which
accelerates long-context **prefill** by restricting each token's
attention to:

1. An "anchor" block of the first ``anchor_size`` tokens (shared global
   context — instructions, system prompt, retrieval header, etc.), and
2. Its own local ``block_size`` chunk.

During prefill the input is split into chunks, and each chunk attends
only to ``anchor ∪ self``. This converts the quadratic prefill cost
``O(N^2)`` into roughly ``O(N * (anchor_size + block_size))`` at a
negligible quality cost for RAG-style workloads.

This module provides a **dense reference**: it builds the attention
mask as an ``mx.array`` and lets the standard softmax kernel consume
it. A sparse / block-sparse optimization (the real speedup) is a
follow-up item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


@dataclass
class StarAttentionConfig:
    """Hyperparameters for Star Attention prefill.

    Attributes
    ----------
    block_size:
        Size of each local chunk. The paper uses values between 2k and
        16k tokens.
    anchor_size:
        Size of the global anchor block at the start of the sequence.
        The paper uses ``anchor_size = block_size`` as a solid default.
    causal:
        If True (standard decoder-only setup), disallow attention to
        future tokens in addition to the block+anchor restriction.
    """

    block_size: int = 4096
    anchor_size: int = 4096
    causal: bool = True


NEG_INF = -1e9


def star_attention_mask(
    seq_len: int,
    cfg: StarAttentionConfig,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Build a dense Star Attention mask of shape ``(seq_len, seq_len)``.

    Positions allowed to attend are 0; disallowed positions are a large
    negative number (``-1e9``) so that after softmax their probability
    collapses to zero.

    Rules
    -----
    For query index ``q`` and key index ``k``, attention is allowed iff::

        k < anchor_size                         # global anchor
        OR  block_id(k) == block_id(q)          # same local chunk
        AND (not causal OR k <= q)

    where ``block_id(i) = (i - anchor_size) // block_size`` for
    ``i >= anchor_size``, and ``block_id(i) = -1`` (anchor) otherwise.
    """
    anchor = cfg.anchor_size
    block = cfg.block_size

    idx = mx.arange(seq_len)
    q = idx[:, None]  # (N, 1)
    k = idx[None, :]  # (1, N)

    is_anchor_key = k < anchor
    block_id_q = mx.where(q < anchor, mx.full(q.shape, -1), (q - anchor) // block)
    block_id_k = mx.where(k < anchor, mx.full(k.shape, -1), (k - anchor) // block)
    same_block = (block_id_q == block_id_k) & (q >= anchor) & (k >= anchor)
    allowed = is_anchor_key | same_block
    if cfg.causal:
        allowed = allowed & (k <= q)

    mask = mx.where(
        allowed,
        mx.zeros((seq_len, seq_len), dtype=dtype),
        mx.full((seq_len, seq_len), NEG_INF, dtype=dtype),
    )
    return mask


def _softmax_attention(
    q: mx.array, k: mx.array, v: mx.array, mask: mx.array
) -> mx.array:
    """Plain scaled-dot-product attention with additive mask.

    Shapes: ``q,k,v`` are ``(B, H, T, D)``; ``mask`` broadcasts to
    ``(T, T)`` or ``(B, H, T, T)``.
    """
    scale = float(q.shape[-1]) ** -0.5
    scores = (q @ k.swapaxes(-1, -2)) * scale + mask
    probs = mx.softmax(scores, axis=-1)
    return probs @ v


def chunked_prefill_with_star(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cfg: StarAttentionConfig,
    attention_fn: Callable[[mx.array, mx.array, mx.array, mx.array], mx.array] | None = None,
) -> mx.array:
    """Run a Star Attention prefill.

    Parameters
    ----------
    q, k, v:
        Query / key / value tensors shaped ``(B, H, T, D)``.
    cfg:
        :class:`StarAttentionConfig`.
    attention_fn:
        Optional override for the inner attention kernel. Defaults to a
        plain softmax attention; callers that have a flash-attention
        kernel available can pass it here.

    Returns
    -------
    Output tensor of shape ``(B, H, T, D)``.

    Notes
    -----
    This is a **dense reference**: we build the full ``(T, T)`` mask and
    let softmax zero out the forbidden positions. The win comes from a
    future block-sparse rewrite that only materializes the anchor and
    local blocks.
    """
    T = q.shape[-2]
    attention_fn = attention_fn or _softmax_attention

    if T <= cfg.anchor_size + cfg.block_size:
        # Short sequence — no benefit from chunking, use a single call.
        mask = star_attention_mask(T, cfg, dtype=q.dtype)
        return attention_fn(q, k, v, mask)

    # Chunked path: for each block, assemble the anchor + local window
    # for both keys and queries, compute attention, and stitch the
    # outputs back in place.
    anchor = cfg.anchor_size
    block = cfg.block_size
    out = mx.zeros_like(q)

    # Anchor region uses its own causal attention (attends to itself only).
    if anchor > 0:
        anchor_mask = star_attention_mask(anchor, cfg, dtype=q.dtype)
        out_anchor = attention_fn(
            q[..., :anchor, :],
            k[..., :anchor, :],
            v[..., :anchor, :],
            anchor_mask,
        )
        out = _scatter_slice(out, out_anchor, anchor_start=0)

    # Iterate local chunks.
    cursor = anchor
    while cursor < T:
        end = min(cursor + block, T)
        # Keys/values visible to this chunk = anchor + own chunk.
        kv_k = mx.concatenate([k[..., :anchor, :], k[..., cursor:end, :]], axis=-2)
        kv_v = mx.concatenate([v[..., :anchor, :], v[..., cursor:end, :]], axis=-2)
        q_chunk = q[..., cursor:end, :]

        chunk_len = end - cursor
        kv_len = anchor + chunk_len
        # Build a causal mask of shape (chunk_len, kv_len) where the
        # first ``anchor`` columns are always visible and the remaining
        # ``chunk_len`` columns are a lower-triangular mask.
        cols = mx.arange(kv_len)
        rows = mx.arange(chunk_len)
        anchor_col = cols < anchor
        # Local columns: visible iff (col_idx - anchor) <= row.
        local_col_idx = cols - anchor
        local_visible = (local_col_idx >= 0) & (local_col_idx <= rows[:, None])
        allowed = anchor_col[None, :] | local_visible
        if not cfg.causal:
            # Still respect anchor gating but let the chunk see its whole self.
            allowed = anchor_col[None, :] | (local_col_idx[None, :] >= 0)
        mask = mx.where(
            allowed,
            mx.zeros((chunk_len, kv_len), dtype=q.dtype),
            mx.full((chunk_len, kv_len), NEG_INF, dtype=q.dtype),
        )
        out_chunk = attention_fn(q_chunk, kv_k, kv_v, mask)
        out = _scatter_slice(out, out_chunk, anchor_start=cursor)
        cursor = end

    return out


def _scatter_slice(dest: mx.array, src: mx.array, anchor_start: int) -> mx.array:
    """Return a copy of ``dest`` with ``src`` written at
    ``[..., anchor_start:anchor_start+src.shape[-2], :]``.

    MLX arrays are immutable so we rebuild via concatenation. This keeps
    the reference implementation simple; a production rewrite would
    allocate the output buffer once and use scatter updates.
    """
    length = src.shape[-2]
    end = anchor_start + length
    before = dest[..., :anchor_start, :]
    after = dest[..., end:, :]
    return mx.concatenate([before, src, after], axis=-2)


__all__ = [
    "StarAttentionConfig",
    "star_attention_mask",
    "chunked_prefill_with_star",
]
