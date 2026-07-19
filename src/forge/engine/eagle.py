"""EAGLE-3: Target-attached speculative decoding head (Phase 7.5).

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) attaches
a lightweight auto-regressive head to the target model that predicts the next
token from hidden states of the penultimate layer. EAGLE-3 (NeurIPS 2025,
arXiv 2503.01840) removes the feature-prediction loss and adds training-time
simulation, pushing speculative speed-ups to ~6.5x on common benchmarks.

This module provides:
  * EagleHeadConfig — metadata + hyperparameters of a trained head
  * EagleHead (mlx.nn.Module) — single-layer transformer block + LM head
  * load_eagle_head — load head weights from a local path or HF repo
  * generate_eagle_draft — produce K draft tokens from hidden states
  * EAGLE_HEAD_REGISTRY — known pre-trained heads for popular targets

NOTE: Pre-trained EAGLE-3 heads for MLX are not yet broadly published. This
module is designed so that when a head becomes available (either from the
official EAGLE repo ported to MLX, or re-trained locally with the provided
training stub), it plugs directly into MLXEngine via EngineConfig.eagle_head_path.
Until a head is loaded, `select_eagle_head()` returns None and callers should
fall back to N-gram / draft-model speculative decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Registry of known EAGLE-3 heads (populated as upstream MLX ports appear)
# ---------------------------------------------------------------------------

EAGLE_HEAD_REGISTRY: dict[str, dict[str, Any]] = {
    # Format: target_model_substring -> head metadata
    # "llama-3-8b": {
    #     "repo": "yuhuili/EAGLE3-LLaMA3-Instruct-8B",
    #     "hidden_size": 4096,
    #     "vocab_size": 128256,
    # },
    # "qwen2.5-7b": {...},
}


@dataclass
class EagleHeadConfig:
    """Hyperparameters of a trained EAGLE-3 head."""

    hidden_size: int
    vocab_size: int
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-5
    # EAGLE-3 specifics
    num_layers: int = 1  # Single transformer block is canonical
    use_feature_loss: bool = False  # EAGLE-3 drops this
    tie_lm_head: bool = True  # Reuse target model's lm_head when possible
    # Draft control
    default_k: int = 5  # Number of draft tokens to generate per step
    max_k: int = 8


@dataclass
class EagleHeadInfo:
    """Metadata returned when an EAGLE head is selected for a target model."""

    target_model: str
    head_path: str | None  # Local path or HF repo id
    config: EagleHeadConfig
    source: str  # "registry", "local", or "custom"
    available: bool = False  # True iff weights are actually downloadable/loadable


def select_eagle_head(
    target_model_id: str,
    local_heads_dir: Path | None = None,
) -> EagleHeadInfo | None:
    """Find a compatible EAGLE-3 head for the given target model.

    Resolution order:
      1. Explicit local directory (local_heads_dir/<target_name>/)
      2. EAGLE_HEAD_REGISTRY entry (matched by substring)
      3. None (caller should fall back to alternate speculative method)
    """
    target_lower = target_model_id.lower()

    # 1. Local directory
    if local_heads_dir is not None:
        safe = target_model_id.replace("/", "--")
        candidate = local_heads_dir / safe
        if candidate.exists() and (candidate / "config.json").exists():
            cfg = _load_head_config(candidate)
            return EagleHeadInfo(
                target_model=target_model_id,
                head_path=str(candidate),
                config=cfg,
                source="local",
                available=True,
            )

    # 2. Registry lookup
    for key, meta in EAGLE_HEAD_REGISTRY.items():
        if key in target_lower:
            cfg = EagleHeadConfig(
                hidden_size=meta["hidden_size"],
                vocab_size=meta["vocab_size"],
                num_attention_heads=meta.get("num_attention_heads", 32),
                intermediate_size=meta.get("intermediate_size", 11008),
            )
            return EagleHeadInfo(
                target_model=target_model_id,
                head_path=meta["repo"],
                config=cfg,
                source="registry",
                available=True,
            )

    return None


def _load_head_config(path: Path) -> EagleHeadConfig:
    """Load EagleHeadConfig from a config.json in a local head directory."""
    import json

    data = json.loads((path / "config.json").read_text())
    return EagleHeadConfig(
        hidden_size=data["hidden_size"],
        vocab_size=data["vocab_size"],
        num_attention_heads=data.get("num_attention_heads", 32),
        intermediate_size=data.get("intermediate_size", 11008),
        rms_norm_eps=data.get("rms_norm_eps", 1e-5),
        num_layers=data.get("num_layers", 1),
        default_k=data.get("default_k", 5),
        max_k=data.get("max_k", 8),
    )


# ---------------------------------------------------------------------------
# EAGLE-3 head module (MLX). Imports are deferred so that the rest of the
# CLI works on machines without mlx installed (e.g. during `forge analyze`).
# ---------------------------------------------------------------------------


def build_eagle_head(config: EagleHeadConfig):
    """Instantiate an mlx.nn.Module implementing a single EAGLE-3 decoder block.

    Returns an nn.Module with .forward(hidden_states, past_token_ids) -> logits.
    This is a reference implementation; actual weights must be loaded via
    `load_eagle_head` before use.
    """
    import mlx.core as mx  # noqa: F401  (required at call time)
    import mlx.nn as nn

    class EagleBlock(nn.Module):
        def __init__(self, cfg: EagleHeadConfig):
            super().__init__()
            self.cfg = cfg
            self.input_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.self_attn = nn.MultiHeadAttention(
                dims=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
                bias=False,
            )
            self.post_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.mlp_gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.mlp_up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.mlp_down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        def __call__(self, hidden_states):
            import mlx.nn.functional as F

            h = self.input_norm(hidden_states)
            h = self.self_attn(h, h, h) + hidden_states
            h2 = self.post_norm(h)
            gated = F.silu(self.mlp_gate(h2)) * self.mlp_up(h2)
            h = self.mlp_down(gated) + h
            return self.lm_head(h)

    return EagleBlock(config)


def load_eagle_head(info: EagleHeadInfo):
    """Load an EAGLE-3 head's weights and return an instantiated module.

    Returns (module, tokenizer_hint) or (None, reason) on failure.
    """
    if not info.available or info.head_path is None:
        return None, "EAGLE head not available"

    try:
        import mlx.core as mx

        head = build_eagle_head(info.config)
        weights_file = Path(info.head_path) / "weights.safetensors"
        if not weights_file.exists():
            return None, f"weights.safetensors not found at {info.head_path}"

        weights = mx.load(str(weights_file))
        head.load_weights(list(weights.items()))
        return head, None
    except Exception as e:  # pragma: no cover - environment dependent
        return None, f"load failed: {e}"


def generate_eagle_draft(
    head,
    hidden_states,
    k: int = 5,
    temperature: float = 0.0,
) -> list[int]:
    """Generate K draft tokens from the last hidden state of the target model.

    EAGLE auto-regressively rolls the head forward K times, feeding its own
    embedding back in. Returns a list of draft token ids.
    """
    import mlx.core as mx

    drafts: list[int] = []
    h = hidden_states
    for _ in range(k):
        logits = head(h)
        # Greedy (or low-temperature) sampling
        if temperature <= 0:
            next_id = int(mx.argmax(logits[..., -1, :], axis=-1).item())
        else:
            probs = mx.softmax(logits[..., -1, :] / temperature, axis=-1)
            next_id = int(mx.random.categorical(mx.log(probs)).item())
        drafts.append(next_id)
        # Feed the predicted token embedding back into the head. A production
        # implementation would reuse the head's embedding table; as a
        # placeholder we re-use the last hidden state (low fidelity but
        # structurally correct for the API).
        h = hidden_states  # NOTE: replace with head.embed(next_id) once trained
    return drafts


def format_eagle_info(info: EagleHeadInfo | None) -> str:
    """Human-readable EAGLE head status line."""
    if info is None:
        return "EAGLE-3 head: not available for this target (fallback to N-gram/draft)"
    status = "ready" if info.available else "not loadable"
    return (
        f"EAGLE-3 head: {info.head_path} ({info.source}, {status}) "
        f"k={info.config.default_k}, vocab={info.config.vocab_size}"
    )
