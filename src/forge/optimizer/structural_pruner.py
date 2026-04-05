"""Structural pruning pipeline (Phase 11.6).

Implements two complementary width/depth reductions:

1. **Block pruning** (*ShortGPT*, arXiv:2403.03853): rank transformer
   blocks by how much they change the residual stream and drop the
   least important ones. The signal used here is the cosine distance
   between a block's input and output hidden states on a calibration
   set — blocks whose output is almost identical to their input are
   near-identity and can be removed with minimal quality loss.

2. **Activation-aware SVD** (*ASVD*, arXiv:2312.05821): replace a
   linear layer ``W`` with a low-rank product ``U @ V`` chosen so that
   the reconstruction error is small on a calibration activation
   distribution. The rank per layer is picked adaptively from a target
   error budget.

The output of both passes is a standard HuggingFace model directory
(``config.json`` + safetensors shards + tokenizer files) so downstream
forge stages (``convert_to_mlx`` etc.) can consume it unchanged.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Block-level pruning (ShortGPT)
# ---------------------------------------------------------------------------


@dataclass
class BlockPrunerConfig:
    """Configuration for :func:`prune_blocks`.

    Attributes
    ----------
    target_layers:
        Target number of layers after pruning. Mutually exclusive with
        ``prune_ratio``.
    prune_ratio:
        Fraction of layers to remove (e.g. 0.2 = drop the 20% least
        important layers).
    calibration_samples:
        Number of text samples to feed through the model to measure
        block importance.
    max_length:
        Truncation length for calibration inputs.
    metric:
        ``"cosine"`` (ShortGPT default) or ``"l2"``. Cosine measures
        directional change in the residual stream; L2 captures magnitude.
    preserve_first:
        Number of early layers always kept (embedding-adjacent layers
        tend to be critical).
    preserve_last:
        Number of late layers always kept (LM-head-adjacent layers).
    """

    target_layers: int | None = None
    prune_ratio: float | None = 0.2
    calibration_samples: int = 32
    max_length: int = 512
    metric: str = "cosine"
    preserve_first: int = 2
    preserve_last: int = 2


@dataclass
class BlockImportance:
    layer_index: int
    score: float  # higher = more important (bigger residual change)


def _block_importance_scores(
    hidden_in: np.ndarray, hidden_out: np.ndarray, metric: str
) -> float:
    """Aggregate a single block's importance over a calibration batch.

    ``hidden_in``/``hidden_out`` are ``(B, T, D)`` float arrays captured
    from the residual stream before and after the block.
    """
    a = hidden_in.reshape(-1, hidden_in.shape[-1]).astype(np.float32)
    b = hidden_out.reshape(-1, hidden_out.shape[-1]).astype(np.float32)
    if metric == "l2":
        return float(np.mean(np.linalg.norm(a - b, axis=-1)))
    # Cosine distance (ShortGPT): 1 - cos_sim
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    cos = np.sum(a_n * b_n, axis=-1)
    return float(np.mean(1.0 - cos))


def _collect_residual_streams(
    model: Any,
    tokenizer: Any,
    calibration_texts: list[str],
    max_length: int,
) -> list[np.ndarray]:
    """Run the model with ``output_hidden_states=True`` and return a list of
    per-layer hidden-state tensors, each ``(B, T, D)``.

    The returned list has length ``num_layers + 1``: entry 0 is the
    embedding output, entry ``i`` is the output of layer ``i-1``.
    """
    import torch  # type: ignore[import-untyped]

    all_states: list[np.ndarray] | None = None
    model.eval()
    with torch.no_grad():
        for text in calibration_texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            out = model(**inputs, output_hidden_states=True)
            states = [h.detach().cpu().float().numpy() for h in out.hidden_states]
            if all_states is None:
                all_states = [[s] for s in states]  # type: ignore[list-item]
            else:
                for i, s in enumerate(states):
                    all_states[i].append(s)  # type: ignore[index]
    assert all_states is not None
    # Concat along batch axis (pad-to-max handled implicitly by truncation).
    merged: list[np.ndarray] = []
    for per_layer in all_states:
        try:
            merged.append(np.concatenate(per_layer, axis=0))  # type: ignore[arg-type]
        except ValueError:
            # Different seq lengths — fall back to stacking flattened.
            flat = [p.reshape(-1, p.shape[-1]) for p in per_layer]  # type: ignore[union-attr]
            merged.append(np.concatenate(flat, axis=0)[None, ...])
    return merged


def prune_blocks(
    model_path: str | Path,
    output_path: str | Path,
    cfg: BlockPrunerConfig | None = None,
    calibration_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Prune whole transformer blocks using ShortGPT scoring.

    The model is loaded via ``transformers``, run forward on a small
    calibration corpus to capture residual streams, each block is scored
    by how much it perturbs those residuals, and the lowest-scoring
    blocks are removed (respecting ``preserve_first``/``preserve_last``).
    The surviving blocks are renumbered, the config's
    ``num_hidden_layers`` is updated, and the result is saved to
    ``output_path`` as a regular HF model directory.

    Returns a metadata dict describing the pruning decisions.
    """
    cfg = cfg or BlockPrunerConfig()
    from transformers import (  # type: ignore[import-untyped]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype="auto"
    )
    num_layers = model.config.num_hidden_layers

    # Default calibration corpus if the caller didn't supply one.
    if calibration_texts is None:
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a shocking turn of events, researchers discovered that",
            "def fibonacci(n):\n    if n < 2:\n        return n",
            "The history of Apple Silicon begins with the M1 chip.",
        ] * (cfg.calibration_samples // 4 + 1)
        calibration_texts = calibration_texts[: cfg.calibration_samples]

    hidden_states = _collect_residual_streams(
        model, tokenizer, calibration_texts, cfg.max_length
    )

    # Score each block using input/output residual streams.
    scores: list[BlockImportance] = []
    for i in range(num_layers):
        h_in = hidden_states[i]
        h_out = hidden_states[i + 1]
        # Align shapes when calibration produced a flattened fallback.
        min_rows = min(h_in.shape[0], h_out.shape[0])
        s = _block_importance_scores(h_in[:min_rows], h_out[:min_rows], cfg.metric)
        scores.append(BlockImportance(layer_index=i, score=s))

    # Determine how many to drop.
    if cfg.target_layers is not None:
        keep_count = cfg.target_layers
    else:
        ratio = cfg.prune_ratio or 0.0
        keep_count = max(1, int(round(num_layers * (1.0 - ratio))))

    must_keep = set(range(cfg.preserve_first)) | set(
        range(num_layers - cfg.preserve_last, num_layers)
    )
    prunable = [s for s in scores if s.layer_index not in must_keep]
    prunable_sorted = sorted(prunable, key=lambda s: s.score)  # ascending
    drop_count = max(0, num_layers - keep_count)
    to_drop = {s.layer_index for s in prunable_sorted[:drop_count]}
    kept_indices = [i for i in range(num_layers) if i not in to_drop]

    # Mutate the model's layer list in place.
    layer_container = _find_layer_container(model)
    new_layers = type(layer_container)([layer_container[i] for i in kept_indices])
    _replace_layer_container(model, new_layers)
    model.config.num_hidden_layers = len(kept_indices)

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    meta = {
        "method": "shortgpt_block_prune",
        "original_layers": num_layers,
        "kept_layers": len(kept_indices),
        "dropped_layers": sorted(to_drop),
        "metric": cfg.metric,
        "scores": [{"layer": s.layer_index, "score": s.score} for s in scores],
    }
    (output_path / "pruning_report.json").write_text(json.dumps(meta, indent=2))
    return meta


def _find_layer_container(model: Any) -> Any:
    """Locate the ``nn.ModuleList`` holding transformer blocks.

    Handles common HF layouts: ``model.model.layers`` (Llama/Qwen/etc.),
    ``model.transformer.h`` (GPT-2/Falcon), ``model.gpt_neox.layers``.
    """
    for path in (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    ):
        node: Any = model
        ok = True
        for name in path:
            if not hasattr(node, name):
                ok = False
                break
            node = getattr(node, name)
        if ok:
            return node
    raise RuntimeError("Could not locate transformer layer container on model")


def _replace_layer_container(model: Any, new_container: Any) -> None:
    for path in (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    ):
        node: Any = model
        ok = True
        for name in path[:-1]:
            if not hasattr(node, name):
                ok = False
                break
            node = getattr(node, name)
        if ok and hasattr(node, path[-1]):
            setattr(node, path[-1], new_container)
            return
    raise RuntimeError("Could not replace transformer layer container on model")


# ---------------------------------------------------------------------------
# Activation-aware SVD (ASVD)
# ---------------------------------------------------------------------------


@dataclass
class ASVDConfig:
    """Configuration for :class:`ASVDDecomposer`.

    Attributes
    ----------
    target_ratio:
        Average rank ratio across layers (0 < ratio <= 1). A ratio of
        0.5 cuts parameter count in half.
    error_budget:
        Per-layer Frobenius reconstruction error budget, as a fraction
        of the original activation-weighted matrix norm. If set, it
        overrides ``target_ratio`` and picks the smallest rank that stays
        under this error.
    min_rank:
        Minimum allowed rank (safety floor).
    """

    target_ratio: float = 0.5
    error_budget: float | None = None
    min_rank: int = 8


@dataclass
class ASVDLayerResult:
    name: str
    original_shape: tuple[int, int]
    rank: int
    error: float


def asvd_decompose(
    weight: mx.array | np.ndarray,
    activation_scale: mx.array | np.ndarray | None = None,
    cfg: ASVDConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Activation-aware SVD decomposition of a linear weight.

    Parameters
    ----------
    weight:
        2-D matrix ``(out_features, in_features)``.
    activation_scale:
        Optional 1-D tensor of length ``in_features`` giving the RMS
        activation magnitude per input channel. Columns with larger
        activations are weighted more heavily during SVD so the
        reconstruction prioritizes them (this is the "activation-aware"
        part of ASVD). If None, falls back to plain SVD.
    cfg:
        :class:`ASVDConfig`.

    Returns
    -------
    (U, V, error) where ``W ≈ U @ V``; ``U`` has shape
    ``(out_features, rank)`` and ``V`` has shape ``(rank, in_features)``.
    """
    cfg = cfg or ASVDConfig()
    W = np.asarray(weight, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"asvd_decompose expects 2-D weight, got shape {W.shape}")
    out_features, in_features = W.shape

    if activation_scale is not None:
        s = np.asarray(activation_scale, dtype=np.float32).reshape(-1)
        if s.shape[0] != in_features:
            raise ValueError("activation_scale must have length in_features")
        s = np.clip(s, 1e-8, None)
        W_scaled = W * s[None, :]
    else:
        s = None
        W_scaled = W

    U_full, sing, Vt_full = np.linalg.svd(W_scaled, full_matrices=False)

    # Choose rank.
    total_energy = float(np.sum(sing**2))
    if cfg.error_budget is not None and total_energy > 0:
        cum = np.cumsum(sing[::-1] ** 2)[::-1] / total_energy
        # smallest rank where residual energy <= error_budget^2
        budget_sq = cfg.error_budget**2
        mask = cum <= budget_sq
        rank_from_budget = int(np.argmax(mask)) if mask.any() else len(sing)
        rank = max(cfg.min_rank, rank_from_budget)
    else:
        rank = max(cfg.min_rank, int(round(cfg.target_ratio * min(out_features, in_features))))
    rank = min(rank, len(sing))

    U = U_full[:, :rank] * sing[:rank][None, :]
    Vt = Vt_full[:rank, :]
    if s is not None:
        Vt = Vt / s[None, :]

    recon = U @ Vt
    error = float(np.linalg.norm(W - recon) / (np.linalg.norm(W) + 1e-8))
    return U, Vt, error


class ASVDDecomposer:
    """Apply :func:`asvd_decompose` to every linear layer of a HF model.

    The output is a sibling directory with each ``Linear(out, in)``
    replaced by two sequential linears ``Linear(rank, in)`` ->
    ``Linear(out, rank)``. We emit a plain HF model by modifying the
    state dict in place and relying on downstream conversion (which
    treats the decomposed weights as regular linears) — this keeps the
    output compatible with vanilla HF loaders at the cost of not
    reducing inference FLOPs unless the loader is taught about the
    factorization. A production variant would also patch the model's
    ``forward`` to use the factored form directly.
    """

    def __init__(self, cfg: ASVDConfig | None = None):
        self.cfg = cfg or ASVDConfig()
        self.results: list[ASVDLayerResult] = []

    def decompose_directory(
        self,
        model_path: str | Path,
        output_path: str | Path,
        activation_scales: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Walk a HF model, decompose each linear weight, and save results.

        Parameters
        ----------
        model_path:
            Source HF model directory.
        output_path:
            Destination directory for the decomposed model.
        activation_scales:
            Optional mapping ``{weight_name: per_input_channel_scale}``
            produced by a calibration run (e.g. AWQ-style). When
            omitted, falls back to plain SVD rank reduction.
        """
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype="auto"
        )
        import torch  # type: ignore[import-untyped]

        self.results = []
        with torch.no_grad():
            for name, module in model.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                # Skip the LM head and embeddings to preserve output quality.
                if any(k in name.lower() for k in ("lm_head", "embed")):
                    continue
                W = module.weight.detach().cpu().float().numpy()
                act_scale = (
                    activation_scales.get(name + ".weight") if activation_scales else None
                )
                U, Vt, err = asvd_decompose(W, act_scale, self.cfg)
                approx = torch.from_numpy((U @ Vt).astype(np.float32))
                module.weight.copy_(approx.to(module.weight.dtype))
                self.results.append(
                    ASVDLayerResult(
                        name=name,
                        original_shape=tuple(W.shape),
                        rank=int(U.shape[1]),
                        error=err,
                    )
                )

        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        meta = {
            "method": "asvd",
            "target_ratio": self.cfg.target_ratio,
            "error_budget": self.cfg.error_budget,
            "layers": [
                {
                    "name": r.name,
                    "shape": list(r.original_shape),
                    "rank": r.rank,
                    "error": r.error,
                }
                for r in self.results
            ],
        }
        (output_path / "asvd_report.json").write_text(json.dumps(meta, indent=2))
        return meta


__all__ = [
    "BlockPrunerConfig",
    "BlockImportance",
    "prune_blocks",
    "ASVDConfig",
    "ASVDLayerResult",
    "ASVDDecomposer",
    "asvd_decompose",
]
