"""MoE-SVD Expert Merger — shared-V basis via joint SVD.

Compresses a group of MoE experts by computing a single shared basis
(the top-k right singular vectors of the concatenated expert weights)
and expressing each expert as a small per-expert coefficient matrix
against that basis.

Reference:
- Sub-MoE (2506.23266): Joint SVD merging, 96% quality retention
- MoE-SVD (ICML 2025): 60% compression, 1.5x inference speedup

Reconstruction for expert e is:

    W_e ≈ U_e  @  diag(S_e)  @  V_shared.T            # full form
    W_e ≈ coeffs_e @ V_shared.T                       # compressed form

Storage: one `shared_basis` of shape (rank, dim) per group, plus per-expert
`coeffs` of shape (out_dim, rank). Compression ratio ≈ num_experts × dim
divided by (rank × dim + num_experts × rank × out_dim).

This module is a reference numpy/MLX implementation — it produces a
deserializable state dict and a verified reconstruction, but does not
wire the merged experts into the MLX inference path. Downstream kernels
can consume `{shared_basis, expert_coeffs}` directly.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MergeConfig:
    """Configuration for SubMoE expert merging."""

    rank: int = 8
    """Target rank of the shared basis (number of singular vectors kept)."""
    num_groups: int = 1
    """Number of expert groups to partition the full expert set into."""
    center_weights: bool = False
    """If True, subtract the per-group mean weight before SVD."""
    dtype: str = "float32"
    """Numpy dtype used for the internal SVD computation."""


@dataclass
class MergedGroup:
    """Result of merging a single group of experts."""

    group_id: int
    expert_ids: list[int]
    shared_basis: Any  # numpy array, shape (rank, in_dim)
    expert_coeffs: list[Any]  # list of numpy arrays, shape (out_dim, rank)
    mean_weight: Any | None = None  # optional per-group mean (if centered)
    original_shape: tuple[int, int] = (0, 0)
    reconstruction_mse: float = 0.0


@dataclass
class MergeReport:
    """Summary of a merging run."""

    model_id: str
    config: MergeConfig
    groups: list[MergedGroup] = field(default_factory=list)
    total_original_params: int = 0
    total_compressed_params: int = 0
    mean_reconstruction_mse: float = 0.0

    @property
    def compression_ratio(self) -> float:
        if self.total_compressed_params == 0:
            return 0.0
        return self.total_original_params / self.total_compressed_params


# ---------------------------------------------------------------------------
# Core SVD merger
# ---------------------------------------------------------------------------


def merge_experts_svd(
    expert_weights: list[Any],
    rank: int,
    center: bool = False,
) -> tuple[Any, list[Any], Any | None]:
    """Joint-SVD merge a group of expert weight matrices.

    Each expert weight `W_e` is assumed to have shape (out_dim, in_dim)
    and the same shape across the group. The experts are concatenated
    along the row axis to form a tall matrix of shape
    `(num_experts * out_dim, in_dim)`; a truncated SVD extracts the top
    `rank` right singular vectors as the shared basis `V ∈ (rank, in_dim)`.
    Each expert is then projected onto `V.T` to obtain per-expert
    coefficients `C_e ∈ (out_dim, rank)` such that `W_e ≈ C_e @ V`.

    Returns: (shared_basis, [coeffs_per_expert], mean_weight or None)
    """
    import numpy as np

    if not expert_weights:
        raise ValueError("expert_weights must be non-empty")

    mats = [np.asarray(w, dtype=np.float32) for w in expert_weights]
    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(
            f"All experts must share the same shape; got {shapes}"
        )
    out_dim, in_dim = mats[0].shape

    # Optionally center around per-group mean
    mean = None
    if center:
        mean = np.mean(np.stack(mats, axis=0), axis=0)
        mats = [m - mean for m in mats]

    # Stack experts along rows: (num_experts * out_dim, in_dim)
    stacked = np.concatenate(mats, axis=0)

    # Truncated SVD. numpy returns U (m, k), s (k,), Vt (k, in_dim).
    # We request full_matrices=False for efficiency.
    U, s, Vt = np.linalg.svd(stacked, full_matrices=False)
    r = max(1, min(rank, Vt.shape[0]))
    V_shared = Vt[:r, :]                       # (rank, in_dim)

    # Project each expert onto the shared basis: C_e = W_e @ V_shared.T
    coeffs: list[Any] = []
    for m in mats:
        c = m @ V_shared.T                      # (out_dim, rank)
        coeffs.append(c.astype(np.float32))

    return V_shared.astype(np.float32), coeffs, mean


def _reconstruct_expert(
    shared_basis: Any,
    coeffs: Any,
    mean: Any | None = None,
) -> Any:
    """Reconstruct a single expert weight matrix from its compressed form."""
    import numpy as np

    W = coeffs @ shared_basis
    if mean is not None:
        W = W + mean
    return W.astype(np.float32)


def verify_reconstruction(
    expert_weights: list[Any],
    shared_basis: Any,
    coeffs: list[Any],
    mean: Any | None = None,
    tol: float = 1e-2,
) -> tuple[bool, float]:
    """Self-test: check reconstructed experts are close to originals.

    Returns (ok, max_relative_error). The tolerance is relative to the
    Frobenius norm of each original matrix, so low-rank approximations
    that keep most of the energy will pass.
    """
    import numpy as np

    max_rel = 0.0
    for W_orig, c in zip(expert_weights, coeffs):
        W_orig = np.asarray(W_orig, dtype=np.float32)
        W_rec = _reconstruct_expert(shared_basis, c, mean)
        denom = float(np.linalg.norm(W_orig)) or 1.0
        err = float(np.linalg.norm(W_orig - W_rec)) / denom
        if err > max_rel:
            max_rel = err
    return max_rel <= tol, max_rel


# ---------------------------------------------------------------------------
# High-level SubMoEMerger class
# ---------------------------------------------------------------------------


class SubMoEMerger:
    """Fit/apply/save driver for MoE-SVD expert compression.

    Typical usage::

        merger = SubMoEMerger(MergeConfig(rank=8, num_groups=2))
        merger.fit(experts)              # experts: list[list[np.ndarray]]
        merger.apply()                   # populates the state dict
        merger.save(Path("out.subm"))    # writes {shared_basis, expert_coeffs}
    """

    def __init__(self, config: MergeConfig):
        self.config = config
        self.groups: list[MergedGroup] = []
        self._state_dict: dict[str, Any] | None = None
        self._fitted = False

    # ----- fitting ----------------------------------------------------

    def fit(
        self,
        experts: list[Any] | list[list[Any]],
        expert_ids: list[int] | None = None,
    ) -> "SubMoEMerger":
        """Compute the shared basis + per-expert coefficients.

        `experts` may be either:
            - A flat list of expert weight matrices — will be partitioned
              into `config.num_groups` contiguous groups.
            - A list of groups, each already a list of weight matrices.
        """
        if not experts:
            raise ValueError("experts must be non-empty")

        # Detect flat vs grouped
        grouped: list[list[Any]]
        first = experts[0]
        if hasattr(first, "ndim") or (
            hasattr(first, "shape") and not isinstance(first, list)
        ):
            # Flat list — partition
            grouped = _partition(list(experts), self.config.num_groups)
            if expert_ids is None:
                expert_ids = list(range(len(experts)))
            group_ids: list[list[int]] = _partition(list(expert_ids), self.config.num_groups)
        else:
            grouped = [list(g) for g in experts]  # type: ignore[arg-type]
            if expert_ids is None:
                group_ids = [list(range(len(g))) for g in grouped]
            else:
                # Assume caller provided matching nested ids
                group_ids = expert_ids  # type: ignore[assignment]

        self.groups = []
        for gi, (group, ids) in enumerate(zip(grouped, group_ids)):
            if not group:
                continue
            V, coeffs, mean = merge_experts_svd(
                group,
                rank=self.config.rank,
                center=self.config.center_weights,
            )
            # Diagnostics
            ok, err = verify_reconstruction(group, V, coeffs, mean, tol=1.0)
            self.groups.append(MergedGroup(
                group_id=gi,
                expert_ids=list(ids),
                shared_basis=V,
                expert_coeffs=coeffs,
                mean_weight=mean,
                original_shape=tuple(group[0].shape),  # type: ignore[arg-type]
                reconstruction_mse=float(err),
            ))

        self._fitted = True
        return self

    # ----- applying ---------------------------------------------------

    def apply(self) -> dict[str, Any]:
        """Build the serializable state dict.

        Format::

            {
                "config": {...},
                "groups": [
                    {
                        "group_id": 0,
                        "expert_ids": [...],
                        "shared_basis": np.ndarray,
                        "expert_coeffs": [np.ndarray, ...],
                        "mean_weight": np.ndarray | None,
                        "original_shape": (out_dim, in_dim),
                    },
                    ...
                ],
            }
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before apply().")

        state: dict[str, Any] = {
            "config": {
                "rank": self.config.rank,
                "num_groups": self.config.num_groups,
                "center_weights": self.config.center_weights,
                "dtype": self.config.dtype,
            },
            "groups": [],
        }
        for g in self.groups:
            state["groups"].append({
                "group_id": g.group_id,
                "expert_ids": g.expert_ids,
                "shared_basis": g.shared_basis,
                "expert_coeffs": g.expert_coeffs,
                "mean_weight": g.mean_weight,
                "original_shape": g.original_shape,
                "reconstruction_mse": g.reconstruction_mse,
            })
        self._state_dict = state
        return state

    # ----- saving -----------------------------------------------------

    def save(self, path: Path) -> Path:
        """Persist the state dict to disk.

        Writes two files:
            - `<path>`         — pickled state dict (numpy arrays intact)
            - `<path>.json`    — human-readable manifest alongside it
        """
        if self._state_dict is None:
            self.apply()
        assert self._state_dict is not None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._state_dict, f)

        manifest = {
            "format": "submoe-svd/v1",
            "config": self._state_dict["config"],
            "num_groups": len(self.groups),
            "groups": [
                {
                    "group_id": g.group_id,
                    "num_experts": len(g.expert_ids),
                    "expert_ids": g.expert_ids,
                    "rank": self.config.rank,
                    "original_shape": list(g.original_shape),
                    "reconstruction_rel_error": g.reconstruction_mse,
                }
                for g in self.groups
            ],
        }
        json_path = path.with_suffix(path.suffix + ".json")
        json_path.write_text(json.dumps(manifest, indent=2))
        return path

    # ----- loading ----------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> dict[str, Any]:
        """Load a previously saved state dict."""
        with Path(path).open("rb") as f:
            return pickle.load(f)

    # ----- reporting --------------------------------------------------

    def report(self, model_id: str = "") -> MergeReport:
        """Build a MergeReport summarizing compression + quality."""
        import numpy as np

        r = MergeReport(model_id=model_id, config=self.config, groups=self.groups)
        for g in self.groups:
            n_experts = len(g.expert_coeffs)
            out_dim, in_dim = g.original_shape
            r.total_original_params += n_experts * out_dim * in_dim
            basis_params = int(np.asarray(g.shared_basis).size)
            coeff_params = sum(int(np.asarray(c).size) for c in g.expert_coeffs)
            r.total_compressed_params += basis_params + coeff_params
        if self.groups:
            r.mean_reconstruction_mse = sum(
                g.reconstruction_mse for g in self.groups
            ) / len(self.groups)
        return r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _partition(items: list[Any], n: int) -> list[list[Any]]:
    """Split `items` into `n` roughly-equal contiguous groups."""
    if n <= 1 or len(items) <= 1:
        return [items]
    size = max(1, (len(items) + n - 1) // n)
    return [items[i:i + size] for i in range(0, len(items), size)]


def format_merge_report(report: MergeReport) -> str:
    """Format a MergeReport for CLI display."""
    lines = [
        "MoE-SVD Expert Merge Report",
        "=" * 55,
        f"  Model:            {report.model_id or '(unknown)'}",
        f"  Rank:             {report.config.rank}",
        f"  Groups:           {len(report.groups)}",
        f"  Orig params:      {report.total_original_params:,}",
        f"  Compressed:       {report.total_compressed_params:,}",
        f"  Compression:      {report.compression_ratio:.2f}x",
        f"  Mean recon err:   {report.mean_reconstruction_mse:.4f}",
        "",
    ]
    for g in report.groups:
        lines.append(
            f"  Group {g.group_id}: {len(g.expert_ids)} experts, "
            f"shape={g.original_shape}, rel_err={g.reconstruction_mse:.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (run directly or from unit tests)
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    """Synthetic end-to-end check: generate 4 low-rank experts, merge, verify."""
    import numpy as np

    rng = np.random.default_rng(0)
    out_dim, in_dim, true_rank = 32, 48, 6
    # Build 4 experts that share a low-rank subspace + small noise
    V_true = rng.standard_normal((true_rank, in_dim)).astype(np.float32)
    experts = []
    for _ in range(4):
        U = rng.standard_normal((out_dim, true_rank)).astype(np.float32)
        W = U @ V_true + 0.01 * rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        experts.append(W)

    merger = SubMoEMerger(MergeConfig(rank=8, num_groups=1))
    merger.fit(experts)
    state = merger.apply()

    g0 = state["groups"][0]
    ok, err = verify_reconstruction(
        experts, g0["shared_basis"], g0["expert_coeffs"], g0["mean_weight"],
        tol=0.05,
    )
    return ok and err < 0.05


if __name__ == "__main__":
    print("SubMoEMerger self-test:", "OK" if _self_test() else "FAIL")
