"""xKV / CommonKV: Cross-Layer SVD KV Cache Sharing."""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# xKV / CommonKV: Cross-Layer SVD KV Cache Sharing
# (arXiv 2503.18893 xKV, arXiv 2508.16134 CommonKV)
#
# Key insight: K and V tensors across neighboring transformer layers are
# strongly correlated. Instead of storing every layer's full K,V, group
# layers and extract a shared low-rank basis via SVD. Each layer then only
# stores a small coefficient matrix that projects onto that basis.
#
# Reported: up to ~6.8x extra compression on top of standard KV quantization.
# This implementation is offline (fit once after prefill or on calibration
# data); runtime attention materializes K,V via `reconstruct()`.
# ---------------------------------------------------------------------------


@dataclass
class XKVConfig:
    """Configuration for cross-layer SVD KV compression (xKV / CommonKV)."""

    group_size: int = 4        # Number of contiguous layers sharing a basis
    rank: int = 128            # Rank of the shared basis (columns kept)
    method: str = "svd"        # Decomposition method: currently "svd" only


@dataclass
class XKVGroup:
    """A group of layers that share a low-rank basis for K and V.

    Attributes:
        layer_indices: Transformer layer indices belonging to this group.
        shared_basis_k: Shape (rank, head_dim). Right singular vectors of the
            stacked K matrix for this group.
        shared_basis_v: Shape (rank, head_dim). Same, for V.
        per_layer_k_coeffs: layer_idx → array of shape
            (num_heads * seq_len, rank). Projection coefficients.
        per_layer_v_coeffs: layer_idx → array of shape
            (num_heads * seq_len, rank). Projection coefficients.
        num_heads: Number of heads (for reshape on reconstruct).
        seq_len: Sequence length captured at fit time.
        head_dim: Head dimension.
    """

    layer_indices: list[int]
    shared_basis_k: object  # np.ndarray
    shared_basis_v: object  # np.ndarray
    per_layer_k_coeffs: dict[int, object] = field(default_factory=dict)
    per_layer_v_coeffs: dict[int, object] = field(default_factory=dict)
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0


class XKVCompressor:
    """Cross-layer SVD compressor for KV caches (xKV / CommonKV).

    Offline pipeline:
        1. Group layers into contiguous chunks of `group_size`.
        2. For each group, stack per-layer K tensors along the token axis to
           form a tall matrix of shape (group_size * num_heads * seq_len, head_dim).
        3. Run SVD, keep top-`rank` right singular vectors as shared basis B_K
           of shape (rank, head_dim).
        4. Project each layer's flattened K onto B_K to obtain coefficients
           of shape (num_heads * seq_len, rank).
        5. Repeat for V.

    Runtime:
        `reconstruct(group, layer_idx)` returns approximate (K, V) tensors for
        one layer reshaped back to (num_heads, seq_len, head_dim).
    """

    def __init__(self, config: XKVConfig | None = None):
        self.config = config or XKVConfig()

    # --- fit ---------------------------------------------------------------

    def fit(
        self,
        layer_kv_tensors: dict[int, tuple[object, object]],
        num_groups: int,
    ) -> list[XKVGroup]:
        """Fit shared bases across layer groups.

        Args:
            layer_kv_tensors: layer_idx → (K, V) where each is an mx.array or
                numpy array of shape (num_heads, seq_len, head_dim).
            num_groups: Number of layer groups to form. If the layer count is
                not evenly divisible, the last group absorbs the remainder.

        Returns:
            List of `XKVGroup` instances (length == num_groups).
        """
        import numpy as np

        layer_ids = sorted(layer_kv_tensors.keys())
        L = len(layer_ids)
        if L == 0:
            return []

        num_groups = max(1, min(num_groups, L))
        # Roughly balanced contiguous grouping
        base = L // num_groups
        rem = L % num_groups
        groups_layer_ids: list[list[int]] = []
        cursor = 0
        for g in range(num_groups):
            size = base + (1 if g < rem else 0)
            groups_layer_ids.append(layer_ids[cursor : cursor + size])
            cursor += size

        rank = self.config.rank
        out: list[XKVGroup] = []

        for gids in groups_layer_ids:
            # Gather K and V stacks (convert mlx -> numpy if needed)
            k_mats = []
            v_mats = []
            num_heads = seq_len = head_dim = 0
            for lid in gids:
                K, V = layer_kv_tensors[lid]
                K_np = _to_numpy(K)
                V_np = _to_numpy(V)
                num_heads, seq_len, head_dim = K_np.shape
                k_mats.append(K_np.reshape(-1, head_dim))
                v_mats.append(V_np.reshape(-1, head_dim))

            K_stack = np.concatenate(k_mats, axis=0)  # (g * nh * s, head_dim)
            V_stack = np.concatenate(v_mats, axis=0)

            # Effective rank is capped by head_dim (right singular space) and
            # the number of rows.
            eff_rank = min(rank, head_dim, K_stack.shape[0])

            # SVD: X = U @ diag(S) @ Vt, Vt has shape (head_dim, head_dim).
            # Right singular vectors are rows of Vt.
            _, _, Vt_k = np.linalg.svd(K_stack, full_matrices=False)
            _, _, Vt_v = np.linalg.svd(V_stack, full_matrices=False)
            basis_k = Vt_k[:eff_rank, :]  # (rank, head_dim)
            basis_v = Vt_v[:eff_rank, :]

            # Per-layer projection coefficients: X @ basis.T  →  (rows, rank)
            per_layer_k: dict[int, object] = {}
            per_layer_v: dict[int, object] = {}
            for lid, k_flat, v_flat in zip(gids, k_mats, v_mats):
                per_layer_k[lid] = k_flat @ basis_k.T
                per_layer_v[lid] = v_flat @ basis_v.T

            out.append(
                XKVGroup(
                    layer_indices=list(gids),
                    shared_basis_k=basis_k,
                    shared_basis_v=basis_v,
                    per_layer_k_coeffs=per_layer_k,
                    per_layer_v_coeffs=per_layer_v,
                    num_heads=num_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                )
            )

        return out

    # --- reconstruct -------------------------------------------------------

    def reconstruct(self, group: XKVGroup, layer_idx: int) -> tuple[object, object]:
        """Reconstruct approximate (K, V) for one layer from a fitted group.

        Returns numpy arrays shaped (num_heads, seq_len, head_dim).
        """
        if layer_idx not in group.per_layer_k_coeffs:
            raise KeyError(
                f"Layer {layer_idx} not present in group (layers={group.layer_indices})"
            )

        k_coeff = group.per_layer_k_coeffs[layer_idx]     # (nh*s, rank)
        v_coeff = group.per_layer_v_coeffs[layer_idx]
        K_flat = k_coeff @ group.shared_basis_k           # (nh*s, head_dim)
        V_flat = v_coeff @ group.shared_basis_v
        shape = (group.num_heads, group.seq_len, group.head_dim)
        return K_flat.reshape(shape), V_flat.reshape(shape)

    # --- size accounting ---------------------------------------------------

    @staticmethod
    def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
        """Return original_bytes / compressed_bytes (guarded against zero)."""
        if compressed_bytes <= 0:
            return float("inf")
        return float(original_bytes) / float(compressed_bytes)

    # --- persistence -------------------------------------------------------

    def save(self, groups: list[XKVGroup], path: str) -> None:
        """Persist fitted groups to a .npz archive at `path`."""
        import numpy as np

        payload: dict[str, object] = {
            "num_groups": np.array(len(groups)),
            "config_group_size": np.array(self.config.group_size),
            "config_rank": np.array(self.config.rank),
            "config_method": np.array(self.config.method),
        }
        for g_idx, g in enumerate(groups):
            prefix = f"g{g_idx}_"
            payload[prefix + "layers"] = np.array(g.layer_indices, dtype=np.int64)
            payload[prefix + "basis_k"] = np.asarray(g.shared_basis_k)
            payload[prefix + "basis_v"] = np.asarray(g.shared_basis_v)
            payload[prefix + "num_heads"] = np.array(g.num_heads)
            payload[prefix + "seq_len"] = np.array(g.seq_len)
            payload[prefix + "head_dim"] = np.array(g.head_dim)
            for lid in g.layer_indices:
                payload[f"{prefix}k_{lid}"] = np.asarray(g.per_layer_k_coeffs[lid])
                payload[f"{prefix}v_{lid}"] = np.asarray(g.per_layer_v_coeffs[lid])
        np.savez(path, **payload)

    def load(self, path: str) -> list[XKVGroup]:
        """Load fitted groups previously saved via `save()`."""
        import numpy as np

        data = np.load(path, allow_pickle=False)
        num_groups = int(data["num_groups"])
        groups: list[XKVGroup] = []
        for g_idx in range(num_groups):
            prefix = f"g{g_idx}_"
            layers = data[prefix + "layers"].tolist()
            basis_k = data[prefix + "basis_k"]
            basis_v = data[prefix + "basis_v"]
            num_heads = int(data[prefix + "num_heads"])
            seq_len = int(data[prefix + "seq_len"])
            head_dim = int(data[prefix + "head_dim"])
            per_layer_k = {lid: data[f"{prefix}k_{lid}"] for lid in layers}
            per_layer_v = {lid: data[f"{prefix}v_{lid}"] for lid in layers}
            groups.append(
                XKVGroup(
                    layer_indices=layers,
                    shared_basis_k=basis_k,
                    shared_basis_v=basis_v,
                    per_layer_k_coeffs=per_layer_k,
                    per_layer_v_coeffs=per_layer_v,
                    num_heads=num_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                )
            )
        return groups


def _to_numpy(arr):
    """Best-effort conversion from mlx.core.array / numpy / list to ndarray."""
    import numpy as np

    if isinstance(arr, np.ndarray):
        return arr
    # mlx arrays expose tolist(); some builds also support np.array(arr) directly.
    try:
        return np.asarray(arr)
    except Exception:  # noqa: BLE001
        return np.array(arr.tolist())


def estimate_xkv_compression(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    group_size: int,
    rank: int,
    dtype_bytes: int = 2,
) -> dict:
    """Analytically estimate xKV compression ratio without running SVD.

    Original per-layer K or V size (bytes):
        num_heads * seq_len * head_dim * dtype_bytes

    Compressed per group:
        shared_basis:   rank * head_dim * dtype_bytes   (shared across the group)
        per-layer coef: num_heads * seq_len * rank * dtype_bytes

    Both K and V follow the same formula, so the total is 2x.
    """
    eff_rank = min(rank, head_dim)
    per_layer_original = num_heads * seq_len * head_dim * dtype_bytes
    original_total = 2 * num_layers * per_layer_original  # 2 for K + V

    num_groups = max(1, (num_layers + group_size - 1) // group_size)
    basis_bytes_per_group = eff_rank * head_dim * dtype_bytes
    coeff_bytes_per_layer = num_heads * seq_len * eff_rank * dtype_bytes
    compressed_total = 2 * (
        num_groups * basis_bytes_per_group
        + num_layers * coeff_bytes_per_layer
    )

    ratio = original_total / compressed_total if compressed_total else float("inf")
    return {
        "num_layers": num_layers,
        "num_groups": num_groups,
        "effective_rank": eff_rank,
        "original_bytes": original_total,
        "compressed_bytes": compressed_total,
        "compression_ratio": ratio,
        "original_mb": original_total / (1024 * 1024),
        "compressed_mb": compressed_total / (1024 * 1024),
        "savings_pct": (1.0 - compressed_total / original_total) * 100.0
        if original_total
        else 0.0,
    }
