"""LAVa: Unified Layer + Head Eviction via Residual Information Loss."""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# LAVa: Unified Layer + Head Eviction via Residual Information Loss
# (arXiv 2509.09754, Sep 2025)
#
# Generalizes Ada-KV by adding the layer dimension. Instead of computing a
# per-head budget in isolation for each layer, LAVa derives a single
# information-loss metric from residual-stream perturbation analysis and uses
# it to allocate a global budget across both layers AND heads jointly.
#
# Metric (simplified surrogate of the paper's derivation):
#   loss(token t @ layer L, head H) = attention_weight(t) * value_norm(t)
#
# This quantifies how much the residual stream would be perturbed if token t
# were removed from head H of layer L. Heads/layers with concentrated
# high-loss tokens get smaller budgets; heads/layers with diffuse loss get
# larger budgets.
# ---------------------------------------------------------------------------


@dataclass
class LAVaConfig:
    """Configuration for LAVa unified layer + head KV eviction."""

    total_budget_ratio: float = 0.2       # Global KV budget as fraction of full cache
    layer_weight_alpha: float = 0.5        # Layer-level allocation power
                                            # (0=uniform, 1=proportional)
    head_weight_alpha: float = 0.5         # Head-level allocation power (0=uniform, 1=proportional)
    min_layer_budget: int = 32             # Minimum tokens kept per layer
    min_head_budget: int = 8               # Minimum tokens kept per head
    window_size: int = 32                  # Always-keep recent tokens (local window)
    sink_size: int = 4                     # Always-keep attention sinks (first N)


@dataclass
class LAVaStats:
    """Statistics about LAVa eviction decisions."""

    total_evictions: int = 0
    layer_budget_distribution: dict[int, int] = field(default_factory=dict)
    head_budget_distribution_per_layer: dict[int, list[int]] = field(default_factory=dict)
    information_loss_score: float = 0.0


class LAVaManager:
    """LAVa: unified layer-wise + head-wise KV eviction.

    Tracks a running "information loss" score per (layer, head, token) derived
    from attention * value_norm. Allocates a global KV budget across layers
    (proportional to layer-level info mass) and then within each layer across
    heads (proportional to head-level info mass), enforcing configurable
    minimum budgets. Tokens kept per head always include attention sinks and
    the recent window; the rest of the budget is filled by top-information
    tokens.
    """

    def __init__(self, config: LAVaConfig | None, num_layers: int, num_heads: int):
        self.config = config or LAVaConfig()
        self.num_layers = num_layers
        self.num_heads = num_heads
        # (layer_idx, head_idx) -> 1D mx.array of per-token information-loss scores
        self._info_loss: dict[tuple[int, int], object] = {}
        # (layer_idx, head_idx) -> scalar total info-mass (sum of loss)
        self._info_mass: dict[tuple[int, int], float] = {}
        self._stats = LAVaStats()

    # --- metric ------------------------------------------------------------

    def compute_residual_info_loss(
        self,
        layer_idx: int,
        head_idx: int,
        attention_scores,
        value_norms,
    ) -> float:
        """Compute residual-stream information loss per token for one head.

        Args:
            layer_idx: Transformer layer index.
            head_idx: Attention head index within the layer.
            attention_scores: 1D mx.array of shape (seq_len,) — attention mass
                this head places on each key position (averaged or summed
                across query positions for the current step).
            value_norms: 1D mx.array of shape (seq_len,) — L2 norm of the
                value vector at each token position for this head.

        Returns:
            Total information loss (sum over tokens) for reporting. The
            per-token loss vector is stored internally.
        """
        import mlx.core as mx

        if attention_scores.ndim > 1:
            attention_scores = mx.mean(attention_scores, axis=0)
        if value_norms.ndim > 1:
            value_norms = mx.mean(value_norms, axis=0)

        per_token = attention_scores * value_norms
        mx.eval(per_token)

        key = (layer_idx, head_idx)
        self._info_loss[key] = per_token
        total = float(mx.sum(per_token).item())
        self._info_mass[key] = total
        return total

    def update(self, layer_idx: int, head_idx: int, attention_scores, value_norms) -> None:
        """Record per-token information-loss history for one (layer, head)."""
        total = self.compute_residual_info_loss(
            layer_idx, head_idx, attention_scores, value_norms
        )
        self._stats.information_loss_score += total

    # --- budget allocation -------------------------------------------------

    def allocate_budgets(self, total_budget: int) -> dict[int, list[int]]:
        """Split a global token budget across layers and heads.

        Allocation rule:
          1. Compute layer_mass[L] = sum over heads of info_mass[L, H].
          2. Raise to alpha power to interpolate uniform ↔ proportional.
          3. Split total_budget across layers proportional to layer weight,
             enforcing min_layer_budget.
          4. Within each layer, split its layer budget across heads
             proportional to head_mass[L, H]^alpha, enforcing min_head_budget.

        Returns:
            Mapping layer_idx → list of per-head budgets (length num_heads).
        """
        cfg = self.config
        L, H = self.num_layers, self.num_heads

        # 1. Per-layer mass (fall back to uniform if no data yet)
        layer_mass: list[float] = []
        head_mass_per_layer: list[list[float]] = []
        for lyr in range(L):
            head_masses = [self._info_mass.get((lyr, h), 0.0) for h in range(H)]
            head_mass_per_layer.append(head_masses)
            layer_mass.append(sum(head_masses))

        total_mass = sum(layer_mass)
        if total_mass <= 0.0:
            # No observations yet — uniform split
            uniform_layer = max(cfg.min_layer_budget, total_budget // max(L, 1))
            uniform_head = max(cfg.min_head_budget, uniform_layer // max(H, 1))
            result = {lyr: [uniform_head] * H for lyr in range(L)}
            self._stats.layer_budget_distribution = {
                lyr: uniform_head * H for lyr in range(L)
            }
            self._stats.head_budget_distribution_per_layer = {
                lyr: [uniform_head] * H for lyr in range(L)
            }
            return result

        # 2. Layer weights (alpha interpolation)
        alpha_l = cfg.layer_weight_alpha
        layer_weights = [(m / total_mass) ** alpha_l for m in layer_mass]
        w_sum = sum(layer_weights) or 1.0
        layer_weights = [w / w_sum for w in layer_weights]

        # 3. Reserve minimums, distribute the rest proportionally
        min_reserve = cfg.min_layer_budget * L
        pool = max(0, total_budget - min_reserve)
        layer_budgets = [
            cfg.min_layer_budget + int(round(w * pool)) for w in layer_weights
        ]

        # 4. Per-layer: split across heads
        result: dict[int, list[int]] = {}
        alpha_h = cfg.head_weight_alpha
        for lyr in range(L):
            lyr_budget = layer_budgets[lyr]
            head_masses = head_mass_per_layer[lyr]
            head_total = sum(head_masses)

            if head_total <= 0.0:
                uniform = max(cfg.min_head_budget, lyr_budget // max(H, 1))
                head_budgets = [uniform] * H
            else:
                head_weights = [(m / head_total) ** alpha_h for m in head_masses]
                hw_sum = sum(head_weights) or 1.0
                head_weights = [w / hw_sum for w in head_weights]
                head_reserve = cfg.min_head_budget * H
                head_pool = max(0, lyr_budget - head_reserve)
                head_budgets = [
                    cfg.min_head_budget + int(round(w * head_pool))
                    for w in head_weights
                ]

            result[lyr] = head_budgets

        self._stats.layer_budget_distribution = {
            lyr: sum(result[lyr]) for lyr in range(L)
        }
        self._stats.head_budget_distribution_per_layer = dict(result)
        return result

    # --- selection ---------------------------------------------------------

    def select_tokens_to_keep(
        self,
        layer_idx: int,
        head_idx: int,
        attention_scores,
        budget: int,
    ) -> list[int]:
        """Select token indices to keep for one (layer, head) under a budget.

        Always keeps sink_size earliest + window_size most recent tokens; the
        remaining budget is filled with the highest information-loss tokens.
        """
        import mlx.core as mx

        if attention_scores.ndim > 1:
            attention_scores = mx.mean(attention_scores, axis=0)
        seq_len = attention_scores.shape[0]

        if seq_len <= budget:
            return list(range(seq_len))

        cfg = self.config
        sink = set(range(min(cfg.sink_size, seq_len)))
        recent_start = max(0, seq_len - cfg.window_size)
        recent = set(range(recent_start, seq_len))
        protected = sink | recent

        remaining = budget - len(protected)
        if remaining <= 0:
            keep = sorted(protected)[:budget]
            self._stats.total_evictions += seq_len - len(keep)
            return keep

        # Use stored info loss if we have it, else fall back to attention scores.
        key = (layer_idx, head_idx)
        if key in self._info_loss:
            scores = self._info_loss[key]
            if scores.shape[0] < seq_len:
                scores = mx.pad(scores, [(0, seq_len - scores.shape[0])])
            elif scores.shape[0] > seq_len:
                scores = scores[:seq_len]
        else:
            scores = attention_scores

        mask = mx.ones(seq_len)
        for idx in protected:
            mask = mask.at[idx].add(-1.0)
        masked = scores * mask

        top = mx.argpartition(masked, kth=-remaining)[-remaining:]
        mx.eval(top)
        hh = set(top.tolist())

        keep = sorted(protected | hh)[:budget]
        self._stats.total_evictions += seq_len - len(keep)
        return keep

    def stats(self) -> LAVaStats:
        """Return current LAVa statistics snapshot."""
        return self._stats


def format_lava_report(manager: LAVaManager) -> str:
    """Format a human-readable summary of a LAVaManager's state."""
    cfg = manager.config
    stats = manager.stats()

    lines = [
        "LAVa Unified KV Eviction Report",
        "=" * 70,
        f"  Layers: {manager.num_layers}, Heads/layer: {manager.num_heads}",
        f"  Budget ratio: {cfg.total_budget_ratio:.2f}",
        f"  Layer alpha: {cfg.layer_weight_alpha}, Head alpha: {cfg.head_weight_alpha}",
        f"  Min layer budget: {cfg.min_layer_budget}, Min head budget: {cfg.min_head_budget}",
        f"  Sink: {cfg.sink_size}, Window: {cfg.window_size}",
        "",
        f"  Total evictions:        {stats.total_evictions:,}",
        f"  Aggregate info loss:    {stats.information_loss_score:.4f}",
        "",
    ]

    if stats.layer_budget_distribution:
        lines.append("  Layer budget distribution:")
        lines.append(f"  {'Layer':>6}  {'Budget':>10}  {'Heads':>30}")
        lines.append(f"  {'-'*6}  {'-'*10}  {'-'*30}")
        sample_layers = sorted(stats.layer_budget_distribution.keys())
        # Show up to 8 representative layers to keep output compact.
        if len(sample_layers) > 8:
            step = max(1, len(sample_layers) // 8)
            sample_layers = sample_layers[::step]
        for lyr in sample_layers:
            total = stats.layer_budget_distribution[lyr]
            heads = stats.head_budget_distribution_per_layer.get(lyr, [])
            if len(heads) > 6:
                head_repr = (
                    "[" + ", ".join(str(h) for h in heads[:3])
                    + ", ..., " + ", ".join(str(h) for h in heads[-2:]) + "]"
                )
            else:
                head_repr = str(heads)
            lines.append(f"  {lyr:>6}  {total:>10}  {head_repr:>30}")
    else:
        lines.append("  (no budgets allocated yet — call allocate_budgets())")

    return "\n".join(lines)
