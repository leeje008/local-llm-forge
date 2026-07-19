"""Ada-KV per-head budget (NeurIPS 2025): adaptive per-head eviction budgets."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Ada-KV: Per-Head Adaptive Budget Allocation (NeurIPS 2025, 2407.11550)
#
# Different attention heads need different cache sizes.
# High-entropy heads (spread attention) need more tokens cached.
# Low-entropy heads (focused attention) need fewer tokens.
# ---------------------------------------------------------------------------

@dataclass
class AdaKVConfig:
    """Configuration for Ada-KV per-head adaptive budget."""

    total_budget_ratio: float = 0.2   # Total KV budget as fraction of seq_len
    num_sink_tokens: int = 4
    recent_window: int = 128
    min_head_budget_ratio: float = 0.05  # Minimum budget per head (fraction of total)
    entropy_smoothing: float = 0.1       # Smoothing for entropy-based allocation


class AdaKVManager:
    """Ada-KV: Adaptive per-head KV cache budget allocation.

    Extends H2O by giving each attention head a different eviction budget
    based on its attention entropy. High-entropy heads (broad attention patterns)
    get more cache budget; low-entropy heads (focused on few tokens) get less.

    This prevents uniform eviction from destroying heads that genuinely need
    wide context while saving memory on heads that only attend to a few tokens.
    """

    def __init__(self, config: AdaKVConfig | None = None):
        self.config = config or AdaKVConfig()
        # layer → (num_heads, seq_len) cumulative scores
        self._per_head_scores: dict[int, object] = {}
        self._head_entropies: dict[int, object] = {}  # layer → (num_heads,)

    def reset(self):
        self._per_head_scores.clear()
        self._head_entropies.clear()

    def update_scores(self, layer_idx: int, attention_scores):
        """Update per-head attention scores.

        Args:
            attention_scores: Shape (num_heads, seq_len) — per-head attention
                received by each token in this generation step.
        """
        import mlx.core as mx

        if attention_scores.ndim == 1:
            attention_scores = attention_scores[None, :]

        num_heads, seq_len = attention_scores.shape

        # Update cumulative per-head scores
        if layer_idx in self._per_head_scores:
            old = self._per_head_scores[layer_idx]
            if old.shape[1] < seq_len:
                old = mx.pad(old, [(0, 0), (0, seq_len - old.shape[1])])
            elif old.shape[1] > seq_len:
                old = old[:, :seq_len]
            self._per_head_scores[layer_idx] = old * 0.95 + attention_scores
        else:
            self._per_head_scores[layer_idx] = attention_scores

        # Compute per-head entropy (how spread out attention is)
        # Higher entropy → head needs more tokens in cache
        probs = attention_scores / (mx.sum(attention_scores, axis=1, keepdims=True) + 1e-10)
        entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=1)  # (num_heads,)

        smooth = self.config.entropy_smoothing
        if layer_idx in self._head_entropies:
            self._head_entropies[layer_idx] = (
                (1 - smooth) * self._head_entropies[layer_idx] + smooth * entropy
            )
        else:
            self._head_entropies[layer_idx] = entropy

        mx.eval(self._per_head_scores[layer_idx], self._head_entropies[layer_idx])

    def compute_head_budgets(self, layer_idx: int, num_heads: int, seq_len: int) -> list[int]:
        """Compute per-head KV cache budgets based on attention entropy.

        Returns list of budget (number of tokens to keep) per head.
        """
        import mlx.core as mx

        total_budget = max(
            (self.config.num_sink_tokens + self.config.recent_window + 1) * num_heads,
            int(seq_len * self.config.total_budget_ratio * num_heads),
        )
        min_per_head = max(
            self.config.num_sink_tokens + self.config.recent_window + 1,
            int(seq_len * self.config.min_head_budget_ratio),
        )

        if layer_idx not in self._head_entropies:
            # No data yet — uniform allocation
            uniform = total_budget // num_heads
            return [max(min_per_head, uniform)] * num_heads

        entropies = self._head_entropies[layer_idx]
        mx.eval(entropies)
        ent_list = entropies.tolist()

        # Normalize entropies to get allocation weights
        total_ent = sum(ent_list) + 1e-10
        weights = [e / total_ent for e in ent_list]

        # Distribute budget proportionally, enforcing minimum
        budgets = []
        remaining = total_budget
        for w in weights:
            b = max(min_per_head, int(w * total_budget))
            b = min(b, seq_len)  # Can't keep more than seq_len
            budgets.append(b)
            remaining -= b

        # Redistribute any remaining budget to highest-entropy heads
        if remaining > 0:
            sorted_heads = sorted(range(num_heads), key=lambda i: ent_list[i], reverse=True)
            for i in sorted_heads:
                add = min(remaining, seq_len - budgets[i])
                budgets[i] += add
                remaining -= add
                if remaining <= 0:
                    break

        return budgets

    def select_tokens_per_head(
        self, layer_idx: int, num_heads: int, seq_len: int,
    ) -> list[list[int]]:
        """Select which tokens to keep for each head independently.

        Returns list of (sorted token indices) per head.
        """
        import mlx.core as mx

        budgets = self.compute_head_budgets(layer_idx, num_heads, seq_len)

        if layer_idx not in self._per_head_scores:
            # No data — keep everything or uniform selection
            return [list(range(min(b, seq_len))) for b in budgets]

        scores = self._per_head_scores[layer_idx]  # (num_heads, seq_len)
        result = []

        for h in range(num_heads):
            budget = budgets[h]
            if seq_len <= budget:
                result.append(list(range(seq_len)))
                continue

            # Protected: sink + recent
            sink = set(range(min(self.config.num_sink_tokens, seq_len)))
            recent_start = max(0, seq_len - self.config.recent_window)
            recent = set(range(recent_start, seq_len))
            protected = sink | recent

            hh_budget = budget - len(protected)
            if hh_budget > 0:
                head_scores = scores[h, :seq_len]
                mask = mx.ones(seq_len)
                for idx in protected:
                    mask = mask.at[idx].add(-1.0)
                masked = head_scores * mask
                top_idx = mx.argpartition(masked, kth=-hh_budget)[-hh_budget:]
                mx.eval(top_idx)
                hh = set(top_idx.tolist())
            else:
                hh = set()

            keep = sorted(protected | hh)[:budget]
            result.append(keep)

        return result
