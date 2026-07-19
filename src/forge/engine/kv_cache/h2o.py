"""H2O token eviction (2306.14048): Heavy-Hitter Oracle, keep top-20% tokens."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# H2O: Heavy-Hitter Oracle Token Eviction (2306.14048)
#
# Key insight: ~20% of tokens accumulate ~80% of attention weight.
# Keep "attention sink" tokens (first few) + heavy-hitters + recent window.
# ---------------------------------------------------------------------------

@dataclass
class H2OConfig:
    """Configuration for H2O KV cache eviction."""

    budget_ratio: float = 0.2        # Keep top 20% of tokens
    num_sink_tokens: int = 4         # Always keep first N tokens (attention sinks)
    recent_window: int = 128         # Always keep last N tokens (local context)
    score_decay: float = 0.95        # Exponential decay for historical scores


class H2OEvictionManager:
    """Heavy-Hitter Oracle KV cache eviction manager.

    Tracks cumulative attention scores across generation steps and evicts
    tokens that receive the least attention. Always preserves:
    - Attention sink tokens (first few tokens)
    - Recent window tokens (last N tokens)
    - Heavy-hitter tokens (top-K by cumulative attention)
    """

    def __init__(self, config: H2OConfig | None = None):
        self.config = config or H2OConfig()
        self._cumulative_scores: dict[int, object] = {}  # layer → scores array

    def reset(self):
        """Reset all tracked scores (call on new conversation)."""
        self._cumulative_scores.clear()

    def update_scores(self, layer_idx: int, attention_scores):
        """Update cumulative attention scores for a layer.

        Args:
            layer_idx: Transformer layer index.
            attention_scores: Shape (num_heads, seq_len) — mean attention
                received by each token across all query positions in this step.
        """
        import mlx.core as mx

        # Average across heads → (seq_len,)
        if attention_scores.ndim > 1:
            token_scores = mx.mean(attention_scores, axis=0)
        else:
            token_scores = attention_scores

        decay = self.config.score_decay
        if layer_idx in self._cumulative_scores:
            old = self._cumulative_scores[layer_idx]
            seq_len = token_scores.shape[0]
            if old.shape[0] < seq_len:
                # Extend with zeros for new tokens
                old = mx.pad(old, [(0, seq_len - old.shape[0])])
            elif old.shape[0] > seq_len:
                old = old[:seq_len]
            self._cumulative_scores[layer_idx] = old * decay + token_scores
        else:
            self._cumulative_scores[layer_idx] = token_scores

        mx.eval(self._cumulative_scores[layer_idx])

    def select_tokens_to_keep(self, layer_idx: int, seq_len: int) -> list[int]:
        """Determine which token positions to keep in the KV cache.

        Returns sorted list of token indices to retain.
        """
        import mlx.core as mx

        budget = max(
            self.config.num_sink_tokens + self.config.recent_window + 1,
            int(seq_len * self.config.budget_ratio),
        )

        if seq_len <= budget:
            return list(range(seq_len))

        # Always keep: sink tokens + recent window
        sink = set(range(min(self.config.num_sink_tokens, seq_len)))
        recent_start = max(0, seq_len - self.config.recent_window)
        recent = set(range(recent_start, seq_len))
        protected = sink | recent

        # Remaining budget for heavy-hitters
        hh_budget = budget - len(protected)

        if hh_budget > 0 and layer_idx in self._cumulative_scores:
            scores = self._cumulative_scores[layer_idx]
            if scores.shape[0] < seq_len:
                scores = mx.pad(scores, [(0, seq_len - scores.shape[0])])

            # Mask protected tokens so they don't compete for hh_budget
            mask = mx.ones(seq_len)
            for idx in protected:
                mask = mask.at[idx].add(-1.0)  # set to 0
            masked_scores = scores[:seq_len] * mask

            # Top-K heavy hitters from unprotected tokens
            top_indices = mx.argpartition(masked_scores, kth=-hh_budget)[-hh_budget:]
            mx.eval(top_indices)
            hh = set(top_indices.tolist())
        else:
            hh = set()

        keep = sorted(protected | hh)
        return keep[:budget]

    def get_eviction_stats(self, layer_idx: int, seq_len: int) -> dict:
        """Get eviction statistics for reporting."""
        keep = self.select_tokens_to_keep(layer_idx, seq_len)
        return {
            "seq_len": seq_len,
            "kept": len(keep),
            "evicted": seq_len - len(keep),
            "eviction_pct": (seq_len - len(keep)) / max(seq_len, 1) * 100,
            "budget_ratio": self.config.budget_ratio,
        }
