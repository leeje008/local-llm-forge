from __future__ import annotations

# ===========================================================================
# N-gram Self-Speculation (Phase 7.1)
#
# Uses token generation history to predict future tokens via N-gram matching.
# No separate draft model needed — zero additional memory cost.
# Inspired by llama.cpp PR #18471 and lookup decoding approaches.
# ===========================================================================

class NGramDrafter:
    """N-gram based self-speculative drafter.

    Maintains a hash table of N-gram patterns observed during generation.
    When a pattern matches recent tokens, its continuation is used as
    the draft sequence. Falls back to greedy/random sampling if no match.

    Key advantage: Zero memory overhead — no separate draft model required.
    Effective for repetitive/structured text (code, templates, lists).
    """

    def __init__(self, n: int = 3, max_draft: int = 5, table_size: int = 65536):
        """
        Args:
            n: N-gram order (context window for matching).
            max_draft: Maximum number of draft tokens to produce.
            table_size: Maximum entries in the N-gram table.
        """
        self.n = n
        self.max_draft = max_draft
        self.table_size = table_size
        # Key: tuple of N tokens → Value: list of (next_token, count) pairs
        self._table: dict[tuple[int, ...], dict[int, int]] = {}
        self._history: list[int] = []

    def reset(self):
        """Clear all learned N-gram patterns."""
        self._table.clear()
        self._history.clear()

    def observe(self, token_id: int):
        """Observe a generated token and update N-gram statistics."""
        self._history.append(token_id)

        # Update N-gram table with all valid N-gram windows
        for order in range(1, self.n + 1):
            if len(self._history) > order:
                key = tuple(self._history[-(order + 1):-1])
                if key not in self._table:
                    if len(self._table) >= self.table_size:
                        continue  # Table full, skip
                    self._table[key] = {}
                continuation = self._table[key]
                continuation[token_id] = continuation.get(token_id, 0) + 1

    def draft(self, context_tokens: list[int] | None = None) -> list[int]:
        """Generate draft tokens based on N-gram matching.

        Args:
            context_tokens: Recent context (uses internal history if None).

        Returns:
            List of predicted next tokens (may be empty if no match).
        """
        ctx = context_tokens if context_tokens is not None else self._history
        if len(ctx) < 1:
            return []

        draft_tokens = []
        current = list(ctx)

        for _ in range(self.max_draft):
            best_token = None
            best_score = 0

            # Try longest N-gram first (highest specificity), fall back to shorter
            for order in range(min(self.n, len(current)), 0, -1):
                key = tuple(current[-order:])
                if key in self._table:
                    continuations = self._table[key]
                    if continuations:
                        # Pick most frequent continuation
                        candidate = max(continuations, key=continuations.get)
                        score = continuations[candidate] * order  # Weight by order
                        if score > best_score:
                            best_token = candidate
                            best_score = score

            if best_token is None:
                break  # No match found

            draft_tokens.append(best_token)
            current.append(best_token)

        return draft_tokens

    def get_stats(self) -> dict:
        """Get N-gram table statistics."""
        total_entries = sum(len(v) for v in self._table.values())
        return {
            "ngram_order": self.n,
            "table_keys": len(self._table),
            "total_entries": total_entries,
            "history_len": len(self._history),
        }
