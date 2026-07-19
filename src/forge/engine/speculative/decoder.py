from __future__ import annotations

from dataclasses import dataclass, field

from .adaptive_k import AdaptiveKController
from .ngram import NGramDrafter

# ===========================================================================
# N-gram Self-Speculative Decode Loop (Phase 7.1 — real integration)
#
# A greedy, exactness-preserving self-speculative decode loop that ties the
# NGramDrafter (draft) and AdaptiveKController (dynamic draft length) into an
# actual token-generation loop. It is deliberately decoupled from MLXEngine:
# the caller supplies a `forward_fn(tokens) -> logits_rows` returning the
# next-token logits for every position of `tokens` (one transformer forward),
# so the loop is unit-testable with a deterministic mock and reused verbatim
# by the real engine. Verification is greedy (argmax), which makes speculation
# output-identical to plain greedy decoding while accepting matched drafts.
# ===========================================================================


def _argmax_row(row) -> int:
    """Argmax over a single logits row (MLX array or plain sequence)."""
    if hasattr(row, "shape"):
        try:
            import mlx.core as mx
            return int(mx.argmax(row).item())
        except Exception:
            pass
    return int(max(range(len(row)), key=lambda j: row[j]))


@dataclass
class NGramSpecResult:
    """Result of a self-speculative n-gram decode run."""

    tokens: list[int] = field(default_factory=list)
    steps: int = 0
    total_drafted: int = 0
    total_accepted: int = 0
    final_k: int = 0
    stopped_on_eos: bool = False

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / max(self.total_drafted, 1)

    @property
    def avg_tokens_per_step(self) -> float:
        return len(self.tokens) / max(self.steps, 1)


class NGramSpeculativeDecoder:
    """Greedy self-speculative decode loop (n-gram draft + adaptive K).

    Args:
        ngram: NGramDrafter used to propose draft tokens from history.
        adaptive_k: Controller for dynamic draft length (defaults created lazily).
        use_adaptive_k: Whether `adaptive_k` drives the draft length. When
            False, a fixed draft length `fixed_k` is used.
        fixed_k: Draft length used when adaptive K is disabled.
    """

    def __init__(
        self,
        ngram: NGramDrafter | None = None,
        adaptive_k: AdaptiveKController | None = None,
        use_adaptive_k: bool = True,
        fixed_k: int = 3,
    ):
        self.ngram = ngram or NGramDrafter()
        self.use_adaptive_k = use_adaptive_k
        self.adaptive_k = adaptive_k or AdaptiveKController()
        self.fixed_k = fixed_k

    def _draft_len(self) -> int:
        return self.adaptive_k.current_k if self.use_adaptive_k else self.fixed_k

    def generate(
        self,
        prompt_tokens: list[int],
        forward_fn,
        max_tokens: int = 128,
        eos_token_id: int | None = None,
    ) -> NGramSpecResult:
        """Run the self-speculative greedy decode loop.

        Args:
            prompt_tokens: Prompt token ids (context seed).
            forward_fn: Callable(tokens: list[int]) -> logits_rows, where
                logits_rows[j] are the next-token logits after position j
                (one transformer forward over `tokens`). MLX arrays and plain
                nested sequences are both supported.
            max_tokens: Maximum number of tokens to generate.
            eos_token_id: Optional stop token.

        Returns:
            NGramSpecResult with generated tokens and speculation statistics.
        """
        result = NGramSpecResult()
        tokens = list(prompt_tokens)
        for t in tokens:
            self.ngram.observe(t)

        if not tokens:
            return result  # nothing to condition on

        while len(result.tokens) < max_tokens:
            k = max(1, self._draft_len())
            drafts = self.ngram.draft(tokens)[:k]

            logits_rows = forward_fn(tokens + drafts)
            base = len(tokens) - 1

            # Greedy verification of the drafted prefix.
            accepted: list[int] = []
            for i, d in enumerate(drafts):
                if _argmax_row(logits_rows[base + i]) == d:
                    accepted.append(d)
                else:
                    break

            # Correction / bonus token from the target at the first unaccepted
            # position — this guarantees at least one token of progress/step.
            bonus = _argmax_row(logits_rows[base + len(accepted)])
            new_tokens = accepted + [bonus]

            result.steps += 1
            result.total_drafted += len(drafts)
            result.total_accepted += len(accepted)
            if self.use_adaptive_k and drafts:
                self.adaptive_k.record_round(len(drafts), len(accepted))

            stop = False
            for tok in new_tokens:
                if len(result.tokens) >= max_tokens:
                    break
                self.ngram.observe(tok)
                tokens.append(tok)
                result.tokens.append(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    stop = True
                    result.stopped_on_eos = True
                    break
            if stop:
                break

        result.final_k = self._draft_len()
        return result
