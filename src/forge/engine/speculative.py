"""Speculative decoding support — draft model selection, N-gram self-speculation,
PEARL scheduling, tree attention verification, and adaptive draft length.

Phase 7 enhancements:
- N-gram self-speculation: Draft from token history, no separate model needed
- PEARL scheduling (ICLR 2025): Pre-verify + post-verify overlap, +50% on any SD
- Tree attention: Branch candidates instead of chain, higher acceptance rates
- Adaptive K: Dynamic draft length based on observed acceptance rate
- OmniDraft: Single 68M universal drafter for all target architectures
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Known draft model mappings: architecture → small model
DRAFT_MODELS: dict[str, list[str]] = {
    "llama": [
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "meta-llama/Llama-3.2-1B",
    ],
    "qwen2": [
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ],
    "qwen3": [
        "mlx-community/Qwen3-0.6B-4bit",
        "Qwen/Qwen3-0.6B",
    ],
    "mistral": [
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    ],
    "gemma": [
        "mlx-community/gemma-2-2b-it-4bit",
    ],
    "phi": [
        "mlx-community/Phi-3.5-mini-instruct-4bit",
    ],
}

# Universal fallback
FALLBACK_DRAFTS = [
    "mlx-community/SmolLM2-360M-Instruct-4bit",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]


@dataclass
class DraftModelInfo:
    """Information about a selected draft model."""

    model_id: str
    estimated_size_gb: float
    architecture_match: bool
    source: str  # "architecture_match" | "fallback"


def select_draft_model(
    target_architecture: str,
    available_memory_gb: float = 2.0,
    prefer_mlx_community: bool = True,
) -> DraftModelInfo | None:
    """Select the best draft model for speculative decoding.

    Prioritizes mlx-community quantized models (already in MLX format).
    """
    arch = target_architecture.lower()

    # Try architecture-matched models first
    for key, candidates in DRAFT_MODELS.items():
        if key in arch:
            for candidate in candidates:
                is_mlx = "mlx-community" in candidate
                if prefer_mlx_community and not is_mlx:
                    continue
                # Rough size estimate: mlx-community 4-bit models are tiny
                est_size = 0.3 if "0.5B" in candidate or "360M" in candidate else 0.8
                if est_size < available_memory_gb:
                    return DraftModelInfo(
                        model_id=candidate,
                        estimated_size_gb=est_size,
                        architecture_match=True,
                        source="architecture_match",
                    )
            # Try non-mlx versions
            if prefer_mlx_community:
                for candidate in candidates:
                    if "mlx-community" not in candidate:
                        est_size = 1.0
                        if est_size < available_memory_gb:
                            return DraftModelInfo(
                                model_id=candidate,
                                estimated_size_gb=est_size,
                                architecture_match=True,
                                source="architecture_match",
                            )

    # Fallback to universal small model
    for fallback in FALLBACK_DRAFTS:
        return DraftModelInfo(
            model_id=fallback,
            estimated_size_gb=0.3,
            architecture_match=False,
            source="fallback",
        )

    return None


def estimate_speedup(
    target_params_b: float,
    draft_params_b: float = 0.5,
    acceptance_rate: float = 0.7,
    num_draft_tokens: int = 3,
) -> float:
    """Estimate speculative decoding speedup factor.

    Based on the formula: speedup ≈ 1 / (1 - acceptance_rate + acceptance_rate/num_draft_tokens)
    adjusted for the overhead of running the draft model.
    """
    if acceptance_rate <= 0:
        return 1.0

    # Theoretical speedup from speculation
    theoretical = 1.0 / (1.0 - acceptance_rate + acceptance_rate / num_draft_tokens)

    # Overhead factor: draft model adds verification cost
    # Larger ratio of draft/target → less overhead
    overhead = 1.0 - (draft_params_b / target_params_b) * 0.1

    return theoretical * overhead


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


# ===========================================================================
# PEARL Scheduling (Phase 7.2) — ICLR 2025, arXiv 2408.11850
#
# Two key optimizations that overlap draft/verify phases:
# 1. Pre-verify: Verify the first draft token during drafting (parallel)
# 2. Post-verify: Generate more draft tokens during target verification
#
# Result: +50% speedup on top of any existing speculative decoding setup.
# ===========================================================================

@dataclass
class PEARLConfig:
    """Configuration for PEARL scheduling."""

    enable_pre_verify: bool = True   # Verify first draft token during drafting
    enable_post_verify: bool = True  # Generate extra drafts during verification
    post_verify_tokens: int = 2      # Extra draft tokens during post-verify phase


@dataclass
class PEARLStats:
    """Runtime statistics for PEARL scheduling."""

    total_steps: int = 0
    pre_verify_hits: int = 0          # First draft accepted without full verify
    post_verify_extra_accepted: int = 0  # Extra tokens from post-verify
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted_tokens / max(self.total_draft_tokens, 1)

    @property
    def pre_verify_hit_rate(self) -> float:
        return self.pre_verify_hits / max(self.total_steps, 1)


class PEARLScheduler:
    """PEARL: Parallel speculative decoding with adaptive draft length.

    Orchestrates the overlap between draft generation and target verification:

    Standard SD:  [--draft--][--verify--][--draft--][--verify--]
    PEARL:        [--draft--]                        (pre-verify first token)
                      [--verify + post-draft--]      (overlap)
                                    [--draft--]      (next round)

    The scheduler manages the state machine for pre-verify and post-verify phases.
    """

    def __init__(self, config: PEARLConfig | None = None):
        self.config = config or PEARLConfig()
        self.stats = PEARLStats()

    def reset(self):
        self.stats = PEARLStats()

    def plan_draft_round(
        self,
        base_draft_len: int,
        recent_acceptance_rate: float,
    ) -> DraftPlan:
        """Plan the next draft round with PEARL optimizations.

        Args:
            base_draft_len: Base number of draft tokens (from adaptive K).
            recent_acceptance_rate: Recent acceptance rate (for adjustment).

        Returns:
            DraftPlan with token counts for each phase.
        """
        self.stats.total_steps += 1

        # Pre-verify: if enabled, we'll verify the first draft token
        # concurrently with generating tokens 2..K
        pre_verify = self.config.enable_pre_verify and base_draft_len > 1

        # Post-verify: during target model verification, generate extra drafts
        post_extra = 0
        if self.config.enable_post_verify and recent_acceptance_rate > 0.5:
            # Only post-verify when acceptance is reasonable
            post_extra = self.config.post_verify_tokens

        return DraftPlan(
            main_draft_len=base_draft_len,
            pre_verify_first=pre_verify,
            post_verify_extra=post_extra,
            total_candidates=base_draft_len + post_extra,
        )

    def record_result(
        self,
        drafted: int,
        accepted: int,
        pre_verify_accepted: bool = False,
        post_verify_accepted: int = 0,
    ):
        """Record the result of a speculative decoding round."""
        self.stats.total_draft_tokens += drafted
        self.stats.total_accepted_tokens += accepted
        if pre_verify_accepted:
            self.stats.pre_verify_hits += 1
        self.stats.post_verify_extra_accepted += post_verify_accepted


@dataclass
class DraftPlan:
    """Plan for a single speculative decoding round."""

    main_draft_len: int          # Number of main draft tokens
    pre_verify_first: bool       # Whether to pre-verify the first token
    post_verify_extra: int       # Extra tokens to draft during verification
    total_candidates: int        # Total candidate tokens


# ===========================================================================
# Tree Attention Verification (Phase 7.3) — Sequoia (2402.12374)
#
# Instead of a linear chain of draft tokens, generate a tree of candidates.
# The target model verifies the entire tree in one forward pass using a
# tree attention mask. This increases effective acceptance rate because
# if one branch fails, another may succeed.
# ===========================================================================

@dataclass
class TreeNode:
    """Node in a speculative decoding tree."""

    token_id: int
    depth: int
    parent_idx: int           # Index of parent in flat node list (-1 for root)
    children_idx: list[int] = field(default_factory=list)
    log_prob: float = 0.0     # Log probability from draft model
    accepted: bool = False


@dataclass
class TreeConfig:
    """Configuration for tree-based speculative decoding."""

    max_depth: int = 5        # Maximum tree depth
    max_branches: int = 3     # Maximum branches per node (top-k from draft)
    max_nodes: int = 32       # Maximum total nodes in tree
    temperature: float = 1.0  # Temperature for draft sampling


class TreeDrafter:
    """Tree-based speculative decoding draft generator.

    Generates a tree of candidate continuations instead of a single chain.
    The target model verifies all paths in one forward pass using a tree
    attention mask, then selects the longest accepted path.

    Tree structure example (depth=3, branches=2):
        root
        ├── "the" (0.8)
        │   ├── "cat" (0.6) → "sat" (0.4)
        │   └── "dog" (0.3) → "ran" (0.5)
        └── "a" (0.2)
            ├── "big" (0.5) → "red" (0.3)
            └── "small" (0.4)
    """

    def __init__(self, config: TreeConfig | None = None):
        self.config = config or TreeConfig()

    def build_tree(self, draft_logits_fn, context_tokens: list[int]) -> list[TreeNode]:
        """Build a speculation tree by repeatedly calling the draft model.

        Args:
            draft_logits_fn: Callable(token_ids: list[int]) → logits array
                A function that takes a token sequence and returns next-token logits.
            context_tokens: The tokens preceding the tree root.

        Returns:
            Flat list of TreeNode. Index 0 is a virtual root (no token).
        """
        import mlx.core as mx

        cfg = self.config
        nodes: list[TreeNode] = []

        # Virtual root node
        root = TreeNode(token_id=-1, depth=0, parent_idx=-1)
        nodes.append(root)

        # BFS to build tree level by level
        frontier = [0]  # Indices of nodes to expand

        while frontier and len(nodes) < cfg.max_nodes:
            next_frontier = []
            for parent_idx in frontier:
                parent = nodes[parent_idx]
                if parent.depth >= cfg.max_depth:
                    continue

                # Get the token path from root to this node
                path = self._get_path_tokens(nodes, parent_idx)
                full_context = context_tokens + path

                if not full_context:
                    continue

                try:
                    logits = draft_logits_fn(full_context)
                    if logits is None:
                        continue

                    # Top-k branches
                    k = min(cfg.max_branches, cfg.max_nodes - len(nodes))
                    if k <= 0:
                        break

                    if hasattr(logits, 'shape'):
                        # MLX array
                        top_k_idx = mx.argpartition(logits, kth=-k)[-k:]
                        mx.eval(top_k_idx)
                        top_k_list = top_k_idx.tolist()
                        log_probs = mx.log(mx.softmax(logits / cfg.temperature))
                        mx.eval(log_probs)
                    else:
                        top_k_list = list(range(min(k, len(logits))))
                        log_probs = None

                    for token_id in top_k_list:
                        if len(nodes) >= cfg.max_nodes:
                            break
                        lp = float(log_probs[token_id]) if log_probs is not None else 0.0
                        child = TreeNode(
                            token_id=token_id,
                            depth=parent.depth + 1,
                            parent_idx=parent_idx,
                            log_prob=lp,
                        )
                        child_idx = len(nodes)
                        nodes.append(child)
                        parent.children_idx.append(child_idx)
                        next_frontier.append(child_idx)

                except Exception:
                    continue

            frontier = next_frontier

        return nodes

    def build_attention_mask(self, nodes: list[TreeNode]) -> list[list[bool]]:
        """Build the tree attention mask for verification.

        Each candidate token can only attend to its ancestors in the tree
        (not to tokens in other branches). This is a causal mask shaped
        by the tree structure.

        Returns:
            2D boolean mask (num_nodes x num_nodes). True = can attend.
        """
        n = len(nodes)
        mask = [[False] * n for _ in range(n)]

        for i in range(n):
            # Each node attends to itself and all ancestors
            j = i
            while j >= 0:
                mask[i][j] = True
                j = nodes[j].parent_idx

        return mask

    def select_longest_accepted_path(self, nodes: list[TreeNode]) -> list[int]:
        """After verification, find the longest path of accepted tokens.

        Returns:
            List of token IDs along the longest accepted path (root to leaf).
        """
        best_path: list[int] = []

        def dfs(idx: int, current_path: list[int]):
            nonlocal best_path
            node = nodes[idx]

            if idx > 0:  # Skip virtual root
                if not node.accepted:
                    return
                current_path = current_path + [node.token_id]

            if len(current_path) > len(best_path):
                best_path = current_path

            for child_idx in node.children_idx:
                dfs(child_idx, current_path)

        dfs(0, [])
        return best_path

    def _get_path_tokens(self, nodes: list[TreeNode], node_idx: int) -> list[int]:
        """Get the token path from root to the given node."""
        path = []
        idx = node_idx
        while idx > 0:  # Stop before virtual root
            path.append(nodes[idx].token_id)
            idx = nodes[idx].parent_idx
        path.reverse()
        return path

    def get_tree_stats(self, nodes: list[TreeNode]) -> dict:
        """Get statistics about a built tree."""
        if not nodes:
            return {"num_nodes": 0}
        depths = [n.depth for n in nodes if n.depth > 0]
        return {
            "num_nodes": len(nodes) - 1,  # Exclude virtual root
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "num_leaves": sum(1 for n in nodes if not n.children_idx and n.depth > 0),
            "branching_factor": (len(nodes) - 1) / max(
                sum(1 for n in nodes if n.children_idx), 1
            ),
        }


# ===========================================================================
# Adaptive K — Dynamic Draft Length (Phase 7.3)
#
# Automatically adjusts the number of draft tokens based on observed
# acceptance rate. High acceptance → more drafts. Low → fewer.
# ===========================================================================

@dataclass
class AdaptiveKConfig:
    """Configuration for adaptive draft length."""

    initial_k: int = 3             # Starting draft length
    min_k: int = 1                 # Minimum draft length
    max_k: int = 10                # Maximum draft length
    target_acceptance: float = 0.7 # Target acceptance rate
    increase_threshold: float = 0.8 # Increase K if acceptance > this
    decrease_threshold: float = 0.4 # Decrease K if acceptance < this
    window_size: int = 10           # Rolling window for acceptance tracking
    step_size: int = 1              # K adjustment step


class AdaptiveKController:
    """Dynamically adjusts speculative decoding draft length.

    Tracks rolling acceptance rate and increases K when the draft model
    is performing well (high acceptance), decreases when it's struggling.

    This prevents wasting compute on long drafts that will be rejected,
    while maximizing throughput when the draft model is accurate.
    """

    def __init__(self, config: AdaptiveKConfig | None = None):
        self.config = config or AdaptiveKConfig()
        self._current_k = self.config.initial_k
        self._history: list[float] = []  # Recent acceptance rates

    @property
    def current_k(self) -> int:
        return self._current_k

    def reset(self):
        self._current_k = self.config.initial_k
        self._history.clear()

    def record_round(self, drafted: int, accepted: int):
        """Record the result of a speculative round and adjust K.

        Args:
            drafted: Number of tokens drafted.
            accepted: Number of tokens accepted by target model.
        """
        if drafted == 0:
            return

        rate = accepted / drafted
        self._history.append(rate)

        # Keep only recent window
        if len(self._history) > self.config.window_size:
            self._history = self._history[-self.config.window_size:]

        # Adjust K based on rolling average
        avg_rate = sum(self._history) / len(self._history)

        if avg_rate >= self.config.increase_threshold:
            self._current_k = min(
                self._current_k + self.config.step_size,
                self.config.max_k,
            )
        elif avg_rate <= self.config.decrease_threshold:
            self._current_k = max(
                self._current_k - self.config.step_size,
                self.config.min_k,
            )

    def get_stats(self) -> dict:
        avg_rate = sum(self._history) / max(len(self._history), 1)
        return {
            "current_k": self._current_k,
            "avg_acceptance_rate": round(avg_rate, 3),
            "window_size": len(self._history),
        }


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


# ===========================================================================
# OmniDraft — Universal Cross-Vocabulary Drafter (Phase 7.4)
# NeurIPS 2025, arXiv 2507.02659
#
# Single 68M-parameter model that works with ANY target model via
# cross-vocabulary mapping. Replaces per-architecture draft model lookup.
# ===========================================================================

# Known OmniDraft model IDs (when available on HuggingFace)
OMNIDRAFT_MODELS = [
    "mlx-community/OmniDraft-68M-4bit",  # Primary (if released for MLX)
    "omnidraft/OmniDraft-68M",            # Original
]


@dataclass
class OmniDraftInfo:
    """Information about the OmniDraft universal drafter."""

    model_id: str
    estimated_size_gb: float = 0.05   # ~50MB for 68M params 4-bit
    supports_all_architectures: bool = True
    source: str = "omnidraft"


def select_omnidraft(
    available_memory_gb: float = 0.5,
) -> OmniDraftInfo | None:
    """Select the OmniDraft universal draft model.

    OmniDraft is a single 68M-parameter model that works with any target
    model architecture via cross-vocabulary mapping + online n-gram cache.

    Returns:
        OmniDraftInfo if available, None otherwise.
    """
    for model_id in OMNIDRAFT_MODELS:
        return OmniDraftInfo(
            model_id=model_id,
            estimated_size_gb=0.05,
        )
    return None


# ===========================================================================
# Unified Speculative Decoding Strategy Selector
# ===========================================================================

class SpeculativeMethod(str):
    NONE = "none"
    DRAFT_MODEL = "draft_model"     # Traditional: separate draft model
    NGRAM = "ngram"                 # N-gram self-speculation
    OMNIDRAFT = "omnidraft"         # Universal 68M drafter
    REDRAFTER = "redrafter"         # Apple ReDrafter
    EAGLE3 = "eagle3"               # Target-attached EAGLE-3 head (Phase 7.5)
    CAS_SPEC = "cas_spec"           # Cascade Adaptive Self-Speculative (Phase 12, NeurIPS 2025)


@dataclass
class SpeculativeConfig:
    """Unified speculative decoding configuration."""

    method: str = SpeculativeMethod.NONE
    # Draft model
    draft_model_path: str | None = None
    # N-gram
    ngram_order: int = 3
    ngram_max_draft: int = 5
    # Tree
    use_tree: bool = False
    tree_branches: int = 3
    tree_depth: int = 5
    tree_max_nodes: int = 32
    # Adaptive K
    use_adaptive_k: bool = True
    initial_k: int = 3
    # PEARL
    use_pearl: bool = True
    # OmniDraft
    omnidraft_model: str | None = None
    # CAS-Spec (Phase 12) — populated lazily; forward reference to avoid
    # ordering issues since CASSpecConfig is defined below this file.
    cas_spec_config: "CASSpecConfig | None" = None


def select_best_speculative_strategy(
    target_architecture: str,
    target_params_b: float,
    available_memory_gb: float,
) -> SpeculativeConfig:
    """Auto-select the best speculative decoding strategy.

    Decision logic:
    1. If memory allows a draft model → use architecture-matched draft
    2. If target is large (30B+) → use N-gram (saves memory for the model)
    3. If OmniDraft available → universal drafter
    4. Default → N-gram (always works, zero memory)
    """
    config = SpeculativeConfig(use_adaptive_k=True, use_pearl=True)

    # N-gram is always available as baseline
    config.ngram_order = 3
    config.ngram_max_draft = 5

    # Memory check for draft model
    draft_memory_budget = min(2.0, available_memory_gb * 0.05)

    if target_params_b >= 30:
        # Large model: prefer N-gram to save memory
        config.method = SpeculativeMethod.NGRAM
        return config

    # Phase 12 Tier S: CAS-Spec cascade for mid-size models with sufficient memory.
    # Cascaded self-speculation (layer-skip + int4 fast decode + ngram) beats
    # single-tier approaches on 1-15B models when we have headroom for activation
    # quantization buffers (~16GB minimum working set including the target).
    if target_params_b < 15 and available_memory_gb >= 16:
        config.method = SpeculativeMethod.CAS_SPEC
        config.cas_spec_config = build_default_cas_spec_config(
            target_params_b=target_params_b,
            available_memory_gb=available_memory_gb,
        )
        return config

    # Try architecture-matched draft
    draft = select_draft_model(target_architecture, draft_memory_budget)
    if draft and draft.estimated_size_gb <= draft_memory_budget:
        config.method = SpeculativeMethod.DRAFT_MODEL
        config.draft_model_path = draft.model_id
        config.use_tree = True  # Use tree for better acceptance
        return config

    # Fallback to N-gram
    config.method = SpeculativeMethod.NGRAM
    return config


def format_speculative_report(config: SpeculativeConfig) -> str:
    """Format speculative decoding configuration for display."""
    lines = [
        "Speculative Decoding Configuration",
        "=" * 55,
        f"  Method:       {config.method}",
    ]

    if config.draft_model_path:
        lines.append(f"  Draft Model:  {config.draft_model_path}")
    if config.method == SpeculativeMethod.NGRAM:
        lines.append(f"  N-gram Order: {config.ngram_order}")
        lines.append(f"  Max Draft:    {config.ngram_max_draft}")
    if config.use_tree:
        lines.append(f"  Tree:         branches={config.tree_branches}, depth={config.tree_depth}")
    if config.use_adaptive_k:
        lines.append(f"  Adaptive K:   initial={config.initial_k}")
    if config.use_pearl:
        lines.append("  PEARL:        pre-verify + post-verify enabled")

    return "\n".join(lines)


# ===========================================================================
# CAS-Spec — Cascade Adaptive Self-Speculative Decoding (Phase 12, Tier S)
# NeurIPS 2025, arXiv 2510.26843
#
# Self-speculation via *cascaded* drafters at different cost/quality points:
#   Tier 0 (cheapest): n-gram lookup          — ~10ms / token, ~70% accept
#   Tier 1: layer-skip partial forward        — ~25ms / token, ~55% accept
#   Tier 2: int4 fast-path decode             — ~60ms / token, ~40% accept
#
# The Dynamic Tree Cascade (DyTC) router picks the tier whose expected
# throughput (acceptance_rate * k / latency) is highest for the current
# context, and can fuse multiple tiers into a small speculation tree so that
# a cheap tier's branches are re-expanded by a more expensive tier when they
# look promising. Tiers whose learned acceptance falls below `prune_threshold`
# are dropped from the tree.
#
# Reported gains in the paper:
#   +47% over a single-tier cascade baseline
#   1.1x - 2.3x end-to-end vs. vanilla decode
#
# NOTE: Real layer-skip and int4-fast-path drafting requires hooks inside
# MLXEngine (exposing a partial-forward / quantized-forward path). Those are
# added in `engine/mlx_engine.py` in a companion change; this file only
# defines the structural router and drafter and uses NGramDrafter as a
# placeholder for the deeper tiers until those hooks land.
# ===========================================================================


@dataclass
class CascadeLevel:
    """One tier of the CAS-Spec cascade.

    A level bundles a drafting strategy ID, its calibrated cost (latency per
    token), its learned quality (expected acceptance rate), and a cap on how
    many tokens it may draft per round. The DyTC router uses the
    latency/acceptance pair to rank levels by expected throughput.
    """

    name: str
    draft_fn_id: str              # "ngram" | "layer_skip_<N>" | "int4_decode"
    expected_latency_ms: float    # per token, calibrated
    expected_acceptance_rate: float = 0.5  # learned online via EMA
    max_draft_length: int = 4
    config_overrides: dict = field(default_factory=dict)


@dataclass
class CASSpecConfig:
    """Full cascade configuration for CAS-Spec."""

    levels: list[CascadeLevel]
    routing_strategy: str = "dytc"        # "dytc" | "fixed" | "greedy"
    acceptance_ema_alpha: float = 0.2     # online update rate
    min_samples_for_route: int = 20       # warmup: cycle through levels
    tree_branch_factor: int = 3           # DyTC tree branching per level
    tree_max_depth: int = 4
    prune_threshold: float = 0.15         # drop branches below this acceptance
    history_window: int = 64              # per-level acceptance history length


@dataclass
class CASSpecStats:
    """Runtime statistics for a CAS-Spec decoding session."""

    tokens_generated: int = 0
    level_usage_counts: dict[str, int] = field(default_factory=dict)
    level_acceptance_history: dict[str, list[float]] = field(default_factory=dict)
    total_drafts_generated: int = 0
    total_drafts_accepted: int = 0
    avg_speedup_estimate: float = 1.0

    def summary(self) -> dict:
        accept_rate = self.total_drafts_accepted / max(self.total_drafts_generated, 1)
        per_level: dict[str, dict] = {}
        for name, hist in self.level_acceptance_history.items():
            if hist:
                per_level[name] = {
                    "uses": self.level_usage_counts.get(name, 0),
                    "avg_acceptance": round(sum(hist) / len(hist), 3),
                    "last_acceptance": round(hist[-1], 3),
                    "samples": len(hist),
                }
            else:
                per_level[name] = {
                    "uses": self.level_usage_counts.get(name, 0),
                    "avg_acceptance": 0.0,
                    "last_acceptance": 0.0,
                    "samples": 0,
                }
        return {
            "tokens_generated": self.tokens_generated,
            "total_drafts_generated": self.total_drafts_generated,
            "total_drafts_accepted": self.total_drafts_accepted,
            "overall_acceptance_rate": round(accept_rate, 3),
            "avg_speedup_estimate": round(self.avg_speedup_estimate, 3),
            "per_level": per_level,
        }


class DyTCRouter:
    """Dynamic Tree Cascade router for CAS-Spec.

    Maintains per-level EMAs of acceptance rate and latency, scores levels by
    `acceptance * max_draft_length / latency_ms` (expected accepted tokens per
    ms), and exposes:

    - `route()`: pick one level for the current context. During warmup (fewer
      than `min_samples_for_route` observations across all levels) the router
      cycles through levels round-robin to build statistics. A per-context
      hint cache (keyed by `context_hash`) keeps routing stable across turns
      of the same multi-turn conversation.
    - `update()`: EMA update from a completed round.
    - `build_tree()`: build a small speculation tree rooted at `root_level`
      where promising children are re-expanded by cheaper tiers (reusing the
      existing `TreeNode` / `TreeConfig` dataclasses).
    """

    def __init__(self, config: CASSpecConfig):
        self.config = config
        # EMAs start from the calibrated defaults baked into each level.
        self._acceptance_ema: dict[str, float] = {
            lv.name: lv.expected_acceptance_rate for lv in config.levels
        }
        self._latency_ema: dict[str, float] = {
            lv.name: lv.expected_latency_ms for lv in config.levels
        }
        self._sample_counts: dict[str, int] = {lv.name: 0 for lv in config.levels}
        self._stats = CASSpecStats()
        for lv in config.levels:
            self._stats.level_usage_counts[lv.name] = 0
            self._stats.level_acceptance_history[lv.name] = []
        # Per-context hint cache: last-used level per context_hash.
        self._context_hint: dict[str, str] = {}
        self._round_robin_idx = 0

    # ---- routing ---------------------------------------------------------

    def _total_samples(self) -> int:
        return sum(self._sample_counts.values())

    def _score(self, level: CascadeLevel) -> float:
        acc = self._acceptance_ema[level.name]
        lat = max(self._latency_ema[level.name], 1e-3)
        return (acc * level.max_draft_length) / lat

    def _level_by_name(self, name: str) -> CascadeLevel | None:
        for lv in self.config.levels:
            if lv.name == name:
                return lv
        return None

    def route(self, context_hash: str, recent_tokens: list[int]) -> CascadeLevel:
        """Pick a cascade level for the current decoding step."""
        levels = self.config.levels
        if not levels:
            raise ValueError("DyTCRouter: no cascade levels configured")

        # Warmup: round-robin so every level gets observed.
        if self._total_samples() < self.config.min_samples_for_route:
            chosen = levels[self._round_robin_idx % len(levels)]
            self._round_robin_idx += 1
            return chosen

        # Fixed / greedy strategies bypass DyTC scoring.
        if self.config.routing_strategy == "fixed":
            return levels[0]
        if self.config.routing_strategy == "greedy":
            # Greedy = always the highest-acceptance level regardless of cost.
            return max(levels, key=lambda lv: self._acceptance_ema[lv.name])

        # DyTC: throughput-optimal, with a context-stability bias.
        best_name_hint = self._context_hint.get(context_hash)
        scored = sorted(levels, key=self._score, reverse=True)
        best = scored[0]

        # Hysteresis: if the cached level is within 10% of the best score,
        # stick with it to avoid oscillation within a single conversation.
        if best_name_hint:
            hinted = self._level_by_name(best_name_hint)
            if hinted is not None:
                if self._score(hinted) >= self._score(best) * 0.9:
                    best = hinted

        self._context_hint[context_hash] = best.name
        return best

    # ---- online updates --------------------------------------------------

    def update(
        self,
        level_name: str,
        drafts_proposed: int,
        drafts_accepted: int,
        latency_ms: float,
    ):
        """EMA update for acceptance + latency after a round."""
        if level_name not in self._acceptance_ema:
            return  # unknown level, ignore
        alpha = self.config.acceptance_ema_alpha

        if drafts_proposed > 0:
            rate = drafts_accepted / drafts_proposed
            self._acceptance_ema[level_name] = (
                (1 - alpha) * self._acceptance_ema[level_name] + alpha * rate
            )
            hist = self._stats.level_acceptance_history[level_name]
            hist.append(rate)
            if len(hist) > self.config.history_window:
                del hist[: len(hist) - self.config.history_window]

        if latency_ms > 0 and drafts_proposed > 0:
            per_token = latency_ms / max(drafts_proposed, 1)
            self._latency_ema[level_name] = (
                (1 - alpha) * self._latency_ema[level_name] + alpha * per_token
            )

        self._sample_counts[level_name] += 1
        self._stats.level_usage_counts[level_name] = (
            self._stats.level_usage_counts.get(level_name, 0) + 1
        )
        self._stats.total_drafts_generated += drafts_proposed
        self._stats.total_drafts_accepted += drafts_accepted
        # Running throughput estimate as a proxy for end-to-end speedup.
        self._stats.avg_speedup_estimate = max(
            1.0,
            1.0 + self._stats.total_drafts_accepted
            / max(self._stats.total_drafts_generated, 1)
            * 1.5,  # heuristic scale calibrated to paper's 1.1-2.3x range
        )

    # ---- tree construction -----------------------------------------------

    def build_tree(
        self,
        root_context: list[int],
        root_level: CascadeLevel,
    ) -> list[TreeNode]:
        """Build a DyTC speculation tree.

        The root is expanded with `root_level.max_draft_length` children
        (one per speculated token), and each child is further expanded using
        progressively cheaper tiers while their expected acceptance stays
        above `prune_threshold`. Branches below the threshold are pruned.

        We reuse the existing `TreeNode` dataclass (line ~368) so the output
        is compatible with `TreeDrafter.build_attention_mask` /
        `select_longest_accepted_path`.
        """
        cfg = self.config
        nodes: list[TreeNode] = [TreeNode(token_id=-1, depth=0, parent_idx=-1)]

        # Order levels from most expensive (root tier) to cheapest.
        ordered = sorted(
            cfg.levels,
            key=lambda lv: lv.expected_latency_ms,
            reverse=True,
        )
        # Find the starting index so the root uses root_level.
        try:
            start = ordered.index(root_level)
        except ValueError:
            start = 0

        def expand(parent_idx: int, depth: int, level_idx: int):
            if depth >= cfg.tree_max_depth:
                return
            if level_idx >= len(ordered):
                return
            level = ordered[level_idx]
            if self._acceptance_ema[level.name] < cfg.prune_threshold:
                return  # prune: this tier is too weak to be worth branching
            branches = min(cfg.tree_branch_factor, level.max_draft_length)
            for b in range(branches):
                # Placeholder token IDs: the actual tokens are filled in by
                # CASSpecDrafter.propose() when it runs the draft function.
                # We pre-allocate structural nodes so that a tree attention
                # mask can be built upfront.
                child = TreeNode(
                    token_id=-2 - b,   # sentinel; replaced at draft-time
                    depth=depth + 1,
                    parent_idx=parent_idx,
                    log_prob=math.log(
                        max(self._acceptance_ema[level.name], 1e-4)
                    ),
                )
                child_idx = len(nodes)
                nodes.append(child)
                nodes[parent_idx].children_idx.append(child_idx)
                # Each child is re-expanded with the *next cheaper* tier.
                expand(child_idx, depth + 1, level_idx + 1)

        expand(0, 0, start)
        return nodes

    # ---- stats -----------------------------------------------------------

    def stats(self) -> CASSpecStats:
        return self._stats


class CASSpecDrafter:
    """Cascade Adaptive Self-Speculative drafter.

    Wires together a `DyTCRouter`, the existing `NGramDrafter`, and stubs for
    the layer-skip / int4-fast-path tiers. On each `step()` the drafter asks
    the router for a tier, drafts `k` tokens using that tier, returns them
    for verification by the target model, and updates the router with the
    observed acceptance rate.

    Real layer-skip / int4 drafting require MLXEngine hooks (a partial
    forward that runs only the first `N` layers, and a forward that runs the
    model under aggressive int4 activation quantization). Until those land
    these tiers fall back to NGramDrafter so the class is end-to-end
    testable today.
    """

    def __init__(
        self,
        target_model_info,
        cascade_config: CASSpecConfig,
        router: DyTCRouter | None = None,
    ):
        self.target_model_info = target_model_info
        self.config = cascade_config
        self.router = router or DyTCRouter(cascade_config)
        # Shared n-gram drafter reused across tiers that currently stub-out
        # to it. A single instance keeps the n-gram table warm.
        self._ngram = NGramDrafter(n=3, max_draft=8)
        self._last_level: CascadeLevel | None = None

    # ---- tier implementations -------------------------------------------

    def _ngram_draft(self, level: CascadeLevel, context: list[int]) -> list[int]:
        self._ngram.max_draft = level.max_draft_length
        return self._ngram.draft(context)[: level.max_draft_length]

    def _layer_skip_draft(
        self,
        n_skip: int,
        level: CascadeLevel,
        context: list[int],
    ) -> list[int]:
        """Layer-skip self-draft stub.

        TODO(phase12): wire to `MLXEngine.partial_forward(layers=N-n_skip)`
        in `src/forge/engine/mlx_engine.py` once that hook is exposed. Until
        then we reuse the n-gram table so this tier is structurally present
        and the DyTC router can learn against it in tests.
        """
        _ = n_skip  # hook placeholder
        return self._ngram_draft(level, context)

    def _int4_fast_decode_draft(
        self,
        level: CascadeLevel,
        context: list[int],
    ) -> list[int]:
        """Int4 activation-quantized fast-path draft stub.

        TODO(phase12): wire to `MLXEngine.int4_fast_decode(...)` once the
        int4 activation-quantization path lands (companion change in
        `engine/mlx_engine.py`). Falls back to n-gram today.
        """
        return self._ngram_draft(level, context)

    # ---- public API ------------------------------------------------------

    def observe(self, token_id: int):
        """Feed an accepted/generated token into the n-gram table."""
        self._ngram.observe(token_id)

    def propose(
        self,
        prompt_tokens: list[int],
        k: int | None = None,
    ) -> list[int]:
        """Route + draft one batch of tokens."""
        context_hash = str(hash(tuple(prompt_tokens[-16:])))
        level = self.router.route(context_hash, prompt_tokens[-8:])
        self._last_level = level

        if k is not None:
            # Temporary override of max_draft_length for this round only.
            effective_max = min(k, level.max_draft_length)
        else:
            effective_max = level.max_draft_length

        did = level.draft_fn_id
        if did == "ngram":
            drafts = self._ngram_draft(level, prompt_tokens)
        elif did.startswith("layer_skip_"):
            try:
                n_skip = int(did.split("_")[-1])
            except ValueError:
                n_skip = 0
            drafts = self._layer_skip_draft(n_skip, level, prompt_tokens)
        elif did == "int4_decode":
            drafts = self._int4_fast_decode_draft(level, prompt_tokens)
        else:
            drafts = self._ngram_draft(level, prompt_tokens)

        return drafts[:effective_max]

    def verify(
        self,
        drafts: list[int],
        target_logits,
    ) -> tuple[int, list[int]]:
        """Simple greedy verification against target top-1 predictions.

        Compatible with the existing tree verification helpers: callers that
        already have target logits per position can compare argmax-by-position
        to the draft sequence. Returns `(accepted_count, accepted_tokens)`.
        """
        if target_logits is None or not drafts:
            return 0, []
        accepted: list[int] = []
        try:
            import mlx.core as mx  # noqa: F401
            has_mlx = True
        except Exception:
            has_mlx = False

        for i, draft_token in enumerate(drafts):
            row = target_logits[i] if hasattr(target_logits, "__len__") else None
            if row is None:
                break
            try:
                if has_mlx and hasattr(row, "shape"):
                    import mlx.core as mx
                    pred = int(mx.argmax(row).item())
                else:
                    pred = int(max(range(len(row)), key=lambda j: row[j]))
            except Exception:
                break
            if pred == draft_token:
                accepted.append(draft_token)
            else:
                break
        return len(accepted), accepted

    def step(self, prompt_state) -> tuple[int, str]:
        """One full round: propose + verify + update.

        `prompt_state` is a light protocol: it must expose `tokens` (a list
        of ints) and optionally `target_logits_for(drafts)` returning the
        per-position target logits used for verification. This keeps the
        drafter decoupled from MLXEngine so it can be unit-tested without
        a real model.
        """
        tokens: list[int] = getattr(prompt_state, "tokens", []) or []
        drafts = self.propose(tokens)

        target_logits = None
        if hasattr(prompt_state, "target_logits_for"):
            try:
                target_logits = prompt_state.target_logits_for(drafts)
            except Exception:
                target_logits = None

        accepted_count, accepted_tokens = self.verify(drafts, target_logits)

        level_name = self._last_level.name if self._last_level else "unknown"
        # Latency accounting: in absence of a real timer we attribute the
        # level's calibrated per-token cost. MLXEngine integration will
        # replace this with a wall-clock measurement.
        latency_ms = (
            (self._last_level.expected_latency_ms if self._last_level else 10.0)
            * max(len(drafts), 1)
        )
        self.router.update(
            level_name=level_name,
            drafts_proposed=len(drafts),
            drafts_accepted=accepted_count,
            latency_ms=latency_ms,
        )
        for tok in accepted_tokens:
            self.observe(tok)
        self.router.stats().tokens_generated += accepted_count
        return accepted_count, level_name


def build_default_cas_spec_config(
    target_params_b: float,
    available_memory_gb: float,
) -> CASSpecConfig:
    """Heuristic default cascade for a given model size.

    - 7B-class models get the full 3-tier cascade (ngram → layer-skip-6 →
      int4-decode). The int4 tier is worthwhile because a 7B forward is
      cheap enough that an activation-quantized fast path is a meaningful
      win over the full-precision target.
    - 30B+ models drop the int4 tier: at that scale even int4 decode costs
      dominate, so we stick with ngram + a deeper layer-skip (skip 10
      layers) to keep drafting cheap relative to verification.
    - Intermediate sizes (15-30B) use the 30B+ profile to be conservative.
    """
    _ = available_memory_gb  # reserved for future headroom-based tuning

    if target_params_b < 15:
        levels = [
            CascadeLevel(
                name="ngram",
                draft_fn_id="ngram",
                expected_latency_ms=10.0,
                expected_acceptance_rate=0.70,
                max_draft_length=4,
            ),
            CascadeLevel(
                name="layer_skip_6",
                draft_fn_id="layer_skip_6",
                expected_latency_ms=25.0,
                expected_acceptance_rate=0.55,
                max_draft_length=5,
            ),
            CascadeLevel(
                name="int4_decode",
                draft_fn_id="int4_decode",
                expected_latency_ms=60.0,
                expected_acceptance_rate=0.40,
                max_draft_length=3,
            ),
        ]
    else:
        levels = [
            CascadeLevel(
                name="ngram",
                draft_fn_id="ngram",
                expected_latency_ms=10.0,
                expected_acceptance_rate=0.70,
                max_draft_length=3,
            ),
            CascadeLevel(
                name="layer_skip_10",
                draft_fn_id="layer_skip_10",
                expected_latency_ms=40.0,
                expected_acceptance_rate=0.50,
                max_draft_length=4,
            ),
        ]

    return CASSpecConfig(
        levels=levels,
        routing_strategy="dytc",
        acceptance_ema_alpha=0.2,
        min_samples_for_route=20,
        tree_branch_factor=3,
        tree_max_depth=4,
        prune_threshold=0.15,
    )


def format_cas_spec_report(
    config: CASSpecConfig,
    stats: CASSpecStats | None = None,
) -> str:
    """Human-readable CAS-Spec configuration + live stats report."""
    lines = [
        "CAS-Spec Cascade Configuration (Phase 12, Tier S)",
        "=" * 65,
        f"  Routing:       {config.routing_strategy}",
        f"  EMA alpha:     {config.acceptance_ema_alpha}",
        f"  Warmup:        {config.min_samples_for_route} samples",
        f"  Tree:          branch={config.tree_branch_factor}, "
        f"depth={config.tree_max_depth}, prune<{config.prune_threshold}",
        "",
        "  Cascade Levels (fast → slow):",
        "  " + "-" * 61,
        f"  {'name':<18}{'draft_fn':<18}{'k':>4}{'lat(ms)':>10}{'accept':>10}",
    ]
    for lv in config.levels:
        lines.append(
            f"  {lv.name:<18}{lv.draft_fn_id:<18}"
            f"{lv.max_draft_length:>4}"
            f"{lv.expected_latency_ms:>10.1f}"
            f"{lv.expected_acceptance_rate:>10.2f}"
        )

    if stats is not None:
        summary = stats.summary()
        lines.append("")
        lines.append("  Runtime Stats:")
        lines.append("  " + "-" * 61)
        lines.append(f"  Tokens generated:    {summary['tokens_generated']}")
        lines.append(f"  Drafts generated:    {summary['total_drafts_generated']}")
        lines.append(f"  Drafts accepted:     {summary['total_drafts_accepted']}")
        lines.append(
            f"  Overall acceptance:  {summary['overall_acceptance_rate']}"
        )
        lines.append(
            f"  Est. speedup:        {summary['avg_speedup_estimate']}x"
        )
        lines.append("  Per-level:")
        for name, per in summary["per_level"].items():
            lines.append(
                f"    {name:<18} uses={per['uses']:<5} "
                f"avg_acc={per['avg_acceptance']:<6} "
                f"last={per['last_acceptance']:<6} "
                f"n={per['samples']}"
            )

    return "\n".join(lines)
