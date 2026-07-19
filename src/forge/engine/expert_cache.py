"""MoE Expert Cache — LRU / Least-Stale / Predictive policies.

For MoE inference, only a small subset of experts activate per token.
Holding the full expert set in VRAM/unified-memory can exceed budget on
larger models (e.g. Mixtral-8x22B), so we keep a bounded cache of hot
experts and stream in cold experts on demand.

Three eviction policies are implemented:

1. LRU (baseline) — evict least-recently-used.
2. Least-Stale (SpecMD, 2602.03921) — track the activation *interval*
   per expert and evict the one whose time-since-last-use most exceeds
   its historical cycle length. Experts that fire periodically but
   infrequently stay in cache longer than pure LRU would keep them.
3. Predictive — `HiddenStatePredictor` maps the current hidden state to
   the expected expert ids for the *next* token; the cache pre-fetches
   the top-k predicted experts and evicts experts that are neither
   recently used nor in the top-k prediction.

The cache is backend-agnostic: expert "weights" are opaque objects
(numpy arrays, MLX arrays, PyTorch tensors, paths). Only the id →
object mapping is managed here.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Configuration + statistics
# ---------------------------------------------------------------------------


@dataclass
class ExpertCacheConfig:
    """Configuration for ExpertCache."""

    capacity: int = 16
    """Maximum number of experts held in memory."""
    policy: str = "lru"
    """One of 'lru', 'least_stale', 'predictive'."""
    predictive_top_k: int = 4
    """Number of experts to pre-fetch when using the predictive policy."""
    stale_alpha: float = 0.5
    """Blend factor between raw staleness and cycle-weighted staleness."""


@dataclass
class CacheStats:
    """Per-cache runtime statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    prefetches: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    @property
    def prefetch_hit_rate(self) -> float:
        if self.prefetches == 0:
            return 0.0
        return self.prefetch_hits / self.prefetches


# ---------------------------------------------------------------------------
# Hidden-state predictor (Phase 9.5)
# ---------------------------------------------------------------------------


class HiddenStatePredictor:
    """Tiny linear head that maps hidden states to expert activations.

    Fits a `[hidden_dim, num_experts]` weight matrix by numpy least
    squares against one-hot expert activation targets. At inference,
    multiplying a hidden state by W yields per-expert logits; the top-k
    indices are returned as the predicted active experts.

    This is intentionally simple — the point is to demonstrate that
    cheap hidden-state features already predict router decisions well
    enough to drive cache prefetching. Real deployments would train
    per-layer predictors with more features.
    """

    def __init__(self, hidden_dim: int, num_experts: int):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self._weights: Any | None = None  # numpy (hidden_dim, num_experts)
        self._fitted = False

    def fit(
        self,
        hidden_states: Any,
        expert_activations: Any,
    ) -> "HiddenStatePredictor":
        """Fit via least squares.

        Args:
            hidden_states: array-like, shape (N, hidden_dim)
            expert_activations: array-like, shape (N, num_experts) with
                values in {0, 1} or gate scores in [0, 1].
        """
        import numpy as np

        X = np.asarray(hidden_states, dtype=np.float32)
        Y = np.asarray(expert_activations, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.hidden_dim:
            raise ValueError(
                f"hidden_states must be (N, {self.hidden_dim}); got {X.shape}"
            )
        if Y.ndim != 2 or Y.shape[1] != self.num_experts:
            raise ValueError(
                f"expert_activations must be (N, {self.num_experts}); got {Y.shape}"
            )
        if X.shape[0] != Y.shape[0]:
            raise ValueError("hidden_states and expert_activations must share N")

        # Least squares: W = (X^T X)^-1 X^T Y, via numpy.linalg.lstsq for stability
        W, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self._weights = W.astype(np.float32)
        self._fitted = True
        return self

    def predict(self, hidden_state: Any, top_k: int = 4) -> list[int]:
        """Return the top-k expert ids expected to activate.

        `hidden_state` may be a single vector (hidden_dim,) or a batch
        (B, hidden_dim) — for batches, the mean hidden state is used.
        """
        import numpy as np

        if not self._fitted or self._weights is None:
            return []
        h = np.asarray(hidden_state, dtype=np.float32)
        if h.ndim == 2:
            h = h.mean(axis=0)
        if h.shape[-1] != self.hidden_dim:
            return []
        logits = h @ self._weights                  # (num_experts,)
        k = max(1, min(top_k, self.num_experts))
        # argpartition for O(n) top-k then sort the selection
        idx = np.argpartition(-logits, k - 1)[:k]
        idx = idx[np.argsort(-logits[idx])]
        return [int(i) for i in idx]

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Expert cache
# ---------------------------------------------------------------------------


class ExpertCache:
    """Bounded expert weight cache with pluggable eviction policy.

    The cache stores `expert_id → weight` mappings. Every `get` records
    the call tick; every `put` enforces the capacity invariant by
    evicting an expert selected by the configured policy.
    """

    def __init__(
        self,
        config: ExpertCacheConfig,
        predictor: HiddenStatePredictor | None = None,
    ):
        self.config = config
        self.predictor = predictor
        self.stats = CacheStats()

        # Ordered by recency for LRU semantics
        self._store: OrderedDict[int, Any] = OrderedDict()
        # Global monotonic tick
        self._tick: int = 0
        # Per-expert last access tick
        self._last_access: dict[int, int] = {}
        # Per-expert historical activation interval (EMA of tick deltas)
        self._interval_ema: dict[int, float] = defaultdict(lambda: 1.0)
        # Set of expert ids marked as "prefetched" by the predictor
        self._prefetched: set[int] = set()

        if config.policy not in ("lru", "least_stale", "predictive"):
            raise ValueError(f"Unknown policy: {config.policy}")

    # ----- core API ---------------------------------------------------

    def __contains__(self, expert_id: int) -> bool:
        return expert_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def get(self, expert_id: int) -> Any | None:
        """Return the expert weight if cached, else None.

        Updates recency and (for least-stale) the interval EMA.
        """
        self._tick += 1
        if expert_id in self._store:
            self.stats.hits += 1
            # Was this a prefetched entry that actually got used?
            if expert_id in self._prefetched:
                self.stats.prefetch_hits += 1
                self._prefetched.discard(expert_id)
            # Update LRU order
            self._store.move_to_end(expert_id)
            # Update interval EMA
            last = self._last_access.get(expert_id)
            if last is not None:
                delta = self._tick - last
                prev = self._interval_ema[expert_id]
                # EMA with alpha=0.3
                self._interval_ema[expert_id] = 0.7 * prev + 0.3 * delta
            self._last_access[expert_id] = self._tick
            return self._store[expert_id]

        self.stats.misses += 1
        return None

    def put(self, expert_id: int, weight: Any) -> None:
        """Insert an expert, evicting one if over capacity."""
        self._tick += 1
        if expert_id in self._store:
            self._store[expert_id] = weight
            self._store.move_to_end(expert_id)
            self._last_access[expert_id] = self._tick
            return

        while len(self._store) >= self.config.capacity:
            victim = self._select_victim()
            if victim is None:
                break
            self._evict(victim)

        self._store[expert_id] = weight
        self._last_access[expert_id] = self._tick

    # ----- eviction ---------------------------------------------------

    def _select_victim(self) -> int | None:
        """Choose an expert to evict under the current policy."""
        if not self._store:
            return None

        policy = self.config.policy

        if policy == "lru":
            # OrderedDict: oldest is first
            return next(iter(self._store))

        if policy == "least_stale":
            # Score = (now - last_access) / interval_ema
            # Higher score = more stale than its natural cycle ⇒ evict.
            # But don't evict an expert whose current gap is still within
            # its typical cycle (gives periodic experts a chance to fire).
            now = self._tick
            alpha = self.config.stale_alpha
            best_id = None
            best_score = -1.0
            for eid in self._store:
                # Never evict freshly inserted entries without data
                last = self._last_access.get(eid, now)
                gap = max(1, now - last)
                cycle = max(1.0, self._interval_ema.get(eid, 1.0))
                # Blend raw staleness with cycle-normalized staleness
                score = alpha * gap + (1 - alpha) * (gap / cycle)
                # Keep prefetched items alive unless truly stale
                if eid in self._prefetched:
                    score *= 0.5
                if score > best_score:
                    best_score = score
                    best_id = eid
            return best_id

        if policy == "predictive":
            # Evict the LRU expert that is NOT in the current prefetch set
            for eid in self._store:
                if eid not in self._prefetched:
                    return eid
            # All entries are prefetched — fall back to LRU
            return next(iter(self._store))

        return next(iter(self._store))

    def _evict(self, expert_id: int) -> None:
        self._store.pop(expert_id, None)
        self._prefetched.discard(expert_id)
        self.stats.evictions += 1

    # ----- predictive prefetch ----------------------------------------

    def prefetch_from_hidden_state(
        self,
        hidden_state: Any,
        loader,
    ) -> list[int]:
        """Ask the predictor which experts to pre-load, then load them.

        `loader(expert_id)` is a callback returning the weight tensor.
        Returns the list of expert ids that were actually prefetched
        (i.e. not already resident).
        """
        if self.predictor is None or not self.predictor.is_fitted:
            return []
        top = self.predictor.predict(hidden_state, top_k=self.config.predictive_top_k)
        prefetched_now: list[int] = []
        for eid in top:
            self._prefetched.add(eid)
            if eid not in self._store:
                self.stats.prefetches += 1
                try:
                    w = loader(eid)
                except Exception:
                    continue
                self.put(eid, w)
                prefetched_now.append(eid)
        return prefetched_now

    # ----- reporting --------------------------------------------------

    @property
    def hit_rate(self) -> float:
        return self.stats.hit_rate

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of cache state + stats."""
        return {
            "policy": self.config.policy,
            "capacity": self.config.capacity,
            "size": len(self._store),
            "resident_expert_ids": list(self._store.keys()),
            "tick": self._tick,
            "stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.hit_rate,
                "evictions": self.stats.evictions,
                "prefetches": self.stats.prefetches,
                "prefetch_hit_rate": self.stats.prefetch_hit_rate,
            },
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_cache_report(cache: ExpertCache) -> str:
    """Format an ExpertCache snapshot for CLI display."""
    snap = cache.snapshot()
    s = snap["stats"]
    lines = [
        "Expert Cache Report",
        "=" * 55,
        f"  Policy:           {snap['policy']}",
        f"  Capacity:         {snap['capacity']}",
        f"  Resident:         {snap['size']} experts",
        f"  Tick:             {snap['tick']}",
        "",
        f"  Hits:             {s['hits']}",
        f"  Misses:           {s['misses']}",
        f"  Hit rate:         {s['hit_rate']:.2%}",
        f"  Evictions:        {s['evictions']}",
        f"  Prefetches:       {s['prefetches']}",
        f"  Prefetch hit:     {s['prefetch_hit_rate']:.2%}",
    ]
    if snap["resident_expert_ids"]:
        sample = snap["resident_expert_ids"][:16]
        tail = " ..." if len(snap["resident_expert_ids"]) > 16 else ""
        lines.append(f"  Resident ids:     {sample}{tail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    """Synthetic sanity check for all three policies + the predictor."""
    import numpy as np

    # ----- LRU -----
    cache = ExpertCache(ExpertCacheConfig(capacity=2, policy="lru"))
    cache.put(0, "w0")
    cache.put(1, "w1")
    cache.get(0)            # touch 0 → 1 becomes LRU
    cache.put(2, "w2")      # should evict 1
    assert 1 not in cache and 0 in cache and 2 in cache

    # ----- Least-Stale -----
    cache = ExpertCache(ExpertCacheConfig(capacity=2, policy="least_stale"))
    cache.put(0, "w0")
    cache.put(1, "w1")
    for _ in range(5):
        cache.get(0)
        cache.get(1)
    cache.put(2, "w2")
    assert len(cache) == 2

    # ----- Predictor + predictive cache -----
    rng = np.random.default_rng(0)
    hidden_dim, num_experts, N = 8, 4, 200
    # Synthetic: expert id = argmax of first 4 hidden dims
    X = rng.standard_normal((N, hidden_dim)).astype(np.float32)
    Y = np.zeros((N, num_experts), dtype=np.float32)
    for i in range(N):
        Y[i, int(np.argmax(X[i, :num_experts]))] = 1.0

    predictor = HiddenStatePredictor(hidden_dim, num_experts)
    predictor.fit(X, Y)

    # Check prediction accuracy on training data
    correct = 0
    for i in range(N):
        top = predictor.predict(X[i], top_k=1)
        if top and top[0] == int(np.argmax(X[i, :num_experts])):
            correct += 1
    if correct / N < 0.7:
        return False

    cache = ExpertCache(
        ExpertCacheConfig(capacity=3, policy="predictive", predictive_top_k=2),
        predictor=predictor,
    )
    def _loader(eid: int) -> str:
        return f"w{eid}"
    cache.prefetch_from_hidden_state(X[0], _loader)
    return cache.stats.prefetches > 0


if __name__ == "__main__":
    print("ExpertCache self-test:", "OK" if _self_test() else "FAIL")
