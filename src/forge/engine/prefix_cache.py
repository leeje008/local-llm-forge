"""Radix Tree Prefix Cache — SGLang-inspired in-memory KV cache sharing.

Implements a radix tree (compact trie) indexed by token sequences. When a new
prompt shares a prefix with a previously cached prompt, the shared KV cache
is reused instantly, reducing TTFT from O(full_prompt) to O(new_suffix_only).

Key benefits for multi-turn conversations:
- System prompt KV is computed once, reused across all turns
- Conversation history prefix is shared across follow-up turns
- Cross-conversation sharing for identical system prompts

Based on: SGLang RadixAttention (2312.07104), vllm-mlx prefix caching (2601.19139)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """A cached KV state for a token sequence."""

    tokens: tuple[int, ...]       # The token sequence
    kv_state: Any                  # The KV cache tensors (opaque — framework-specific)
    created_at: float = 0.0       # Timestamp
    last_accessed: float = 0.0    # Last access timestamp
    access_count: int = 0         # Number of times reused
    memory_bytes: int = 0         # Estimated memory usage

    def touch(self):
        self.last_accessed = time.monotonic()
        self.access_count += 1


class RadixNode:
    """Node in the radix tree (compact trie).

    Each edge represents a sequence of tokens (not just a single token),
    enabling O(n/k) lookup where k is the average edge length.
    """

    __slots__ = ("children", "edge_tokens", "cache_entry", "is_leaf")

    def __init__(self):
        self.children: dict[int, RadixNode] = {}  # first_token → child node
        self.edge_tokens: tuple[int, ...] = ()     # tokens on the edge to this node
        self.cache_entry: CacheEntry | None = None  # KV cache stored at this node
        self.is_leaf: bool = False


@dataclass
class PrefixCacheStats:
    """Statistics about prefix cache performance."""

    total_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_saved: int = 0       # Tokens that didn't need recomputation
    total_entries: int = 0
    total_memory_bytes: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(self.total_lookups, 1)

    @property
    def memory_mb(self) -> float:
        return self.total_memory_bytes / (1024 * 1024)


class RadixPrefixCache:
    """In-memory radix tree prefix cache for KV state reuse.

    Usage:
        cache = RadixPrefixCache(max_memory_mb=2048)

        # Store KV after computing
        tokens = tokenizer.encode("You are a helpful assistant...")
        cache.insert(tokens, kv_state)

        # Later, find longest matching prefix
        new_tokens = tokenizer.encode("You are a helpful assistant. User: hello")
        match = cache.find_longest_prefix(new_tokens)
        if match:
            # Reuse match.kv_state, only compute suffix tokens
            suffix = new_tokens[match.prefix_len:]
    """

    def __init__(self, max_memory_mb: float = 2048.0, max_entries: int = 1000):
        self._root = RadixNode()
        self._max_memory = int(max_memory_mb * 1024 * 1024)
        self._max_entries = max_entries
        self.stats = PrefixCacheStats()

    def insert(
        self,
        tokens: list[int] | tuple[int, ...],
        kv_state: Any,
        memory_bytes: int = 0,
    ) -> bool:
        """Insert a token sequence and its KV cache into the radix tree.

        Args:
            tokens: The full token sequence.
            kv_state: The KV cache state (opaque object).
            memory_bytes: Estimated memory of kv_state in bytes.

        Returns:
            True if inserted, False if eviction failed to make room.
        """
        if not tokens:
            return False

        tokens_tuple = tuple(tokens)

        # Check memory budget
        if memory_bytes > 0:
            while (
                self.stats.total_memory_bytes + memory_bytes > self._max_memory
                or self.stats.total_entries >= self._max_entries
            ):
                if not self._evict_one():
                    return False  # Cannot make room

        entry = CacheEntry(
            tokens=tokens_tuple,
            kv_state=kv_state,
            created_at=time.monotonic(),
            last_accessed=time.monotonic(),
            access_count=0,
            memory_bytes=memory_bytes,
        )

        self._insert_into_tree(tokens_tuple, entry)
        self.stats.total_entries += 1
        self.stats.total_memory_bytes += memory_bytes
        return True

    def find_longest_prefix(self, tokens: list[int] | tuple[int, ...]) -> PrefixMatch | None:
        """Find the longest cached prefix of the given token sequence.

        Returns:
            PrefixMatch with the KV state and prefix length, or None if no match.
        """
        self.stats.total_lookups += 1
        tokens_tuple = tuple(tokens)

        best_match: CacheEntry | None = None
        best_len = 0

        node = self._root
        pos = 0

        while pos < len(tokens_tuple):
            first_token = tokens_tuple[pos]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            edge = child.edge_tokens

            # Match as many tokens along this edge as possible
            match_len = 0
            while match_len < len(edge) and (pos + match_len) < len(tokens_tuple):
                if edge[match_len] != tokens_tuple[pos + match_len]:
                    break
                match_len += 1

            pos += match_len

            if match_len < len(edge):
                # Partial edge match — check if this node has a cache entry
                break

            # Full edge matched
            if child.cache_entry is not None:
                best_match = child.cache_entry
                best_len = pos

            node = child

        if best_match is not None:
            best_match.touch()
            self.stats.cache_hits += 1
            self.stats.total_tokens_saved += best_len
            return PrefixMatch(
                kv_state=best_match.kv_state,
                prefix_len=best_len,
                tokens_saved=best_len,
                entry=best_match,
            )

        self.stats.cache_misses += 1
        return None

    def clear(self):
        """Clear all cached entries."""
        self._root = RadixNode()
        self.stats = PrefixCacheStats()

    def get_all_entries(self) -> list[CacheEntry]:
        """Get all cache entries (for inspection/debugging)."""
        entries = []
        self._collect_entries(self._root, entries)
        return entries

    # --- Internal methods ---

    def _insert_into_tree(self, tokens: tuple[int, ...], entry: CacheEntry):
        """Insert tokens into the radix tree, splitting edges as needed."""
        node = self._root
        pos = 0

        while pos < len(tokens):
            first_token = tokens[pos]

            if first_token not in node.children:
                # Create new leaf node with remaining tokens as edge
                new_node = RadixNode()
                new_node.edge_tokens = tokens[pos:]
                new_node.cache_entry = entry
                new_node.is_leaf = True
                node.children[first_token] = new_node
                return

            child = node.children[first_token]
            edge = child.edge_tokens

            # Find common prefix length between edge and remaining tokens
            common = 0
            remaining = tokens[pos:]
            while common < len(edge) and common < len(remaining):
                if edge[common] != remaining[common]:
                    break
                common += 1

            if common == len(edge):
                # Full edge match — continue to child
                pos += common
                if pos == len(tokens):
                    # Exact match — update entry at this node
                    old_entry = child.cache_entry
                    child.cache_entry = entry
                    if old_entry:
                        self.stats.total_memory_bytes -= old_entry.memory_bytes
                        self.stats.total_entries -= 1
                    return
                node = child
            else:
                # Partial match — split the edge
                # Create split node at the divergence point
                split_node = RadixNode()
                split_node.edge_tokens = edge[:common]

                # Old child becomes child of split_node with remaining edge
                child.edge_tokens = edge[common:]
                split_node.children[edge[common]] = child

                # New leaf for the inserted tokens
                if pos + common < len(tokens):
                    new_leaf = RadixNode()
                    new_leaf.edge_tokens = tokens[pos + common:]
                    new_leaf.cache_entry = entry
                    new_leaf.is_leaf = True
                    split_node.children[tokens[pos + common]] = new_leaf
                else:
                    # The split point is exactly where we want to insert
                    split_node.cache_entry = entry

                node.children[first_token] = split_node
                return

        # tokens exhausted — store at current node
        node.cache_entry = entry

    def _evict_one(self) -> bool:
        """Evict the least recently used cache entry. Returns False if empty."""
        entries = self.get_all_entries()
        if not entries:
            return False

        # LRU eviction: remove the entry accessed longest ago
        lru = min(entries, key=lambda e: e.last_accessed)
        self._remove_entry(lru)
        self.stats.evictions += 1
        return True

    def _remove_entry(self, entry: CacheEntry):
        """Remove a specific entry from the tree."""
        # Walk the tree to find and remove
        self._remove_from_node(self._root, entry.tokens, 0)
        self.stats.total_entries -= 1
        self.stats.total_memory_bytes -= entry.memory_bytes

    def _remove_from_node(self, node: RadixNode, tokens: tuple[int, ...], pos: int) -> bool:
        """Recursively remove entry. Returns True if this node can be pruned."""
        if pos >= len(tokens):
            if node.cache_entry and node.cache_entry.tokens == tokens:
                node.cache_entry = None
                return not node.children  # Prune if no children
            return False

        first_token = tokens[pos]
        if first_token not in node.children:
            return False

        child = node.children[first_token]
        new_pos = pos + len(child.edge_tokens)

        if self._remove_from_node(child, tokens, new_pos):
            del node.children[first_token]
            # Compact: if this node has only one child and no entry, merge
            if len(node.children) == 1 and node.cache_entry is None and node is not self._root:
                only_key = next(iter(node.children))
                only_child = node.children[only_key]
                node.edge_tokens = node.edge_tokens + only_child.edge_tokens
                node.children = only_child.children
                node.cache_entry = only_child.cache_entry
                node.is_leaf = only_child.is_leaf

        return False

    def _collect_entries(self, node: RadixNode, entries: list[CacheEntry]):
        if node.cache_entry is not None:
            entries.append(node.cache_entry)
        for child in node.children.values():
            self._collect_entries(child, entries)


@dataclass
class PrefixMatch:
    """Result of a prefix cache lookup."""

    kv_state: Any           # The cached KV state
    prefix_len: int         # Number of tokens matched
    tokens_saved: int       # Tokens that don't need recomputation
    entry: CacheEntry       # The matched cache entry


def format_prefix_cache_report(cache: RadixPrefixCache) -> str:
    """Format prefix cache statistics for display."""
    s = cache.stats
    lines = [
        "Prefix Cache Statistics",
        "=" * 50,
        f"  Entries:       {s.total_entries}",
        f"  Memory:        {s.memory_mb:.1f} MB",
        f"  Lookups:       {s.total_lookups}",
        f"  Hits:          {s.cache_hits} ({s.hit_rate:.1%})",
        f"  Misses:        {s.cache_misses}",
        f"  Tokens Saved:  {s.total_tokens_saved:,}",
        f"  Evictions:     {s.evictions}",
    ]

    entries = cache.get_all_entries()
    if entries:
        lines.append("")
        lines.append("  Cached Prefixes:")
        for e in sorted(entries, key=lambda x: -x.access_count)[:5]:
            age = time.monotonic() - e.created_at
            lines.append(
                f"    [{e.access_count} hits, {age:.0f}s ago] "
                f"{len(e.tokens)} tokens, {e.memory_bytes / 1024:.0f} KB"
            )

    return "\n".join(lines)
