from __future__ import annotations

from dataclasses import dataclass, field

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
