"""Core KV cache data structures shared across compression/eviction strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KVCacheStats:
    """Statistics about KV cache usage."""

    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    current_seq_len: int = 0
    max_seq_len: int = 0
    memory_used_mb: float = 0.0
    memory_capacity_mb: float = 0.0
    utilization_pct: float = 0.0


class KVCompressionMethod(str, Enum):
    NONE = "none"
    TURBO = "turbo"          # TurboQuant: Walsh-Hadamard + Lloyd-Max VQ
    FP8 = "fp8"              # MLX native kv_bits=8


class KVEvictionPolicy(str, Enum):
    NONE = "none"
    SLIDING = "sliding"      # StreamingLLM: attention sink + sliding window
    H2O = "h2o"              # Heavy-Hitter Oracle: keep top-K by cumulative attention
    ADA_KV = "ada_kv"        # Ada-KV: per-head adaptive budget H2O
