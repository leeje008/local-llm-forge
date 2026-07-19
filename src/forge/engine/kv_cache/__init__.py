"""KV cache management and optimization utilities.

Includes:
- TurboQuant KV compression (ICLR 2026): Walsh-Hadamard + Lloyd-Max VQ, 5.5x compression
- H2O token eviction (2306.14048): Heavy-Hitter Oracle, keep top-20% tokens
- Ada-KV per-head budget (NeurIPS 2025): adaptive per-head eviction budgets
- Standard KV cache estimation and recommendations
"""

from __future__ import annotations

from .ada_kv import AdaKVConfig, AdaKVManager
from .base import KVCacheStats, KVCompressionMethod, KVEvictionPolicy
from .estimation import (
    estimate_kv_cache_memory,
    estimate_max_context,
    format_kv_report,
    recommend_kv_optimization,
)
from .h2o import H2OConfig, H2OEvictionManager
from .lava import LAVaConfig, LAVaManager, LAVaStats, format_lava_report
from .turboquant import CompressedKV, TurboQuantCompressor, TurboQuantConfig
from .xkv import XKVCompressor, XKVConfig, XKVGroup, estimate_xkv_compression

__all__ = [
    "AdaKVConfig",
    "AdaKVManager",
    "CompressedKV",
    "H2OConfig",
    "H2OEvictionManager",
    "KVCacheStats",
    "KVCompressionMethod",
    "KVEvictionPolicy",
    "LAVaConfig",
    "LAVaManager",
    "LAVaStats",
    "TurboQuantCompressor",
    "TurboQuantConfig",
    "XKVConfig",
    "XKVCompressor",
    "XKVGroup",
    "estimate_kv_cache_memory",
    "estimate_max_context",
    "estimate_xkv_compression",
    "format_kv_report",
    "format_lava_report",
    "recommend_kv_optimization",
]
