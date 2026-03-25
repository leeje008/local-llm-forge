"""KV cache management and optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass


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


def estimate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,  # FP16
    batch_size: int = 1,
) -> float:
    """Estimate KV cache memory in MB.

    KV cache = 2 (K+V) × layers × kv_heads × head_dim × seq_len × batch × dtype
    """
    bytes_total = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return bytes_total / (1024 * 1024)


def estimate_max_context(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    available_memory_mb: float,
    dtype_bytes: int = 2,
    batch_size: int = 1,
) -> int:
    """Calculate maximum context length that fits in available memory."""
    per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * batch_size * dtype_bytes
    if per_token_bytes == 0:
        return 0
    max_tokens = int((available_memory_mb * 1024 * 1024) / per_token_bytes)
    return max_tokens


def recommend_kv_optimization(
    model_params_b: float,
    context_length: int,
    available_memory_gb: float,
) -> dict:
    """Recommend KV cache optimizations based on model and memory constraints.

    Returns a dict of recommendations.
    """
    recommendations = {}

    # FP8 KV cache: halves KV memory, minimal quality impact
    # Worthwhile when KV cache is >20% of total memory
    kv_rough_gb = context_length * 0.0001 * model_params_b  # rough estimate
    kv_pct = (kv_rough_gb / available_memory_gb) * 100

    if kv_pct > 20:
        recommendations["fp8_kv"] = {
            "enabled": True,
            "reason": f"KV cache is ~{kv_pct:.0f}% of memory, FP8 saves ~{kv_rough_gb/2:.1f}GB",
            "savings_gb": kv_rough_gb / 2,
        }

    # Sliding window attention: useful for very long contexts
    if context_length > 16384:
        recommendations["sliding_window"] = {
            "enabled": True,
            "window_size": 8192,
            "reason": f"Context {context_length:,} is long, sliding window limits KV growth",
        }

    # GQA ratio optimization
    # (Already handled at model level, but note it here)
    recommendations["gqa_note"] = {
        "enabled": True,
        "reason": "GQA reduces KV cache proportionally to head ratio",
    }

    return recommendations


def format_kv_report(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_lengths: list[int] | None = None,
) -> str:
    """Format KV cache analysis report."""
    if context_lengths is None:
        context_lengths = [2048, 4096, 8192, 16384, 32768]

    lines = [
        "KV Cache Analysis",
        "=" * 50,
        f"  Layers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}",
        "",
        f"  {'Context':>10}  {'KV Memory (FP16)':>16}  {'KV Memory (FP8)':>15}",
        f"  {'-'*10}  {'-'*16}  {'-'*15}",
    ]

    for ctx in context_lengths:
        mem_fp16 = estimate_kv_cache_memory(num_layers, num_kv_heads, head_dim, ctx)
        mem_fp8 = mem_fp16 / 2
        lines.append(
            f"  {ctx:>10,}  {mem_fp16:>13.0f} MB  {mem_fp8:>12.0f} MB"
        )

    return "\n".join(lines)
