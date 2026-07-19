"""Standard KV cache estimation and recommendations."""

from __future__ import annotations

from .base import KVCompressionMethod

# ---------------------------------------------------------------------------
# Original utility functions (preserved from Phase 1-4)
# ---------------------------------------------------------------------------

def estimate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,  # FP16
    batch_size: int = 1,
    compression: KVCompressionMethod = KVCompressionMethod.NONE,
) -> float:
    """Estimate KV cache memory in MB.

    KV cache = 2 (K+V) x layers x kv_heads x head_dim x seq_len x batch x dtype
    """
    bytes_total = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes

    if compression == KVCompressionMethod.TURBO:
        # ~5.5x compression: 16-bit → ~2.9 effective bits
        bytes_total = bytes_total * 2.9 / 16.0
    elif compression == KVCompressionMethod.FP8:
        bytes_total = bytes_total / 2  # FP16 → FP8

    return bytes_total / (1024 * 1024)


def estimate_max_context(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    available_memory_mb: float,
    dtype_bytes: int = 2,
    batch_size: int = 1,
    compression: KVCompressionMethod = KVCompressionMethod.NONE,
) -> int:
    """Calculate maximum context length that fits in available memory."""
    per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * batch_size * dtype_bytes
    if per_token_bytes == 0:
        return 0

    if compression == KVCompressionMethod.TURBO:
        per_token_bytes = per_token_bytes * 2.9 / 16.0
    elif compression == KVCompressionMethod.FP8:
        per_token_bytes = per_token_bytes / 2

    max_tokens = int((available_memory_mb * 1024 * 1024) / per_token_bytes)
    return max_tokens


def recommend_kv_optimization(
    model_params_b: float,
    context_length: int,
    available_memory_gb: float,
) -> dict:
    """Recommend KV cache optimizations based on model and memory constraints."""
    recommendations = {}

    kv_rough_gb = context_length * 0.0001 * model_params_b
    kv_pct = (kv_rough_gb / available_memory_gb) * 100

    # TurboQuant: recommended when KV cache is >10% of memory
    if kv_pct > 10:
        savings = kv_rough_gb * (1 - 1 / 5.5)
        recommendations["turbo_kv"] = {
            "enabled": True,
            "reason": (
                f"KV cache ~{kv_pct:.0f}% of memory. TurboQuant saves ~{savings:.1f}GB (5.5x)"
            ),
            "savings_gb": savings,
            "priority": "high" if kv_pct > 30 else "medium",
        }

    # FP8 KV: simpler alternative when KV is moderate
    if kv_pct > 20:
        recommendations["fp8_kv"] = {
            "enabled": True,
            "reason": f"KV cache ~{kv_pct:.0f}% of memory. FP8 saves ~{kv_rough_gb / 2:.1f}GB",
            "savings_gb": kv_rough_gb / 2,
        }

    # H2O eviction: recommended for long contexts
    if context_length > 8192:
        recommendations["h2o_eviction"] = {
            "enabled": True,
            "budget_ratio": 0.2,
            "reason": (
                f"Context {context_length:,} is long. H2O keeps 20% heavy-hitters, saves ~80% KV"
            ),
        }

    # Ada-KV: upgrade H2O with per-head budgets
    if context_length > 16384:
        recommendations["ada_kv"] = {
            "enabled": True,
            "reason": "Very long context benefits from per-head adaptive budgets (Ada-KV)",
        }

    # Sliding window
    if context_length > 16384:
        recommendations["sliding_window"] = {
            "enabled": True,
            "window_size": 8192,
            "reason": f"Context {context_length:,} — sliding window limits KV growth",
        }

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
    """Format KV cache analysis report with compression comparison."""
    if context_lengths is None:
        context_lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

    lines = [
        "KV Cache Analysis",
        "=" * 70,
        f"  Layers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}",
        "",
        f"  {'Context':>10}  {'FP16':>10}  {'FP8':>10}  {'TurboQ':>10}  {'H2O+TQ':>10}",
        f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}",
    ]

    for ctx in context_lengths:
        fp16 = estimate_kv_cache_memory(num_layers, num_kv_heads, head_dim, ctx)
        fp8 = estimate_kv_cache_memory(
            num_layers, num_kv_heads, head_dim, ctx,
            compression=KVCompressionMethod.FP8,
        )
        turbo = estimate_kv_cache_memory(
            num_layers, num_kv_heads, head_dim, ctx,
            compression=KVCompressionMethod.TURBO,
        )
        h2o_turbo = turbo * 0.2  # H2O keeps 20%

        def fmt(mb):
            if mb >= 1024:
                return f"{mb / 1024:.1f} GB"
            return f"{mb:.0f} MB"

        lines.append(
            f"  {ctx:>10,}  {fmt(fp16):>10}  {fmt(fp8):>10}  {fmt(turbo):>10}  {fmt(h2o_turbo):>10}"
        )

    lines.append("")
    lines.append("  TurboQ = TurboQuant 3-bit VQ (~5.5x compression)")
    lines.append("  H2O+TQ = TurboQuant + H2O eviction (top 20%)")

    return "\n".join(lines)
