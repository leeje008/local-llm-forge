"""Model family alternative recommendations.

When a model is too large to run locally, suggest smaller models
from the same family or quality-equivalent alternatives.
"""

from __future__ import annotations

from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.model_inspector import ModelProfile

# Model family alternative database
# Structure: architecture → size_tier → list of (model_id, params_b, reason)
_ALTERNATIVES_DB: dict[str, dict[str, list[tuple[str, float, str]]]] = {
    "llama": {
        "405b": [
            ("meta-llama/Llama-3.3-70B-Instruct", 70.0, "405B급 성능, GPQA/MATH/IFEval에서 동등"),
            ("meta-llama/Llama-3.1-70B-Instruct", 70.0, "강력한 범용 모델"),
            ("meta-llama/Llama-3.1-8B-Instruct", 8.0, "빠른 추론, 기본 태스크용"),
        ],
        "70b": [
            ("Qwen/Qwen2.5-32B-Instruct", 32.0, "32B 4-bit로 충분한 품질"),
            ("meta-llama/Llama-3.1-8B-Instruct", 8.0, "빠른 추론, 기본 태스크"),
            ("mistralai/Mixtral-8x7B-Instruct-v0.1", 46.7, "MoE: 12.9B 활성, 70B급 품질"),
        ],
        "13b": [
            ("meta-llama/Llama-3.1-8B-Instruct", 8.0, "약간 작지만 빠름"),
        ],
    },
    "qwen2": {
        "72b": [
            ("Qwen/Qwen2.5-32B-Instruct", 32.0, "4-bit로 48GB에 적합"),
            ("Qwen/Qwen2.5-14B-Instruct", 14.0, "빠른 추론, 좋은 품질"),
            ("Qwen/Qwen2.5-7B-Instruct", 7.6, "가볍고 빠름"),
        ],
        "32b": [
            ("Qwen/Qwen2.5-14B-Instruct", 14.0, "14B로 여유있는 메모리"),
            ("Qwen/Qwen2.5-7B-Instruct", 7.6, "가볍고 빠름"),
        ],
    },
    "qwen3": {
        "235b": [
            ("Qwen/Qwen3-32B", 32.0, "4-bit로 48GB에 적합"),
            ("Qwen/Qwen3-14B", 14.0, "빠른 추론"),
            ("Qwen/Qwen3-8B", 8.0, "가볍고 빠름"),
        ],
        "30b": [
            ("Qwen/Qwen3-14B", 14.0, "14B 4-bit"),
            ("Qwen/Qwen3-8B", 8.0, "가볍고 빠름"),
        ],
    },
    "mistral": {
        "123b": [
            ("mistralai/Mixtral-8x7B-Instruct-v0.1", 46.7, "MoE: 12.9B 활성"),
            ("mistralai/Mistral-7B-Instruct-v0.3", 7.2, "빠르고 효율적"),
        ],
    },
    "gemma": {
        "27b": [
            ("google/gemma-2-9b-it", 9.0, "9B로 빠른 추론"),
            ("google/gemma-2-2b-it", 2.0, "매우 가벼움"),
        ],
    },
    "phi": {
        "14b": [
            ("microsoft/Phi-3.5-mini-instruct", 3.8, "코딩/구조화 태스크에 강함"),
        ],
    },
}

# MoE alternatives — when dense model too large, suggest MoE with lower active params
_MOE_ALTERNATIVES: list[tuple[str, float, float, str]] = [
    # (model_id, total_params_b, active_params_b, reason)
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", 46.7, 12.9, "8x7B MoE, 12.9B 활성, 70B급 품질"),
    ("Qwen/Qwen1.5-MoE-A2.7B", 14.3, 2.7, "2.7B 활성으로 7B급 성능"),
]

# Universal small model fallbacks
_UNIVERSAL_FALLBACKS: list[tuple[str, float, str]] = [
    ("Qwen/Qwen2.5-32B-Instruct", 32.0, "범용 32B, 4-bit로 ~17GB"),
    ("Qwen/Qwen2.5-14B-Instruct", 14.0, "범용 14B, 4-bit로 ~8GB"),
    ("Qwen/Qwen2.5-7B-Instruct", 7.6, "범용 7B, 4-bit로 ~4GB"),
    ("meta-llama/Llama-3.1-8B-Instruct", 8.0, "범용 8B, 4-bit로 ~4GB"),
]


def _size_tier(params_b: float) -> str:
    """Map parameter count to size tier key."""
    if params_b >= 350:
        return "405b"
    elif params_b >= 100:
        return "123b"
    elif params_b >= 60:
        return "72b"  # also matches 70b
    elif params_b >= 25:
        return "32b"  # also matches 27b, 30b
    elif params_b >= 10:
        return "14b"  # also matches 13b
    return ""


def find_alternatives(
    model: ModelProfile,
    hardware: HardwareProfile,
    max_results: int = 3,
) -> list[dict]:
    """Find alternative models that fit on the given hardware."""
    results = []
    seen = set()
    arch = model.architecture.lower()
    tier = _size_tier(model.total_params_b)

    # 1. Same family alternatives
    family_alts = _ALTERNATIVES_DB.get(arch, {}).get(tier, [])
    # Also check neighboring tiers
    if not family_alts:
        for t_key, t_alts in _ALTERNATIVES_DB.get(arch, {}).items():
            family_alts.extend(t_alts)

    for model_id, params_b, reason in family_alts:
        if model_id in seen or params_b >= model.total_params_b:
            continue
        weight_4bit = params_b * 0.5 * 1.05  # rough 4-bit estimate
        if weight_4bit < hardware.usable_memory_gb * 0.85:
            tps = (
                (hardware.memory_bandwidth_gbs / weight_4bit) * 0.37
                if hardware.memory_bandwidth_gbs > 0
                else 0
            )
            results.append({
                "model_id": model_id,
                "name": model_id.split("/")[-1],
                "params_b": params_b,
                "quant": "int4",
                "estimated_tps": round(tps, 1),
                "reason": reason,
                "command": f"forge optimize {model_id}",
            })
            seen.add(model_id)

    # 2. MoE alternatives (if original is dense)
    if model.model_type == "dense" and model.total_params_b >= 14:
        for moe_id, total_p, active_p, reason in _MOE_ALTERNATIVES:
            if moe_id in seen:
                continue
            weight_4bit = total_p * 0.5 * 1.05
            if weight_4bit < hardware.usable_memory_gb * 0.85:
                tps = (
                    (hardware.memory_bandwidth_gbs / (active_p * 0.5 * 1.05)) * 0.37
                    if hardware.memory_bandwidth_gbs > 0
                    else 0
                )
                results.append({
                    "model_id": moe_id,
                    "name": moe_id.split("/")[-1] + f" (MoE, {active_p:.1f}B active)",
                    "params_b": total_p,
                    "quant": "int4",
                    "estimated_tps": round(tps, 1),
                    "reason": reason,
                    "command": f"forge optimize {moe_id}",
                })
                seen.add(moe_id)

    # 3. Universal fallbacks if still not enough
    if len(results) < max_results:
        for fb_id, fb_params, fb_reason in _UNIVERSAL_FALLBACKS:
            if fb_id in seen or fb_params >= model.total_params_b:
                continue
            weight_4bit = fb_params * 0.5 * 1.05
            if weight_4bit < hardware.usable_memory_gb * 0.85:
                tps = (
                    (hardware.memory_bandwidth_gbs / weight_4bit) * 0.37
                    if hardware.memory_bandwidth_gbs > 0
                    else 0
                )
                results.append({
                    "model_id": fb_id,
                    "name": fb_id.split("/")[-1],
                    "params_b": fb_params,
                    "quant": "int4",
                    "estimated_tps": round(tps, 1),
                    "reason": fb_reason,
                    "command": f"forge optimize {fb_id}",
                })
                seen.add(fb_id)
            if len(results) >= max_results:
                break

    # Sort by params (largest first — closest to original quality)
    results.sort(key=lambda x: x["params_b"], reverse=True)
    return results[:max_results]
