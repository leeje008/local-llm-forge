"""Disk/CPU offloading feasibility estimation.

When a model doesn't fit entirely in GPU memory even with quantization,
estimate whether partial GPU offload via llama.cpp is viable.
"""

from __future__ import annotations

import shutil

from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.memory_calculator import calc_weight_memory
from forge.analyzer.model_inspector import ModelProfile


def estimate_offload(
    model: ModelProfile,
    hardware: HardwareProfile,
    quant: str = "int4",
    min_gpu_pct: float = 0.3,
) -> dict:
    """Estimate partial GPU offload feasibility via llama.cpp.

    llama.cpp supports -ngl (number of GPU layers) to split model between
    GPU and CPU. This estimates how many layers can fit on GPU and the
    resulting performance impact.

    Args:
        model: Model profile
        hardware: Hardware profile
        quant: Quantization level to use for offloaded model
        min_gpu_pct: Minimum fraction of layers on GPU to be considered viable

    Returns:
        Dict with feasibility info, gpu_layers, estimated_tps, command, etc.
    """
    weight_mem = calc_weight_memory(model, quant)
    num_layers = model.num_layers

    if num_layers == 0:
        return {"feasible": False, "reason": "Unknown number of layers"}

    # Memory budget: leave room for KV cache + overhead
    gpu_budget = hardware.usable_memory_gb * 0.75  # 75% for weights
    mem_per_layer = weight_mem / num_layers

    if mem_per_layer <= 0:
        return {"feasible": False, "reason": "Cannot calculate per-layer memory"}

    gpu_layers = min(int(gpu_budget / mem_per_layer), num_layers)
    cpu_layers = num_layers - gpu_layers
    gpu_pct = gpu_layers / num_layers * 100

    if gpu_pct < min_gpu_pct * 100:
        return {
            "feasible": False,
            "reason": f"Only {gpu_pct:.0f}% of layers fit on GPU (minimum {min_gpu_pct*100:.0f}%)",
            "gpu_layers": gpu_layers,
            "gpu_pct": gpu_pct,
        }

    # Estimate performance impact
    # GPU-only baseline TPS
    if hardware.memory_bandwidth_gbs > 0 and weight_mem > 0:
        gpu_only_tps = (hardware.memory_bandwidth_gbs / weight_mem) * 0.37
    else:
        gpu_only_tps = 0

    # CPU layers are ~5-10x slower than GPU layers
    # Rough model: overall TPS ≈ gpu_only_tps * (gpu_pct/100) * penalty_factor
    # penalty_factor accounts for CPU-GPU synchronization overhead
    if gpu_pct >= 90:
        penalty = 0.85  # minimal impact
    elif gpu_pct >= 70:
        penalty = 0.6
    elif gpu_pct >= 50:
        penalty = 0.4
    else:
        penalty = 0.25

    estimated_tps = gpu_only_tps * penalty

    # Context length for remaining memory
    kv_budget_gb = hardware.usable_memory_gb - (gpu_layers * mem_per_layer) - 2  # overhead
    if model.num_kv_heads and model.head_dim and model.num_layers:
        kv_per_token_gb = (
            2 * model.num_layers * (model.num_kv_heads or model.num_attention_heads)
            * model.head_dim * 2
        ) / 1e9
        context_length = int(kv_budget_gb / kv_per_token_gb) if kv_per_token_gb > 0 else 4096
        context_length = max(min(context_length, 8192), 2048)
    else:
        context_length = 4096

    # Build command
    has_llamacpp = shutil.which("llama-cli") or shutil.which("llama.cpp")
    runtime = "llama.cpp" if has_llamacpp else "ollama"

    safe_name = model.model_id.replace("/", "--")
    if runtime == "llama.cpp":
        command = f"llama-cli -m {safe_name}.gguf -ngl {gpu_layers} -c {context_length}"
    else:
        command = f"ollama run {model.model_id}  # Ollama auto-manages offloading"

    return {
        "feasible": True,
        "gpu_layers": gpu_layers,
        "cpu_layers": cpu_layers,
        "total_layers": num_layers,
        "gpu_pct": gpu_pct,
        "gpu_memory_gb": round(gpu_layers * mem_per_layer, 1),
        "total_weight_gb": round(weight_mem, 1),
        "estimated_tps": round(estimated_tps, 1),
        "context_length": context_length,
        "quant": quant,
        "runtime": runtime,
        "command": command,
        "penalty_note": f"~{penalty*100:.0f}% of full GPU speed ({gpu_pct:.0f}% layers on GPU)",
    }
