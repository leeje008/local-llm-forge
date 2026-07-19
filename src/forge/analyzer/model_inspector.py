"""HuggingFace model architecture inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelProfile:
    """Extracted model architecture metadata."""

    model_id: str = ""
    architecture: str = ""  # "llama", "qwen2", "mistral", "mixtral", ...
    model_type: str = "dense"  # "dense" | "moe"
    # Phase 11: transformer | mamba | hybrid-mamba | rwkv7 | mla | bitnet
    architecture_family: str = "transformer"
    total_params_b: float = 0.0  # billions
    num_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    attention_type: str = "MHA"  # "MHA" | "GQA" | "MQA"
    head_dim: int = 0
    vocab_size: int = 0
    max_context: int = 0
    # MoE fields
    num_experts: int | None = None
    num_active_experts: int | None = None
    shared_experts: int | None = None
    # Extra
    torch_dtype: str = ""
    config_raw: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _estimate_params_dense(p: ModelProfile) -> float:
    """Rough parameter count estimate for dense transformer (in billions)."""
    h = p.hidden_size
    i = p.intermediate_size or int(h * 8 / 3)  # SwiGLU default
    n = p.num_layers
    v = p.vocab_size

    # Embedding + LM head
    embed = 2 * v * h
    # Per layer: attention (QKV + O) + FFN (gate + up + down) + norms
    kv_heads = p.num_kv_heads or p.num_attention_heads
    qkv = h * (h + 2 * kv_heads * p.head_dim)
    o_proj = h * h
    ffn = 3 * h * i  # gate + up + down
    norm = 2 * h  # 2 RMSNorm per layer
    per_layer = qkv + o_proj + ffn + norm
    total = embed + n * per_layer
    return total / 1e9


def _estimate_params_moe(p: ModelProfile) -> float:
    """Rough parameter count estimate for MoE transformer (in billions)."""
    h = p.hidden_size
    i = p.intermediate_size or int(h * 8 / 3)
    n = p.num_layers
    v = p.vocab_size
    ne = p.num_experts or 8

    embed = 2 * v * h
    kv_heads = p.num_kv_heads or p.num_attention_heads
    qkv = h * (h + 2 * kv_heads * p.head_dim)
    o_proj = h * h
    attn_per_layer = qkv + o_proj

    # Each expert has its own FFN
    ffn_per_expert = 3 * h * i
    shared = (p.shared_experts or 0) * ffn_per_expert
    router = h * ne  # routing gate
    ffn_total = ne * ffn_per_expert + shared + router

    norm = 2 * h
    per_layer = attn_per_layer + ffn_total + norm
    total = embed + n * per_layer
    return total / 1e9


def _detect_architecture_family(raw: dict, architecture: str) -> str:
    """Classify a HF config into a high-level architecture family.

    Returns one of:
      - "transformer"  (default dense / MoE transformer)
      - "mamba"        (pure state-space, e.g. mamba, mamba2)
      - "hybrid-mamba" (interleaved SSM+attention, e.g. Granite 4.0-H, Jamba, Zamba)
      - "rwkv7"        (RWKV-4/5/6/7 linear recurrence)
      - "mla"          (DeepSeek-V2/V3 multi-head latent attention)
      - "bitnet"       (BitNet b1.58 ternary weights)
    """
    arch = (architecture or "").lower()
    model_type = str(raw.get("model_type", "")).lower()
    archs_list = [str(a).lower() for a in raw.get("architectures", [])]
    combined = " ".join([arch, model_type, *archs_list])

    # BitNet
    if "bitnet" in combined or raw.get("quantization_config", {}).get("quant_method") == "bitnet":
        return "bitnet"

    # RWKV (all generations)
    if "rwkv" in combined:
        return "rwkv7"

    # DeepSeek MLA — presence of kv_lora_rank is the definitive signal
    if (
        "kv_lora_rank" in raw
        or "q_lora_rank" in raw
        or "deepseek_v2" in combined
        or "deepseek_v3" in combined
    ):
        return "mla"

    # Hybrid Mamba / SSM+Attention
    if any(k in combined for k in ("granite", "jamba", "zamba", "hybrid")):
        if "layer_types" in raw or "block_types" in raw or "attn_layer_indices" in raw:
            return "hybrid-mamba"
        # Granite 4.0-H, Jamba, Zamba2 explicitly identify as hybrid even
        # without a layer_types list.
        if "granite" in combined and ("mamba" in combined or raw.get("mamba_d_state")):
            return "hybrid-mamba"
        if "jamba" in combined or "zamba" in combined:
            return "hybrid-mamba"

    # Pure Mamba
    if (
        "mamba" in combined
        or raw.get("mamba_d_state") is not None
        or raw.get("state_size") is not None
    ):
        # Distinguish pure vs hybrid by the absence of attention heads.
        if raw.get("num_attention_heads", 0) == 0:
            return "mamba"
        return "hybrid-mamba"

    return "transformer"


def _detect_attention_type(num_heads: int, num_kv_heads: int) -> str:
    if num_kv_heads == 0 or num_kv_heads == num_heads:
        return "MHA"
    elif num_kv_heads == 1:
        return "MQA"
    else:
        return "GQA"


def inspect(model_id: str, trust_remote_code: bool = False) -> ModelProfile:
    """Inspect a HuggingFace model and return its architecture profile.

    Requires `transformers` to be installed. Downloads only the config
    (not the full model weights).
    """
    from transformers import AutoConfig  # type: ignore[import-untyped]

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    raw = config.to_dict()

    p = ModelProfile()
    p.model_id = model_id
    p.config_raw = raw

    # Architecture name
    architectures = raw.get("architectures", [])
    p.architecture = raw.get("model_type", architectures[0] if architectures else "unknown")

    # Basic dimensions
    p.num_layers = raw.get("num_hidden_layers", 0)
    p.hidden_size = raw.get("hidden_size", 0)
    p.intermediate_size = raw.get("intermediate_size", 0)
    p.num_attention_heads = raw.get("num_attention_heads", 0)
    p.vocab_size = raw.get("vocab_size", 0)
    p.torch_dtype = raw.get("torch_dtype", "")

    # KV heads (GQA/MQA detection)
    p.num_kv_heads = raw.get(
        "num_key_value_heads",
        raw.get("num_kv_heads", p.num_attention_heads),
    )

    # Head dimension
    raw_head_dim = raw.get("head_dim")
    if raw_head_dim:
        p.head_dim = raw_head_dim
    elif p.num_attention_heads > 0 and p.hidden_size > 0:
        p.head_dim = p.hidden_size // p.num_attention_heads
    else:
        p.head_dim = 0

    # Context length
    p.max_context = raw.get(
        "max_position_embeddings",
        raw.get("sliding_window", raw.get("max_sequence_length", 0)),
    )

    # Attention type
    p.attention_type = _detect_attention_type(p.num_attention_heads, p.num_kv_heads)

    # MoE detection
    num_experts = (
        raw.get("num_experts") or raw.get("num_local_experts") or raw.get("n_routed_experts")
    )
    if num_experts:
        p.model_type = "moe"
        p.num_experts = num_experts
        p.num_active_experts = (
            raw.get("num_experts_per_tok")
            or raw.get("num_selected_experts")
            or raw.get("num_activated_experts")
            or raw.get("topk", 2)
        )
        p.shared_experts = raw.get("n_shared_experts", 0) or raw.get("num_shared_experts", 0)
    else:
        p.model_type = "dense"

    # Phase 11: architecture family detection (Mamba / RWKV / MLA / BitNet / hybrid)
    p.architecture_family = _detect_architecture_family(raw, p.architecture)

    # Parameter estimation
    if p.model_type == "moe":
        p.total_params_b = _estimate_params_moe(p)
    else:
        p.total_params_b = _estimate_params_dense(p)

    # Sanity check: some configs have a direct param count
    if "num_parameters" in raw:
        p.total_params_b = raw["num_parameters"] / 1e9

    return p


def inspect_local(config_path: str | Path) -> ModelProfile:
    """Inspect a local model config.json file."""
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.json"

    with open(path) as f:
        raw = json.load(f)

    # Create a minimal profile from raw config
    p = ModelProfile()
    p.model_id = str(path.parent)
    p.config_raw = raw
    p.architecture = raw.get("model_type", "unknown")
    p.num_layers = raw.get("num_hidden_layers", 0)
    p.hidden_size = raw.get("hidden_size", 0)
    p.intermediate_size = raw.get("intermediate_size", 0)
    p.num_attention_heads = raw.get("num_attention_heads", 0)
    p.num_kv_heads = raw.get("num_key_value_heads", raw.get("num_kv_heads", p.num_attention_heads))
    p.vocab_size = raw.get("vocab_size", 0)
    p.max_context = raw.get("max_position_embeddings", 0)
    if p.num_attention_heads > 0 and p.hidden_size > 0:
        p.head_dim = raw.get("head_dim", p.hidden_size // p.num_attention_heads)
    p.attention_type = _detect_attention_type(p.num_attention_heads, p.num_kv_heads)

    num_experts = (
        raw.get("num_experts") or raw.get("num_local_experts") or raw.get("n_routed_experts")
    )
    if num_experts:
        p.model_type = "moe"
        p.num_experts = num_experts
        p.num_active_experts = (
            raw.get("num_experts_per_tok") or raw.get("num_selected_experts") or 2
        )
        p.shared_experts = raw.get("n_shared_experts", 0)
        p.total_params_b = _estimate_params_moe(p)
    else:
        p.model_type = "dense"
        p.total_params_b = _estimate_params_dense(p)

    p.architecture_family = _detect_architecture_family(raw, p.architecture)
    return p


def format_report(m: ModelProfile) -> str:
    """Format a human-readable model profile report."""
    lines = [
        "Model Profile",
        "=" * 50,
        f"  Model:           {m.model_id}",
        f"  Architecture:    {m.architecture}",
        f"  Type:            {m.model_type.upper()}",
        f"  Parameters:      {m.total_params_b:.1f}B",
        f"  Layers:          {m.num_layers}",
        f"  Hidden Size:     {m.hidden_size}",
        f"  Attention:       {m.attention_type} "
        f"({m.num_attention_heads} heads, {m.num_kv_heads} KV heads)",
        f"  Head Dim:        {m.head_dim}",
        f"  Vocab Size:      {m.vocab_size:,}",
        f"  Max Context:     {m.max_context:,}",
    ]
    if m.model_type == "moe":
        lines.extend([
            f"  Experts:         {m.num_experts} total, {m.num_active_experts} active/token",
            f"  Shared Experts:  {m.shared_experts or 0}",
        ])
    if m.torch_dtype:
        lines.append(f"  Dtype:           {m.torch_dtype}")
    return "\n".join(lines)
