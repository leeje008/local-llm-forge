"""MoE Expert Pruning — REAP-inspired expert removal and merging.

Identifies low-activity experts via activation profiling, then removes
or merges them to reduce model size while maintaining quality.

Based on:
- REAP (2510.13999): Router-weighted activation pruning, 50% lossless
- Sub-MoE (2506.23266): Joint SVD merging, 96% quality at 25% pruning
- MoE-SVD (ICML 2025): 60% compression, 1.5x inference speedup
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExpertRanking:
    """Ranking of an expert by importance."""

    expert_id: int
    layer_idx: int
    activation_count: int
    activation_ratio: float  # fraction of tokens that activated this expert
    avg_gate_score: float
    importance_score: float  # combined metric (REAP-style)
    prune_candidate: bool


@dataclass
class PruningPlan:
    """Plan for which experts to prune/merge."""

    model_id: str
    total_experts: int
    experts_to_prune: int
    prune_ratio: float
    method: str  # "remove" | "merge"
    rankings: list[ExpertRanking] = field(default_factory=list)
    estimated_size_reduction_pct: float = 0.0
    estimated_quality_retention_pct: float = 0.0


def analyze_expert_importance(
    model_id: str,
    calibration_prompts: list[str] | None = None,
    num_samples: int = 100,
) -> list[ExpertRanking]:
    """Analyze expert importance using PyTorch forward hooks.

    Captures expert gate activations during calibration and ranks
    experts by combined frequency + gate score (REAP metric).

    Requires: torch, transformers
    """
    if calibration_prompts is None:
        calibration_prompts = _default_calibration_prompts()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return []

    # Load model on CPU (Apple Silicon unified memory)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    # Find MoE gate modules and register hooks
    gate_activations: dict[int, list[torch.Tensor]] = {}  # layer_idx → [gate_outputs]
    hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        # Try different MoE attribute names across architectures
        gate = None
        for attr in ["block_sparse_moe.gate", "mlp.gate", "moe.gate", "switch.gate"]:
            parts = attr.split(".")
            obj = layer
            try:
                for p in parts:
                    obj = getattr(obj, p)
                gate = obj
                break
            except AttributeError:
                continue

        if gate is not None:
            gate_activations[layer_idx] = []

            def make_hook(idx):
                def hook_fn(module, input, output):
                    gate_activations[idx].append(output.detach().cpu())
                return hook_fn

            hooks.append(gate.register_forward_hook(make_hook(layer_idx)))

    if not hooks:
        # Clean up
        for h in hooks:
            h.remove()
        return []

    # Run calibration prompts
    with torch.no_grad():
        for prompt in calibration_prompts[:num_samples]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            try:
                model(**inputs)
            except Exception:
                continue

    # Remove hooks
    for h in hooks:
        h.remove()

    # Analyze activations
    rankings = []
    for layer_idx, acts in gate_activations.items():
        if not acts:
            continue

        # Concatenate all gate outputs: (total_tokens, num_experts)
        all_gates = torch.cat(acts, dim=0)
        if all_gates.ndim == 3:
            all_gates = all_gates.reshape(-1, all_gates.shape[-1])

        num_experts = all_gates.shape[-1]
        total_tokens = all_gates.shape[0]

        # Top-k selection to find which experts were actually activated
        # Typically K=2 for Mixtral, K=4 for Qwen-MoE
        k = min(8, num_experts)  # upper bound
        topk_vals, topk_ids = torch.topk(all_gates, k=k, dim=-1)

        # Count activations per expert
        expert_counts = Counter()
        expert_gate_scores: dict[int, list[float]] = {i: [] for i in range(num_experts)}

        for token_idx in range(total_tokens):
            for j in range(k):
                eid = topk_ids[token_idx, j].item()
                score = topk_vals[token_idx, j].item()
                expert_counts[eid] += 1
                expert_gate_scores[eid].append(score)

        # Calculate importance (REAP-style: frequency × avg gate score)
        for eid in range(num_experts):
            count = expert_counts.get(eid, 0)
            ratio = count / (total_tokens * k) if total_tokens * k > 0 else 0
            avg_score = (
                sum(expert_gate_scores[eid]) / len(expert_gate_scores[eid])
                if expert_gate_scores[eid] else 0
            )
            importance = ratio * avg_score  # REAP combined metric

            rankings.append(ExpertRanking(
                expert_id=eid,
                layer_idx=layer_idx,
                activation_count=count,
                activation_ratio=ratio,
                avg_gate_score=avg_score,
                importance_score=importance,
                prune_candidate=False,
            ))

    # Sort by importance (lowest first = best prune candidates)
    rankings.sort(key=lambda r: r.importance_score)

    return rankings


def create_pruning_plan(
    model_id: str,
    rankings: list[ExpertRanking],
    prune_ratio: float = 0.25,
    method: str = "remove",
    num_experts: int = 8,
) -> PruningPlan:
    """Create a pruning plan from expert rankings.

    Args:
        rankings: Sorted by importance (lowest first)
        prune_ratio: Fraction of experts to prune (0.25 = 25%)
        method: "remove" (delete experts) or "merge" (combine similar)
    """
    experts_to_prune = max(1, int(num_experts * prune_ratio))

    # Mark bottom experts as prune candidates
    pruned = 0
    for r in rankings:
        if pruned < experts_to_prune:
            r.prune_candidate = True
            pruned += 1

    # Estimate quality retention (empirical from REAP paper)
    if prune_ratio <= 0.25:
        quality = 96.0  # Sub-MoE: 96% at 25%
    elif prune_ratio <= 0.50:
        quality = 90.0  # REAP: near-lossless at 50% for code tasks
    else:
        quality = 80.0 - (prune_ratio - 0.5) * 40

    # Size reduction = expert_params_fraction * prune_ratio
    # MoE expert params are typically 60-80% of total
    expert_param_fraction = 0.7
    size_reduction = expert_param_fraction * prune_ratio * 100

    return PruningPlan(
        model_id=model_id,
        total_experts=num_experts,
        experts_to_prune=experts_to_prune,
        prune_ratio=prune_ratio,
        method=method,
        rankings=rankings,
        estimated_size_reduction_pct=size_reduction,
        estimated_quality_retention_pct=quality,
    )


def _default_calibration_prompts() -> list[str]:
    """Default calibration prompts covering diverse domains."""
    return [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the main causes of climate change?",
        "Translate this to French: The weather is beautiful today.",
        "Solve: If a train travels 120km in 2 hours, what is its speed?",
        "Write a SQL query to find the top 5 customers by revenue.",
        "Explain how photosynthesis works step by step.",
        "What is the difference between TCP and UDP protocols?",
        "Write a haiku about artificial intelligence.",
        "Describe the process of cellular respiration.",
        "How does a neural network learn from data?",
        "What are the key principles of object-oriented programming?",
        "Explain quantum entanglement to a 10-year-old.",
        "Write a bash script to find large files in a directory.",
        "What were the main events of World War II?",
        "How does the human immune system fight viruses?",
        "Write a regular expression to match email addresses.",
        "Explain the concept of blockchain technology.",
        "What is the time complexity of merge sort and why?",
        "Describe the water cycle in detail.",
    ]


def format_pruning_plan(plan: PruningPlan) -> str:
    """Format pruning plan for display."""
    lines = [
        "Expert Pruning Plan",
        "=" * 55,
        f"  Model:          {plan.model_id}",
        f"  Method:         {plan.method}",
        f"  Total Experts:  {plan.total_experts}",
        f"  Prune Count:    {plan.experts_to_prune} ({plan.prune_ratio:.0%})",
        f"  Est. Size Red:  {plan.estimated_size_reduction_pct:.1f}%",
        f"  Est. Quality:   {plan.estimated_quality_retention_pct:.0f}%",
        "",
    ]

    # Show prune candidates
    candidates = [r for r in plan.rankings if r.prune_candidate]
    if candidates:
        lines.append("  Prune Candidates (lowest importance):")
        lines.append(f"  {'Layer':>6} {'Expert':>7} {'Freq':>8} {'Gate':>8} {'Score':>8}")
        lines.append(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
        for r in candidates[:10]:
            lines.append(
                f"  {r.layer_idx:>6} {r.expert_id:>7} {r.activation_ratio:>7.3f} "
                f"{r.avg_gate_score:>7.3f} {r.importance_score:>7.4f}"
            )

    # Show top retained experts
    retained = [r for r in plan.rankings if not r.prune_candidate]
    if retained:
        retained.sort(key=lambda r: -r.importance_score)
        lines.append("")
        lines.append("  Top Retained Experts (highest importance):")
        for r in retained[:5]:
            lines.append(
                f"    Layer {r.layer_idx}, Expert {r.expert_id}: "
                f"freq={r.activation_ratio:.3f}, score={r.importance_score:.4f}"
            )

    return "\n".join(lines)


def save_pruning_plan(plan: PruningPlan, path: Path) -> None:
    """Save pruning plan to JSON for later application."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_id": plan.model_id,
        "method": plan.method,
        "prune_ratio": plan.prune_ratio,
        "experts_to_prune": plan.experts_to_prune,
        "prune_candidates": [
            {"layer": r.layer_idx, "expert": r.expert_id, "score": r.importance_score}
            for r in plan.rankings if r.prune_candidate
        ],
        "estimated_size_reduction_pct": plan.estimated_size_reduction_pct,
        "estimated_quality_retention_pct": plan.estimated_quality_retention_pct,
    }
    path.write_text(json.dumps(data, indent=2))
