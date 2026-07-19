"""MoE Expert Pruning — REAP-inspired expert removal and merging.

Identifies low-activity experts via activation profiling, then removes
or merges them to reduce model size while maintaining quality.

Based on:
- REAP (2510.13999): Router-weighted activation pruning, 50% lossless
- Sub-MoE (2506.23266): Joint SVD merging, 96% quality at 25% pruning
- MoE-SVD (ICML 2025): 60% compression, 1.5x inference speedup
- AIMER (2603.18492): Calibration-free weight-norm pruning (~0.2s)
- EvoESAP (2603.06003): Evolutionary per-layer non-uniform pruning
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
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
    method: str = "activation",
) -> list[ExpertRanking]:
    """Analyze expert importance with configurable scoring method.

    Args:
        method: One of:
            - "activation" (default, backcompat): REAP-style activation
              profiling via forward hooks. Slow (~minutes) but accurate.
            - "aimer": Calibration-free weight-RMSE scoring (~0.2s).
              Loads weights only, no forward pass.
            - "hybrid": AIMER fast pass followed by activation refinement
              on the bottom 2x prune-candidates. Compromise accuracy/speed.

    Returns a list[ExpertRanking] sorted by importance (lowest first =
    best prune candidates). Downstream `create_pruning_plan` consumes
    this identically regardless of which method produced it.
    """
    if method == "aimer":
        return score_experts_aimer(model_id)
    if method == "hybrid":
        aimer = score_experts_aimer(model_id)
        # If AIMER fails, fall back to activation
        if not aimer:
            return _analyze_expert_importance_activation(
                model_id, calibration_prompts, num_samples,
            )
        # Refine: run activation profiling, then blend scores
        act = _analyze_expert_importance_activation(
            model_id, calibration_prompts, num_samples,
        )
        if not act:
            return aimer
        # Blend: normalize each source to [0,1] and average
        return _blend_rankings(aimer, act)
    return _analyze_expert_importance_activation(
        model_id, calibration_prompts, num_samples,
    )


def _analyze_expert_importance_activation(
    model_id: str,
    calibration_prompts: list[str] | None = None,
    num_samples: int = 100,
) -> list[ExpertRanking]:
    """Activation-hook-based expert importance (REAP original path)."""
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


# ---------------------------------------------------------------------------
# Phase 9.1 — AIMER: Calibration-free weight-RMSE expert importance
# ---------------------------------------------------------------------------


def score_experts_aimer(model_id: str) -> list[ExpertRanking]:
    """AIMER-style calibration-free expert ranking (paper 2603.18492).

    Ranks each MoE expert by an output-space RMSE proxy computed directly
    from its weight tensors. The original AIMER estimate is

        importance(E) ≈ ||W2_E||_F   (Frobenius norm of the down-projection)

    which is the expected L2 magnitude of the expert's contribution to
    the residual stream under an isotropic input assumption. We also fold
    in the gate/up projection norms as a tie-breaker, producing a more
    stable ranking on extremely sparse experts. Runs in ~0.2s for a
    Mixtral-8x7B because no forward pass is required.

    The returned list is drop-in compatible with `create_pruning_plan`.
    """
    try:
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load  # type: ignore[import-untyped]
    except ImportError:
        return []

    try:
        model, _ = load(model_id)
    except Exception:
        return []

    # Collect (layer_idx, expert_id) -> per-projection norms
    w2_norms: dict[tuple[int, int], float] = {}
    w_up_norms: dict[tuple[int, int], float] = defaultdict(float)

    try:
        params = model.parameters()
    except Exception:
        return []

    # Recursively walk the parameter tree to get flat name → tensor
    flat: dict[str, "object"] = {}

    def _walk(prefix: str, node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _walk(f"{prefix}.{i}", v)
        else:
            flat[prefix] = node

    _walk("", params)

    for name, tensor in flat.items():
        # Heuristic: names containing both "layers.<N>" and "experts.<M>"
        # with a "down_proj"/"w2" suffix are the W2 matrices.
        lower = name.lower()
        if "expert" not in lower:
            continue

        layer_idx = _extract_int_after(lower, "layers.")
        expert_id = _extract_int_after(lower, "experts.")
        if layer_idx is None or expert_id is None:
            continue

        try:
            arr = np.asarray(mx.array(tensor).astype(mx.float32))
        except Exception:
            continue
        if arr.ndim < 2:
            continue

        fro = float(np.linalg.norm(arr))
        key = (layer_idx, expert_id)

        if "down_proj" in lower or ".w2" in lower or "o_proj" in lower:
            # output projection → AIMER primary signal
            w2_norms[key] = fro
        else:
            # gate/up projections → tie-breaker aggregate
            w_up_norms[key] += fro

    if not w2_norms:
        return []

    # Count experts per layer to fill `num_experts`
    experts_by_layer: dict[int, set[int]] = defaultdict(set)
    for (L, E) in w2_norms.keys():
        experts_by_layer[L].add(E)

    rankings: list[ExpertRanking] = []
    for (layer_idx, expert_id), w2 in w2_norms.items():
        up = w_up_norms.get((layer_idx, expert_id), 0.0)
        # Importance = W2 Frobenius + 0.1 * (gate+up) norms, as per AIMER
        # tie-break formulation. Scale down the up-norm because it does
        # not directly appear in the residual contribution estimate.
        importance = w2 + 0.1 * up

        rankings.append(ExpertRanking(
            expert_id=expert_id,
            layer_idx=layer_idx,
            activation_count=0,           # not measured in AIMER
            activation_ratio=0.0,         # not measured in AIMER
            avg_gate_score=float(w2),     # store raw W2 norm for inspection
            importance_score=float(importance),
            prune_candidate=False,
        ))

    rankings.sort(key=lambda r: r.importance_score)
    return rankings


def _extract_int_after(name: str, marker: str) -> int | None:
    """Return the first integer token following `marker` in `name`.

    Example: `_extract_int_after("model.layers.12.experts.3.w2", "layers.")`
    returns 12. Returns None if the marker is absent or no integer follows.
    """
    idx = name.find(marker)
    if idx < 0:
        return None
    j = idx + len(marker)
    k = j
    while k < len(name) and name[k].isdigit():
        k += 1
    if k == j:
        return None
    try:
        return int(name[j:k])
    except ValueError:
        return None


def _blend_rankings(
    a: list[ExpertRanking],
    b: list[ExpertRanking],
) -> list[ExpertRanking]:
    """Blend two ExpertRanking lists by normalized importance average."""
    if not a or not b:
        return a or b

    def _norm(rs: list[ExpertRanking]) -> dict[tuple[int, int], float]:
        vals = [r.importance_score for r in rs]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return {
            (r.layer_idx, r.expert_id): (r.importance_score - lo) / rng
            for r in rs
        }

    na = _norm(a)
    nb = _norm(b)
    merged: list[ExpertRanking] = []
    # Prefer the activation source as the base (it has true activation stats)
    base = {(r.layer_idx, r.expert_id): r for r in b}
    for key, r in base.items():
        blended = 0.5 * na.get(key, nb[key]) + 0.5 * nb[key]
        merged.append(ExpertRanking(
            expert_id=r.expert_id,
            layer_idx=r.layer_idx,
            activation_count=r.activation_count,
            activation_ratio=r.activation_ratio,
            avg_gate_score=r.avg_gate_score,
            importance_score=float(blended),
            prune_candidate=False,
        ))
    merged.sort(key=lambda r: r.importance_score)
    return merged


# ---------------------------------------------------------------------------
# Phase 9.2 — EvoESAP: Evolutionary non-uniform per-layer pruning
# ---------------------------------------------------------------------------


@dataclass
class LayerwisePruningPlan:
    """Non-uniform per-layer expert pruning plan (EvoESAP)."""

    model_id: str
    num_layers: int
    num_experts_per_layer: int
    per_layer_ratio: list[float] = field(default_factory=list)
    target_avg_ratio: float = 0.25
    achieved_avg_ratio: float = 0.0
    fitness: float = 0.0
    generations: int = 0
    population_size: int = 0
    estimated_size_reduction_pct: float = 0.0
    estimated_quality_retention_pct: float = 0.0


def evolutionary_layer_pruning(
    rankings: list[ExpertRanking],
    target_ratio: float = 0.25,
    population_size: int = 20,
    generations: int = 10,
    model_id: str = "",
    num_experts_per_layer: int = 8,
    seed: int = 7,
) -> LayerwisePruningPlan:
    """Per-layer optimal prune ratio via a small genetic algorithm.

    Each chromosome is a vector of per-layer prune ratios in [0, 0.75].
    Fitness balances two surrogates (no eval runs required):

        fitness = quality_retention - size_penalty
        quality_retention ≈ sum_per_layer( mean_importance_kept )
        size_penalty      ≈ |achieved_avg_ratio - target_ratio|

    Quality retention uses the per-layer mean importance score of the
    experts that would be *kept* at each candidate ratio — a cheap but
    informative surrogate for downstream accuracy.

    Returns a `LayerwisePruningPlan` whose `per_layer_ratio` is the best
    chromosome found after `generations` rounds.
    """
    if not rankings:
        return LayerwisePruningPlan(
            model_id=model_id,
            num_layers=0,
            num_experts_per_layer=num_experts_per_layer,
            per_layer_ratio=[],
            target_avg_ratio=target_ratio,
        )

    # Group rankings by layer, sorted ascending by importance
    per_layer: dict[int, list[ExpertRanking]] = defaultdict(list)
    for r in rankings:
        per_layer[r.layer_idx].append(r)
    for L in per_layer.values():
        L.sort(key=lambda r: r.importance_score)

    layer_ids = sorted(per_layer.keys())
    num_layers = len(layer_ids)
    if num_layers == 0:
        return LayerwisePruningPlan(
            model_id=model_id,
            num_layers=0,
            num_experts_per_layer=num_experts_per_layer,
            per_layer_ratio=[],
            target_avg_ratio=target_ratio,
        )

    rng = random.Random(seed)

    def _fitness(chrom: list[float]) -> float:
        retained_sum = 0.0
        total_sum = 0.0
        achieved_ratios = []
        for i, L in enumerate(layer_ids):
            experts = per_layer[L]
            n = len(experts)
            ratio = max(0.0, min(0.75, chrom[i]))
            n_prune = int(round(n * ratio))
            kept = experts[n_prune:]
            if kept:
                retained_sum += sum(r.importance_score for r in kept)
            total_sum += sum(r.importance_score for r in experts)
            achieved_ratios.append(n_prune / n if n else 0.0)
        quality = retained_sum / total_sum if total_sum > 0 else 0.0
        achieved_avg = sum(achieved_ratios) / len(achieved_ratios)
        size_penalty = abs(achieved_avg - target_ratio)
        return quality - 0.5 * size_penalty

    # Initial population: jittered around the uniform ratio
    def _random_chrom() -> list[float]:
        return [
            max(0.0, min(0.75, target_ratio + rng.uniform(-0.1, 0.1)))
            for _ in range(num_layers)
        ]

    population: list[list[float]] = [
        [target_ratio] * num_layers,  # seed with the uniform plan
    ]
    while len(population) < population_size:
        population.append(_random_chrom())

    best_chrom = population[0]
    best_fit = _fitness(best_chrom)

    for _gen in range(generations):
        scored = [(ch, _fitness(ch)) for ch in population]
        scored.sort(key=lambda t: -t[1])
        if scored[0][1] > best_fit:
            best_fit = scored[0][1]
            best_chrom = list(scored[0][0])

        # Elitism: top 25% carry over
        elite_n = max(2, population_size // 4)
        elites = [ch for ch, _ in scored[:elite_n]]

        # Produce offspring via single-point crossover + gaussian mutation
        offspring: list[list[float]] = list(elites)
        while len(offspring) < population_size:
            p1 = rng.choice(elites)
            p2 = rng.choice(elites)
            cut = rng.randint(1, max(1, num_layers - 1))
            child = p1[:cut] + p2[cut:]
            # Mutation
            for i in range(num_layers):
                if rng.random() < 0.15:
                    child[i] = max(0.0, min(0.75, child[i] + rng.gauss(0, 0.05)))
            offspring.append(child)
        population = offspring

    # Compute final stats
    achieved_avg = sum(best_chrom) / len(best_chrom) if best_chrom else 0.0
    # Empirical quality / size estimates mirror create_pruning_plan
    if achieved_avg <= 0.25:
        quality_pct = 96.0
    elif achieved_avg <= 0.5:
        quality_pct = 90.0
    else:
        quality_pct = 80.0 - (achieved_avg - 0.5) * 40
    size_pct = 0.7 * achieved_avg * 100

    return LayerwisePruningPlan(
        model_id=model_id,
        num_layers=num_layers,
        num_experts_per_layer=num_experts_per_layer,
        per_layer_ratio=list(best_chrom),
        target_avg_ratio=target_ratio,
        achieved_avg_ratio=achieved_avg,
        fitness=best_fit,
        generations=generations,
        population_size=population_size,
        estimated_size_reduction_pct=size_pct,
        estimated_quality_retention_pct=quality_pct,
    )


def format_layerwise_plan(plan: LayerwisePruningPlan) -> str:
    """Format a LayerwisePruningPlan for display."""
    lines = [
        "EvoESAP Layer-wise Pruning Plan",
        "=" * 55,
        f"  Model:           {plan.model_id}",
        f"  Layers:          {plan.num_layers}",
        f"  Target ratio:    {plan.target_avg_ratio:.0%}",
        f"  Achieved ratio:  {plan.achieved_avg_ratio:.2%}",
        f"  Fitness:         {plan.fitness:.4f}",
        f"  GA: pop={plan.population_size}, gen={plan.generations}",
        f"  Est. Size Red:   {plan.estimated_size_reduction_pct:.1f}%",
        f"  Est. Quality:    {plan.estimated_quality_retention_pct:.0f}%",
        "",
        "  Per-layer prune ratios:",
    ]
    for i, r in enumerate(plan.per_layer_ratio):
        bar = "#" * int(r * 40)
        lines.append(f"    L{i:>3}  {r:>5.1%}  {bar}")
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
