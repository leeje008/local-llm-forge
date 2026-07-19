# local-llm-forge

**Auto-optimize and deploy LLM models on local Apple Silicon.**

Given any HuggingFace model, forge analyzes your hardware, calculates memory budgets, selects the optimal quantization/runtime strategy, and deploys — all automatically.

## Why?

Running large LLMs locally is a puzzle: Which quantization fits my RAM? Will this 70B model even load? What's the fastest runtime for my chip?

**forge solves this.** Inspired by [flash-moe](https://github.com/danveloper/flash-moe) (which runs a 397B model on a MacBook), but designed to be **universal** — any model, any Apple Silicon Mac.

## Quick Start

```bash
# Setup
git clone https://github.com/leeje008/local-llm-forge.git
cd local-llm-forge
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Analyze a model (no download)
forge analyze Qwen/Qwen2.5-7B-Instruct

# Check if a model can run on your hardware
forge route Qwen/Qwen2.5-72B-Instruct

# Download, quantize, and optimize automatically
forge optimize Qwen/Qwen2.5-7B-Instruct

# Generate text (draft-model speculative decoding)
forge run optimized/Qwen--Qwen2.5-7B-Instruct-q4 "Explain recursion" --auto-draft

# Generate with N-gram self-speculation (no draft model needed)
forge run optimized/Qwen--Qwen2.5-7B-Instruct-q4 "Explain recursion" --ngram-spec

# Benchmark
forge bench optimized/Qwen--Qwen2.5-7B-Instruct-q4
```

## Features

### Model Feasibility Router

The core differentiator. When you specify a model, forge runs a **5-stage decision tree**:

```
[1] Full Precision (FP16)  — fits as-is?
[2] Quantized (int4/int3)  — fits with compression?
[3] Disk Offload           — partial GPU via llama.cpp?
[4] Smaller Alternative    — same family, smaller model?
[5] Cloud Fallback         — route to API?
```

```bash
$ forge route Qwen/Qwen2.5-72B-Instruct

Model Routing Analysis
============================================================
  Model:     Qwen/Qwen2.5-72B-Instruct (72.7B, Dense)
  Memory:    38GB available

  [1] Full Precision (FP16)            ✗  153GB needed
  [2] Quantized (int3, mixed_3_6)      ✓  ~31GB, 92% quality  ←
  [3] Disk Offload (59/80 GPU layers)  ✓  ~2 tok/s
  [4] Mixtral-8x7B (MoE, 12.9B active) ✓  ~15 tok/s
  [5] Cloud API Fallback               ✓
```

### Auto Optimization Pipeline

```
HuggingFace model → Analyze → Strategy → Convert → Quantize → Profile → Deploy
```

- **Hardware detection**: CPU, GPU cores, ANE TOPS, memory bandwidth, MLX/Ollama
- **Memory budget**: Per-quantization estimates with KV cache calculation
- **Strategy selection**: Automatically picks the best quantization + runtime
- **Profile-guided tuning**: Runs quick benchmark, adjusts parameters

### Inference Engine

- **MLX-native**: Zero-copy unified memory, highest throughput on Apple Silicon
- **Speculative decoding**: draft-model (`--draft`, `--auto-draft`, `--redrafter`) and
  **N-gram self-speculation** (`--ngram-spec`) with Adaptive-K draft length — the
  N-gram path runs a real custom decode loop, needs no extra model, and is
  output-identical to greedy decoding
- **KV cache optimization**: FP8 (`--kv-bits 8`), TurboQuant 3-bit compression
  (`--kv-compress turbo`), H2O / Ada-KV / LAVa eviction (`--kv-eviction`,
  `--lava-eviction`), xKV cross-layer SVD (`--xkv-rank`), radix prefix cache
  (`--prefix-cache`)
- **Attention backends**: default MLX or metal-flash-attention via `--attention mfa|auto`
- **Structured decoding**: grammar/JSON-Schema constrained generation
  (`--grammar`, backends: XGrammar-2 / llguidance / Outlines)
- **Prompt caching**: Pre-compute KV cache for repeated system prompts
- **Quantization methods**: MLX native, HQQ 2-3 bit, any4 LUT, GSR rotation,
  D2Quant dual-scale, KL-sensitivity mixed-precision, compound pipeline
- **MoE tooling**: expert importance analysis + pruning (`forge expert-prune`),
  MoE-SVD expert merging (`forge expert-merge`), per-expert asymmetric quantization
- **Serving**: chunked prefill (Sarathi), interruptible sessions (FastServe),
  multi-model LRU hot-loading, PMPD control plane (`forge deploy --chunked-prefill
  --interruptible --multi-model --pmpd`)
- **Long context**: DCA + chunked prefill feasibility analysis (`forge long-context`)

### Benchmarks (M4 Pro, 48GB)

Measured on Qwen2.5-7B 4-bit, 150 tokens (Phase 1–4 implementation):

| Configuration | tok/s | vs Baseline |
|---------------|-------|-------------|
| Baseline (MLX int4) | 55.7 | 1.00x |
| + Speculative (0.5B draft) | 67.6 | 1.21x |
| + KV-bits 8 | 53.8 | ~1.0x (saves memory) |
| + Speculative + KV-bits 8 | 61.5 | 1.10x |

> **Note**: The Phase 6–12 features listed above (TurboQuant, H2O/Ada-KV/LAVa,
> N-gram self-spec, CAS-Spec, mlx-mfa, structured decoding, MoE tooling, serving
> schedulers, …) are **implemented but not yet benchmarked** end-to-end. Numbers
> will be added after measurement; no paper-reported figures are quoted as results.

## CLI Reference

| Command | Description |
|---------|-------------|
| `forge analyze <model>` | Analyze model architecture + memory + ANE + routing |
| `forge route <model>` | Show all feasible execution paths |
| `forge optimize <model>` | Download, quantize, profile (`--bits`, `--method any4\|d2quant\|gsr\|optiq`, `--mixed-precision`, `--compound`, `--per-expert-quant`) |
| `forge run <model> "prompt"` | Generate (`--draft`, `--auto-draft`, `--redrafter`, `--ngram-spec`, `--cas-spec`, `--kv-bits`, `--kv-compress`, `--kv-eviction`, `--lava-eviction`, `--xkv-rank`, `--attention`, `--grammar`, `--prefix-cache`) |
| `forge deploy <model>` | Serve (`--chunked-prefill`, `--interruptible`, `--multi-model`, `--pmpd`, `--use-vllm-mlx`) |
| `forge bench <model>` | Run benchmark suite |
| `forge eval <model>` | Evaluate quality on standard suites |
| `forge profile <model>` | Token-level latency distribution |
| `forge sensitivity <model>` | Per-layer quantization sensitivity |
| `forge expert-prune <model>` | MoE expert importance + pruning plan (`--method aimer\|activation\|hybrid\|evolutionary`) |
| `forge expert-merge <model>` | MoE-SVD expert merging |
| `forge long-context <model>` | Long-context (DCA) feasibility analysis |
| `forge cache <model> -p "..."` | Cache system prompt KV |
| `forge cache-list <model>` | List cached prompts |
| `forge list` | List optimized models |

## Architecture

```
src/forge/
├── analyzer/            # Hardware + model + memory analysis
├── analysis/            # Sensitivity, latency profiling, expert & cache analysis
├── optimizer/           # Strategy selection, quantization, mixed precision,
│                        # expert pruning/merging, structural pruning
├── pipeline/            # Conversion, deployment, benchmarking, evaluation
├── engine/              # MLX inference engine
│   ├── speculative/     # Draft models, N-gram, PEARL, tree, Adaptive-K,
│   │                    # decode loop, OmniDraft, CAS-Spec/DyTC
│   ├── kv_cache/        # TurboQuant, H2O, Ada-KV, LAVa, xKV, estimation
│   ├── scheduler.py     # Chunked prefill, interruptible sessions, multi-model
│   ├── structured_decoding.py, long_context.py, attention_backend.py
│   ├── mla_attention.py, star_attention.py, mamba_engine.py, bitnet_engine.py
│   └── prefix_cache.py, prompt_cache.py, expert_cache.py, eagle.py, redrafter.py
├── router/              # Feasibility routing, alternatives, offloading
├── metal/               # Custom Metal shader kernels
└── cli.py               # CLI entry point
```

## Supported Optimizations

| Technique | Source | Status |
|-----------|--------|--------|
| MLX native int4/int8 quantization | MLX | Working |
| HQQ 2-3 bit quantization | Dropbox HQQ | Working |
| Draft-model speculative decoding | MLX built-in | Working (1.21x measured) |
| N-gram self-spec + Adaptive-K decode loop | lookup decoding | Implemented, tested (mock), not benchmarked |
| FP8 KV cache | MLX kv_bits | Working |
| TurboQuant / H2O / Ada-KV / LAVa / xKV | papers (2025) | Implemented, not benchmarked |
| PEARL / Tree (Sequoia) / CAS-Spec DyTC | papers (2024-25) | Implemented, not wired into decode loop |
| any4 / GSR / D2Quant / mixed-precision | papers (2024-25) | Implemented, not benchmarked |
| MoE expert prune / merge / cache | AIMER, MoE-SVD | Implemented, not benchmarked |
| Serving schedulers (Sarathi, FastServe, PMPD) | papers | Control plane implemented |
| Structured decoding (XGrammar-2 / llguidance) | grammar backends | Implemented, backend auto-detect |
| mlx-mfa attention backend | metal-flash-attention | Adapter implemented (optional dep) |
| Long-context DCA + chunked prefill | Qwen2.5-1M | Feasibility analysis implemented |
| Mamba / RWKV / BitNet / MLA / Star Attention | papers | Scaffolding + fallback paths |
| Model feasibility router | Original | Working |
| Disk offload estimation | llama.cpp | Estimation only |
| ANE hybrid (prefill) | ANEMLL/Orion | Assessment only |
| Ollama integration | Ollama | Modelfile generation |

## Testing

```bash
pip install -e ".[dev]"     # pytest, pytest-cov, ruff
pytest                      # unit tests (src/tests/, mock-based, no model download)
ruff check src/             # lint (0 errors maintained)
```

## Research

The `research/` folder contains detailed analysis:

- **flash-moe analysis**: Full teardown of the 397B-on-laptop project
- **Gap analysis**: 10 optimization techniques flash-moe doesn't implement
- **M4 Pro feasibility**: Verification of each technique on our hardware
- **Deep research**: Auto-pipeline design, quantization comparison, ANE hybrid
- **Feasibility report**: Final assessment with implementation tiers

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- 16GB+ unified memory (48GB recommended)
- Python 3.12+ (developed on 3.14)
- MLX installed (`pip install mlx mlx-lm`)

## License

MIT
