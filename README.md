# local-llm-forge

**Auto-optimize and deploy LLM models on local Apple Silicon.**

Given any HuggingFace model, forge analyzes your hardware, calculates memory budgets, selects the optimal quantization/runtime strategy, and deploys — all automatically.

## Why?

Running large LLMs locally is a puzzle: Which quantization fits my RAM? Will this 70B model even load? What's the fastest runtime for my chip?

**forge solves this.** Inspired by [flash-moe](https://github.com/danveloper/flash-moe) (which runs a 397B model on a MacBook), but designed to be **universal** — any model, any Apple Silicon Mac.

## Quick Start

```bash
# Setup
git clone https://github.com/koyounghun/local-llm-forge.git
cd local-llm-forge
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Analyze a model (no download)
forge analyze Qwen/Qwen2.5-7B-Instruct

# Check if a model can run on your hardware
forge route Qwen/Qwen2.5-72B-Instruct

# Download, quantize, and optimize automatically
forge optimize Qwen/Qwen2.5-7B-Instruct

# Generate text (with speculative decoding)
forge run optimized/Qwen--Qwen2.5-7B-Instruct-q4 "Explain recursion" --draft optimized/qwen-0.5b-draft

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
- **Speculative decoding**: 1.2x speedup with draft model (built into MLX)
- **KV cache quantization**: `--kv-bits 8` for 50% KV memory savings
- **Prompt caching**: Pre-compute KV cache for repeated system prompts
- **HQQ quantization**: 2-3 bit quantization without calibration data

### Benchmarks (M4 Pro, 48GB)

Qwen2.5-7B 4-bit, 150 tokens:

| Configuration | tok/s | vs Baseline |
|---------------|-------|-------------|
| Baseline (MLX int4) | 55.7 | 1.00x |
| + Speculative (0.5B draft) | 67.6 | 1.21x |
| + KV-bits 8 | 53.8 | ~1.0x (saves memory) |
| + Speculative + KV-bits 8 | 61.5 | 1.10x |

## CLI Reference

| Command | Description |
|---------|-------------|
| `forge analyze <model>` | Analyze model architecture + memory + ANE + routing |
| `forge route <model>` | Show all feasible execution paths |
| `forge optimize <model>` | Download, quantize, profile (auto or `--bits N`) |
| `forge run <model> "prompt"` | Generate with MLX engine (`--draft`, `--kv-bits`) |
| `forge deploy <model>` | Start OpenAI-compatible server |
| `forge bench <model>` | Run benchmark suite |
| `forge list` | List optimized models |
| `forge cache <model> -p "..."` | Cache system prompt KV |
| `forge cache-list <model>` | List cached prompts |

## Architecture

```
src/forge/
├── analyzer/          # Hardware + model + memory analysis
├── optimizer/         # Strategy selection, quantization, profiling
├── pipeline/          # Conversion, deployment, benchmarking
├── engine/            # MLX inference, speculative, KV cache, ANE, prompt cache
├── router/            # Feasibility routing, alternatives, offloading
├── metal/             # Custom Metal shader kernels
└── cli.py             # CLI entry point
```

## Supported Optimizations

| Technique | Source | Status |
|-----------|--------|--------|
| MLX native int4/int8 quantization | MLX | Working |
| HQQ 2-3 bit quantization | Dropbox HQQ | Working |
| Speculative decoding | MLX built-in | Working (1.2x speedup) |
| FP8 KV cache | MLX kv_bits | Working |
| Prompt KV caching | mlx-lm | Working |
| Model feasibility router | Original | Working |
| Disk offload estimation | llama.cpp | Estimation only |
| ANE hybrid (prefill) | ANEMLL/Orion | Assessment only |
| Custom Metal kernels | flash-moe inspired | Reference impl |
| Ollama integration | Ollama | Modelfile generation |

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
- Python 3.12+
- MLX installed (`pip install mlx mlx-lm`)

## License

MIT
