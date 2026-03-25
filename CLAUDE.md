# local-llm-forge

## Project Overview
로컬 Apple Silicon 환경에서 LLM 모델을 자동으로 분석, 최적화, 배포하는 CLI 도구.
사용자가 HuggingFace model_id를 지정하면 하드웨어를 분석하고 최적의 양자화/추론 전략을 결정하여 배포 파이프라인을 자동 구축한다.

## Target Hardware
- Apple M4 Pro (14 CPU, 20 GPU, Metal 4, ANE 38 TOPS)
- 48GB Unified Memory
- macOS 26.3.1

## Tech Stack
- **Language**: Python 3.14
- **ML Framework**: MLX 0.31.0 (primary), llama.cpp/Ollama (fallback)
- **Key Libraries**: mlx-lm, transformers, click, pyyaml, huggingface-hub
- **Optional**: hqq (advanced quantization), coremltools (ANE)

## Project Structure
```
src/forge/
├── cli.py                  # CLI entry point (click-based)
├── analyzer/               # Model & hardware analysis
│   ├── model_inspector.py  # HuggingFace model metadata extraction
│   ├── memory_calculator.py # Memory budget calculation
│   └── hardware_profiler.py # Local hardware detection
├── optimizer/              # Optimization strategy
│   ├── strategy_selector.py # Auto strategy selection
│   ├── quantizer.py        # Quantization pipeline
│   └── profiler.py         # Profile-based tuning
├── pipeline/               # Build & deploy
│   ├── converter.py        # Format conversion (MLX/GGUF)
│   ├── deployer.py         # Serving deployment
│   └── benchmarker.py      # Performance benchmarks
├── engine/                 # Inference engine optimizations
│   ├── mlx_engine.py       # MLX-based inference
│   ├── kv_cache.py         # KV cache management
│   ├── speculative.py      # Speculative decoding
│   └── attention.py        # Attention optimizations
└── metal/                  # Custom Metal kernels
    └── kernels.metal
```

## CLI Commands
```bash
forge analyze <model_id>     # Analyze model + estimate memory
forge optimize <model_id>    # Download, convert, quantize, profile
forge deploy <path>          # Start serving
forge bench <path>           # Run benchmarks
forge list                   # List optimized models
```

## Key Design Decisions
1. **MLX-first**: Use MLX as primary runtime for Apple Silicon optimization
2. **Auto-everything**: Model analysis, quantization selection, parameter tuning are all automatic
3. **Fallback chain**: MLX → GGUF → Ollama for maximum model compatibility
4. **Profile-guided**: Short benchmark → auto-adjust parameters → final config

## Coding Conventions
- Python 3.14+ features allowed
- Type hints required on all public functions
- Dataclasses for structured data (ModelProfile, HardwareProfile, etc.)
- Click for CLI framework
- YAML for configuration files

## Build & Run
```bash
# Use project venv (already created at .venv/)
source .venv/bin/activate
pip install -e ".[dev]"
forge analyze Qwen/Qwen2.5-7B-Instruct
```

## Testing
```bash
pytest src/tests/
```

## Reference
- flash-moe: https://github.com/danveloper/flash-moe (inspiration)
- Research docs: see research/ folder
- Design docs: see docs/ folder
