# local-llm-forge 시스템 아키텍처

## 1. 시스템 개요

local-llm-forge는 LLM 모델을 로컬 Apple Silicon 환경에 맞게 자동으로 분석, 최적화, 배포하는 CLI 도구이다.

```
┌─────────────────────────────────────────────────┐
│                  forge CLI                       │
│  analyze │ optimize │ deploy │ bench             │
└────┬─────────┬──────────┬──────────┬────────────┘
     │         │          │          │
     ▼         ▼          ▼          ▼
┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
│ Analyzer│ │ Optimizer│ │Pipeline│ │Benchmarker│
└────┬────┘ └────┬─────┘ └───┬────┘ └─────┬────┘
     │           │            │            │
     ▼           ▼            ▼            ▼
┌──────────────────────────────────────────────────┐
│                  Engine Layer                      │
│  MLX Engine │ KV Cache │ Speculative │ Attention  │
└──────────────────────────────────────────────────┘
     │                    │
     ▼                    ▼
┌──────────┐      ┌─────────────┐
│ MLX/Metal│      │ External    │
│ Runtime  │      │ Ollama/GGUF │
└──────────┘      └─────────────┘
```

## 2. 핵심 모듈

### 2.1 Analyzer (분석기)

모델과 하드웨어를 분석하여 최적화 전략의 입력 데이터를 생성한다.

| 컴포넌트 | 역할 | 입력 | 출력 |
|---------|------|------|------|
| `model_inspector` | 모델 아키텍처 감지 | HF model_id | ModelProfile |
| `memory_calculator` | 메모리 예산 계산 | ModelProfile + HWProfile | MemoryBudget |
| `hardware_profiler` | 하드웨어 스펙 감지 | (시스템) | HardwareProfile |

**ModelProfile**:
```python
@dataclass
class ModelProfile:
    model_id: str
    architecture: str          # "llama", "qwen", "mistral", ...
    model_type: str            # "dense" | "moe"
    total_params: int          # 전체 파라미터 수
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int          # GQA/MQA 판별
    attention_type: str        # "MHA" | "GQA" | "MQA"
    vocab_size: int
    max_context: int
    # MoE 전용
    num_experts: int | None
    num_active_experts: int | None
```

**HardwareProfile**:
```python
@dataclass
class HardwareProfile:
    chip: str                  # "Apple M4 Pro"
    cpu_cores: int             # 14
    gpu_cores: int             # 20
    ane_tops: float            # 38.0
    total_memory_gb: float     # 48.0
    memory_bandwidth_gbs: float # 273.0
    disk_available_gb: float
    has_mlx: bool
    mlx_version: str | None
    has_ollama: bool
    metal_version: int         # 4
```

### 2.2 Optimizer (최적화기)

분석 결과를 바탕으로 최적의 전략을 결정한다.

| 컴포넌트 | 역할 |
|---------|------|
| `strategy_selector` | 양자화/포맷/런타임/추가최적화 자동 결정 |
| `quantizer` | 양자화 파이프라인 (MLX int4 → HQQ → AQLM) |
| `profiler` | 프로파일 기반 파라미터 자동 조정 |

**OptimizationStrategy**:
```python
@dataclass
class OptimizationStrategy:
    quantization: str          # "fp16", "int8", "int4", "int3", "int2"
    quant_method: str          # "mlx_native", "hqq", "aqlm"
    format: str                # "mlx", "gguf"
    runtime: str               # "mlx-lm", "ollama", "llama.cpp"
    context_length: int
    batch_size: int
    use_speculative: bool
    draft_model: str | None
    use_prompt_cache: bool
    expert_cache_size: int | None  # MoE only
    estimated_tps: float       # 예상 tok/s
    estimated_memory_gb: float
```

### 2.3 Pipeline (파이프라인)

전략에 따라 모델을 변환, 양자화, 배포한다.

| 컴포넌트 | 역할 |
|---------|------|
| `converter` | SafeTensors → MLX/GGUF 포맷 변환 |
| `deployer` | 서빙 시작 (HTTP API, OpenAI 호환) |
| `benchmarker` | 표준화된 성능 벤치마크 |

### 2.4 Engine (추론 엔진)

최적화된 추론을 수행하는 핵심 엔진이다.

| 컴포넌트 | 역할 |
|---------|------|
| `mlx_engine` | MLX 기반 추론 루프 |
| `kv_cache` | KV 캐시 관리 (PagedAttention, 압축) |
| `speculative` | Speculative decoding 구현 |
| `attention` | Attention 최적화 (ANE 하이브리드 등) |

## 3. 데이터 흐름

```
forge optimize meta-llama/Llama-3-70B

1. Analyzer
   ├── hardware_profiler.detect() → HardwareProfile
   ├── model_inspector.inspect("meta-llama/Llama-3-70B") → ModelProfile
   └── memory_calculator.calculate(ModelProfile, HardwareProfile) → MemoryBudget

2. Optimizer
   ├── strategy_selector.select(ModelProfile, HardwareProfile, MemoryBudget) → Strategy
   └── (전략 출력: int4, mlx, mlx-lm, 4096 ctx, ...)

3. Pipeline
   ├── converter.download("meta-llama/Llama-3-70B")
   ├── converter.convert(format="mlx")
   ├── quantizer.quantize(method="mlx_native", bits=4)
   ├── profiler.benchmark(tokens=100) → Metrics
   ├── profiler.adjust(Strategy, Metrics) → AdjustedStrategy
   └── deployer.save_config(AdjustedStrategy)

4. Deploy (선택)
   └── deployer.serve(port=8080)
```

## 4. 설정 파일 구조

### 4.1 모델 최적화 결과 (YAML)

```yaml
# configs/optimized-llama-3-70b.yaml
model:
  id: meta-llama/Llama-3-70B
  architecture: llama
  type: dense
  params: 70B

optimization:
  quantization: int4
  method: mlx_native
  format: mlx
  group_size: 128

runtime:
  engine: mlx-lm
  context_length: 4096
  batch_size: 1
  num_threads: 14

performance:
  estimated_tps: 8.5
  estimated_memory_gb: 41.2
  ttft_seconds: 2.3

hardware:
  chip: Apple M4 Pro
  memory_gb: 48
  gpu_cores: 20
```

## 5. 확장 포인트

1. **새로운 양자화 기법**: `quantizer.py`에 새 Quantizer 클래스 추가
2. **새로운 런타임**: `deployer.py`에 새 Deployer 클래스 추가
3. **새로운 최적화**: `strategy_selector.py`의 규칙에 추가
4. **커스텀 커널**: `metal/kernels.metal`에 Metal 셰이더 추가
