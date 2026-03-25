# 심층 리서치: 자동 최적화 파이프라인

> 리서치 일자: 2026-03-25
> 목적: 모델 지정 → 자동 분석 → 자동 최적화 → 자동 배포 파이프라인의 구현 방안 심층 조사

## 1. 기존 도구별 자동화 역량 분석

### 1.1 mlx-lm

**자동화 가능 범위**:
```bash
# 원커맨드 다운로드 + 변환 + 양자화
mlx_lm.convert --model <hf-model-id> -q --q-bits 4 --q-group-size 128

# 자동 다운로드 + 추론
mlx_lm.generate --model <hf-model-id> --prompt "test"

# 프롬프트 캐싱
mlx_lm.cache_prompt --model <id> --prompt-cache-file cache.safetensors
```

**제한사항**:
- 하드웨어 분석 없음 (양자화 수준 수동 결정)
- 메모리 예산 계산 없음
- 프로파일 기반 자동 튜닝 없음
- MoE 모델에 대한 특별 최적화 없음

### 1.2 Ollama

**자동화 가능 범위**:
```bash
# 자동 GPU 감지 + 모델 다운로드 + 실행
ollama run llama3:8b

# HuggingFace 직접 실행
ollama run hf.co/username/model

# Modelfile 기반 커스텀 설정
ollama create my-model -f Modelfile
```

**fitGPU 알고리즘** (자동 GPU 할당):
- VRAM 기반 레이어 분배
- 최소 메모리 예약
- 다중 GPU 레이어 분할

**제한사항**:
- GGUF 포맷만 지원
- 세밀한 양자화 제어 불가
- 커스텀 최적화 적용 불가

### 1.3 AutoRound (Intel)

**자동화 가능 범위**:
```python
from auto_round import AutoRound
# AutoScheme API: 자동 다중 비트 스킴 선택
auto_round = AutoRound(model, tokenizer, enable_alg_ext=True)
auto_round.quantize()
# 내보내기: AutoRound, AutoGPTQ, AutoAWQ, GGUF, LLM-Compressor
auto_round.export("auto_round")
```

**장점**: 가장 정교한 자동 양자화 스킴 선택
**제한사항**: 주로 INT2-INT8, Apple Silicon 네이티브 최적화 부족

### 1.4 transformers AutoConfig

**모델 메타데이터 추출**:
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("model_id")

# Dense vs MoE 감지
is_moe = hasattr(config, 'num_experts') or hasattr(config, 'num_local_experts')

# Attention 타입 감지
if hasattr(config, 'num_key_value_heads'):
    if config.num_key_value_heads == 1:
        attn_type = "MQA"
    elif config.num_key_value_heads < config.num_attention_heads:
        attn_type = "GQA"
    else:
        attn_type = "MHA"

# 파라미터 수 추정
params = estimate_params(config)
```

## 2. forge 자동 파이프라인 설계

### 2.1 전체 흐름

```
[사용자 입력: model_id]
        │
        ▼
┌─────────────────┐
│  1. ANALYZE      │ ← AutoConfig + sysctl
│  모델 분석       │    모델 아키텍처, 파라미터, 어텐션 타입
│  하드웨어 분석   │    CPU, GPU, RAM, 디스크, ANE
│  메모리 계산     │    양자화별 메모리 예산
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. STRATEGIZE   │ ← 전략 선택 엔진
│  양자화 수준     │    FP16 → 4bit → 3bit → 2bit
│  포맷 선택       │    MLX vs GGUF vs CoreML
│  런타임 선택     │    mlx-lm vs Ollama vs llama.cpp
│  추가 최적화     │    speculative, KV 압축 등
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. BUILD        │ ← 변환/양자화 파이프라인
│  다운로드         │    HuggingFace Hub
│  포맷 변환       │    SafeTensors → MLX/GGUF
│  양자화 적용     │    HQQ/AQLM/기본 int4
│  설정 생성       │    컨텍스트, 배치, 스레드
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. PROFILE      │ ← 프로파일 기반 튜닝
│  짧은 벤치마크   │    100 토큰 생성
│  병목 분석       │    TTFT, 토큰/초, 메모리
│  파라미터 조정   │    컨텍스트, 배치 재조정
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. DEPLOY       │ ← 서빙 배포
│  서버 시작       │    HTTP API (OpenAI 호환)
│  설정 저장       │    재현 가능한 config
│  모니터링        │    메모리, throughput
└─────────────────┘
```

### 2.2 전략 선택 알고리즘 (상세)

```python
def select_strategy(model_config, hardware_info):
    available_memory = hardware_info.total_ram - 10  # OS + framework reserve (GB)

    # 1. 양자화 수준 결정
    for quant in ['fp16', 'int8', 'int4', 'int3', 'int2']:
        weight_memory = calc_weight_memory(model_config, quant)
        kv_memory = calc_kv_memory(model_config, seq_len=8192)
        total = weight_memory + kv_memory + 1  # activation buffer

        if total <= available_memory * 0.85:  # 15% 여유
            selected_quant = quant
            break

    # 2. 컨텍스트 길이 최적화
    remaining = available_memory - weight_memory - 1
    kv_per_token = calc_kv_per_token(model_config)
    max_context = int(remaining / kv_per_token)
    context = min(max_context, model_config.max_position_embeddings)

    # 3. 포맷 & 런타임 선택
    if hardware_info.has_mlx:
        format = 'mlx'
        runtime = 'mlx-lm'
    elif hardware_info.has_metal:
        format = 'gguf'
        runtime = 'llama.cpp'
    else:
        format = 'gguf'
        runtime = 'ollama'

    # 4. MoE 전용 최적화
    if model_config.is_moe:
        strategy.expert_offloading = total > available_memory * 0.7
        strategy.expert_cache_size = model_config.num_activated_experts * 2

    # 5. 추가 최적화 결정
    if weight_memory < available_memory * 0.5:
        # 메모리 여유 → speculative decoding 가능
        strategy.speculative = True
        strategy.draft_model = find_small_model(model_config.architecture)

    return strategy
```

### 2.3 메모리 계산 엔진 (상세)

```python
def calc_weight_memory(config, quant):
    """모델 가중치 메모리 계산 (GB)"""
    bits = {'fp16': 16, 'int8': 8, 'int4': 4, 'int3': 3, 'int2': 2}
    bytes_per_param = bits[quant] / 8

    if config.is_moe:
        # MoE: 공유 파라미터 + 모든 expert 파라미터
        shared_params = estimate_shared_params(config)
        expert_params = estimate_expert_params(config) * config.num_experts
        total_params = shared_params + expert_params
    else:
        total_params = estimate_total_params(config)

    return (total_params * bytes_per_param * 1.05) / 1e9  # 5% 오버헤드

def calc_kv_memory(config, seq_len, batch=1):
    """KV 캐시 메모리 계산 (GB)"""
    kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    # 2 = K + V, 2 bytes = FP16
    kv_bytes = 2 * config.num_hidden_layers * kv_heads * head_dim * seq_len * batch * 2
    return kv_bytes / 1e9

def calc_kv_per_token(config):
    """토큰당 KV 캐시 증가량 (GB)"""
    return calc_kv_memory(config, seq_len=1)
```

## 3. 기존 도구 통합 전략

### 3.1 주 파이프라인: MLX 경로
```
HuggingFace → mlx_lm.convert → mlx_lm.generate/serve
```
- 장점: zero-copy, 최고 throughput, 이미 설치됨
- 단점: MLX 미지원 모델 존재 가능

### 3.2 대체 파이프라인: GGUF 경로
```
HuggingFace → convert-hf-to-gguf.py → llama-quantize → ollama/llama.cpp
```
- 장점: 가장 넓은 모델 호환성
- 단점: MLX 대비 낮은 throughput

### 3.3 실험적 파이프라인: ANE 경로
```
HuggingFace → CoreML 변환 → ANE prefill + GPU decode
```
- 장점: 최고 전력 효율
- 단점: 변환 복잡, 큰 모델 미검증

## 4. 프로파일 기반 자동 튜닝

### 4.1 프로파일 메트릭
- **TTFT** (Time To First Token): prefill 속도
- **TPS** (Tokens Per Second): 생성 속도
- **Peak Memory**: 최대 메모리 사용량
- **Memory Utilization**: 메모리 활용률

### 4.2 자동 조정 로직
```python
def auto_tune(model, initial_config):
    # 1. 기본 설정으로 짧은 벤치마크
    baseline = benchmark(model, initial_config, tokens=100)

    # 2. OOM이면 컨텍스트 축소
    if baseline.oom:
        config.context *= 0.5
        return auto_tune(model, config)

    # 3. 메모리 여유가 크면 컨텍스트 확대
    if baseline.peak_memory < available * 0.7:
        config.context = int(config.context * 1.5)

    # 4. 최종 벤치마크
    final = benchmark(model, config, tokens=100)
    return config, final
```

## 5. 참고 자료

- mlx-lm: https://github.com/ml-explore/mlx-lm
- AutoRound: https://github.com/intel/auto-round
- Ollama fitGPU: https://docs.ollama.com/gpu
- transformers AutoConfig: https://huggingface.co/docs/transformers/main_classes/configuration
- vllm-mlx: https://github.com/waybarrios/vllm-mlx
- HuggingFace Hub API: https://huggingface.co/docs/hub
