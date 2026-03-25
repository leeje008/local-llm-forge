# 자동 파이프라인 설계 문서

## 1. 핵심 컨셉

flash-moe는 단일 모델(Qwen3.5-397B)에 수동으로 최적화를 적용하는 "하드코딩" 접근법.
local-llm-forge는 **임의의 모델**을 **자동으로** 최적화하는 "메타 시스템".

```
flash-moe:  Qwen3.5-397B → [수동 최적화 코드] → 4.4 tok/s
forge:      <any model>   → [자동 분석/최적화]  → 최적 설정
```

## 2. 파이프라인 5단계

### Stage 1: ANALYZE (분석)

**입력**: model_id (HuggingFace 형식)
**출력**: ModelProfile + HardwareProfile + MemoryBudget

```python
# 사용 예시
profile = analyzer.inspect("meta-llama/Llama-3-70B")
hardware = analyzer.detect_hardware()
budget = analyzer.calculate_memory(profile, hardware)

# 결과 예시
print(profile)
# ModelProfile(
#   architecture="llama", type="dense", params=70B,
#   layers=80, hidden=8192, attn_heads=64, kv_heads=8,
#   attention="GQA", vocab=128256, max_context=131072
# )

print(budget)
# MemoryBudget(
#   available=38.0GB,
#   fp16_weights=140.0GB,  fp16_fits=False,
#   int4_weights=36.8GB,   int4_fits=False (with KV),
#   int3_weights=27.6GB,   int3_fits=True,
#   int2_weights=18.4GB,   int2_fits=True,
#   kv_per_1k_tokens=0.84GB
# )
```

### Stage 2: STRATEGIZE (전략 수립)

**입력**: ModelProfile + HardwareProfile + MemoryBudget
**출력**: OptimizationStrategy

결정 항목:
1. 양자화 수준 (가장 높은 비트에서 시작, 메모리에 맞을 때까지 내림)
2. 양자화 기법 (4bit+ → MLX native, 3bit → mixed/HQQ, 2bit → HQQ)
3. 포맷 (MLX 우선, GGUF 폴백)
4. 런타임 (mlx-lm 우선, Ollama/llama.cpp 폴백)
5. 컨텍스트 길이 (남은 메모리 기반 최대값)
6. 추가 최적화 (speculative, prompt cache, expert cache)

### Stage 3: BUILD (빌드)

**입력**: model_id + OptimizationStrategy
**출력**: 최적화된 모델 디렉토리

```
./optimized/Llama-3-70B-int3-hqq/
├── config.json
├── model.safetensors     # 양자화된 가중치
├── tokenizer.json
├── forge_config.yaml     # forge 최적화 설정
└── benchmark_results.json
```

단계:
1. `huggingface_hub.snapshot_download()` → 모델 다운로드
2. `mlx_lm.convert()` 또는 `convert-hf-to-gguf.py` → 포맷 변환
3. 양자화 적용 (MLX native / HQQ / AQLM)
4. 설정 파일 생성

### Stage 4: PROFILE (프로파일)

**입력**: 최적화된 모델 + 초기 설정
**출력**: 조정된 설정 + 성능 메트릭

```python
# 100 토큰 벤치마크 실행
metrics = profiler.quick_bench(model_path, config)
# Metrics(ttft=2.3s, tps=8.5, peak_memory=41.2GB, oom=False)

# OOM 발생 시 자동 조정
if metrics.oom:
    config.context_length //= 2
    metrics = profiler.quick_bench(model_path, config)

# 메모리 여유 시 컨텍스트 확대
if metrics.peak_memory < hardware.available * 0.7:
    config.context_length = int(config.context_length * 1.5)
```

### Stage 5: DEPLOY (배포)

**입력**: 최적화된 모델 + 최종 설정
**출력**: 실행 중인 서버 또는 저장된 설정

배포 옵션:
1. **MLX 서버**: OpenAI 호환 HTTP API
2. **Ollama 등록**: Modelfile 자동 생성 → `ollama create`
3. **설정 저장**: 나중에 재사용 가능한 config 파일

## 3. 에러 처리 & 폴백 체인

```
MLX 변환 실패 → GGUF 변환 시도 → Ollama 직접 다운로드
양자화 실패 → 한 단계 높은 비트로 재시도
OOM 발생 → 컨텍스트 50% 축소 → 재시도
성능 미달 → 추가 최적화 비활성화 (speculative 끄기 등)
```

## 4. 설정 캐싱 & 재사용

한 번 최적화한 모델의 설정을 저장하여 재사용:

```yaml
# ~/.forge/profiles/meta-llama--Llama-3-70B.yaml
created: 2026-03-25
hardware_hash: m4pro_48gb_20gpu  # 하드웨어 변경 시 재최적화
strategy:
  quantization: int3
  method: hqq
  format: mlx
  runtime: mlx-lm
  context: 4096
  batch: 1
performance:
  tps: 7.8
  ttft: 3.1
  memory: 42.1
```

## 5. CLI 인터페이스 상세

```bash
# 분석만 (다운로드 없음, 빠름)
forge analyze meta-llama/Llama-3-70B
forge analyze Qwen/Qwen2.5-32B-Instruct
forge analyze mistralai/Mixtral-8x7B-v0.1

# 자동 최적화 (다운로드 + 변환 + 양자화 + 프로파일)
forge optimize meta-llama/Llama-3-70B
forge optimize meta-llama/Llama-3-70B --bits 3 --method hqq
forge optimize meta-llama/Llama-3-70B --speculative --draft TinyLlama-1.1B

# 배포
forge deploy ./optimized/Llama-3-70B-int3
forge deploy ./optimized/Llama-3-70B-int3 --port 8080 --api openai

# 벤치마크
forge bench ./optimized/Llama-3-70B-int3
forge bench --compare int3,int4  # 양자화 수준 비교

# 관리
forge list                    # 최적화된 모델 목록
forge info ./optimized/...    # 모델 상세 정보
forge clean                   # 캐시/임시 파일 정리
```
