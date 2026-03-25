# 최적화 전략 문서

## 1. 전략 선택 흐름

```
모델 분석 → 메모리 계산 → 양자화 선택 → 포맷 선택 → 런타임 선택 → 추가 최적화
```

## 2. 양자화 전략

### 2.1 결정 트리

```
모델이 48GB에 FP16으로 들어가는가?
├─ Yes → FP16 (최고 품질)
└─ No → INT8으로 들어가는가?
   ├─ Yes → INT8
   └─ No → INT4로 들어가는가?
      ├─ Yes → MLX native INT4 (최적 속도/품질 밸런스)
      └─ No → INT3으로 들어가는가?
         ├─ Yes → HQQ INT3 또는 MLX mixed_3_6
         └─ No → INT2로 들어가는가?
            ├─ Yes → HQQ INT2 (품질 저하 경고)
            └─ No → 모델 비지원 (메모리 부족)
```

### 2.2 양자화 기법 선택

| 비트 | 1차 선택 | 2차 선택 | 이유 |
|------|---------|---------|------|
| 8 | MLX native | - | 네이티브 최적화 |
| 4 | MLX native | HQQ | 네이티브 속도 최우선 |
| 3 | MLX mixed_3_6 | HQQ 3-bit | 중요 레이어 6bit 유지 |
| 2 | HQQ 2-bit | AQLM 사전양자화 | 캘리브레이션 없이 빠름 |

### 2.3 Mixed Quantization (MLX)

중요도가 높은 레이어에 더 높은 비트, 나머지에 낮은 비트 할당:
- `mixed_2_6`: attention 6-bit, FFN 2-bit
- `mixed_3_4`: attention 4-bit, FFN 3-bit
- `mixed_3_6`: attention 6-bit, FFN 3-bit
- `mixed_4_6`: attention 6-bit, FFN 4-bit

## 3. 포맷 선택 전략

| 조건 | 포맷 | 런타임 |
|------|------|--------|
| MLX 설치됨 + 지원 모델 | MLX | mlx-lm |
| MLX 미지원 모델 | GGUF | llama.cpp |
| 간단한 배포 원할 때 | GGUF | Ollama |
| ANE 활용 시 | CoreML | coremltools |

## 4. 컨텍스트 길이 전략

```python
# 사용 가능 메모리에서 컨텍스트 계산
remaining = available_memory - model_weight_memory - os_overhead - framework_buffer
kv_per_token = (2 * num_layers * num_kv_heads * head_dim * 2) / 1e9  # GB
max_context = int(remaining / kv_per_token)

# 모델의 최대 지원 컨텍스트와 비교
context = min(max_context, model_max_context)

# 최소 2k 보장
context = max(context, 2048)
```

## 5. 추가 최적화 결정

### 5.1 Speculative Decoding
**적용 조건**:
- 모델 가중치 + draft 모델이 가용 메모리의 70% 이하
- Target 모델이 7B 이상 (작은 모델에서는 오버헤드만 증가)

**Draft 모델 선택**:
| Target 아키텍처 | Draft 모델 후보 |
|----------------|----------------|
| Llama 계열 | Llama-3-1B, TinyLlama-1.1B |
| Qwen 계열 | Qwen-0.5B |
| Mistral 계열 | Mistral-0.1B (존재 시) |
| 범용 | SmolLM-360M |

### 5.2 Prompt 캐싱
**적용 조건**: 항상 활성화 (부작용 없음)

### 5.3 Expert 캐싱 (MoE 전용)
**적용 조건**: MoE 모델일 때 자동 활성화
- 캐시 크기: `num_active_experts * 2`

### 5.4 PagedAttention
**적용 조건**: 컨텍스트 8k 이상에서 활성화
- 메모리 절약: 30-50%

## 6. 성능 예측 공식

```python
# 토큰 생성 속도 이론적 상한 (메모리 대역폭 기반)
theoretical_max_tps = memory_bandwidth_gbs / model_size_gb

# M4 Pro 예시:
# 7B 4-bit (3.7GB): 273 / 3.7 = 73.8 tok/s (이론)
# 32B 4-bit (16.8GB): 273 / 16.8 = 16.3 tok/s (이론)
# 70B 4-bit (36.8GB): 273 / 36.8 = 7.4 tok/s (이론)

# 실제 성능 = 이론 * 효율 계수 (보통 0.5-0.8)
estimated_tps = theoretical_max_tps * efficiency_factor
```

## 7. 에러 처리 & 폴백

| 상황 | 대응 |
|------|------|
| OOM 발생 | 양자화 레벨 한 단계 하향 |
| MLX 변환 실패 | GGUF 폴백 |
| 모델 미지원 | Ollama 자동 다운로드 시도 |
| 성능 기대 미달 | 컨텍스트 축소 → 배치 조정 |
