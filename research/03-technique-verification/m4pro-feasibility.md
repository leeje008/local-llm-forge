# M4 Pro 적용 가능성 검증 (1차)

> 검증 일자: 2026-03-25
> 대상 하드웨어: Apple M4 Pro (14CPU/20GPU, 48GB, Metal 4, ANE 38 TOPS)

## 1. 검증 매트릭스

| # | 기법 | M4 Pro 호환 | 48GB 적합 | 구현 난이도 | 종합 판정 |
|---|------|------------|----------|-----------|----------|
| 1 | 자동 파이프라인 | O | O | 중간 | **적합** |
| 2 | ANE 하이브리드 | O (38 TOPS) | O | 높음 | **적합 (검증필요)** |
| 3 | MLX 프레임워크 | O (설치됨) | O | 낮음 | **매우 적합** |
| 4 | HQQ 양자화 | O | O | 낮음 | **매우 적합** |
| 5 | AQLM 양자화 | △ (GPU 의존) | O | 중간 | **적합** |
| 6 | QuIP# | △ (CUDA 최적화) | O | 높음 | **조건부 적합** |
| 7 | PagedAttention | O | O | 중간 | **적합** |
| 8 | KV 캐시 압축 | O | O | 높음 | **적합** |
| 9 | Speculative Decoding | O | △ (추가 모델 필요) | 중간 | **적합** |
| 10 | Expert 캐싱 | O | O | 낮음 | **적합** |
| 11 | Prompt 캐싱 | O (mlx-lm 내장) | O | 낮음 | **매우 적합** |
| 12 | MoAS | O | O | 높음 | **연구 단계** |

## 2. 상세 검증

### 2.1 자동 최적화 파이프라인

**M4 Pro 호환성**: 완전 호환
- `transformers.AutoConfig`: Python 기반, 하드웨어 무관
- 메모리 계산: sysctl로 시스템 정보 추출 가능
- mlx-lm convert/quantize: MLX 0.31.0에서 즉시 사용 가능

**메모리 영향**: 파이프라인 자체는 최소 (분석/변환 시 일시적 사용)

**구현 경로**:
```
transformers.AutoConfig → 모델 분석
sysctl → 하드웨어 감지
mlx-lm → 변환/양자화
자체 로직 → 전략 결정/프로파일링
```

**판정**: 매우 적합. 기존 도구 조합으로 빠르게 구현 가능.

### 2.2 ANE 하이브리드 추론

**M4 Pro ANE 스펙**:
- 38 TOPS (Neural Engine)
- 2W 전력 소비 (GPU 대비 10x 효율)
- CoreML 모델 포맷 필요

**검증 사항**:
- ANEMLL 프로젝트에서 1B 모델 47-62 tok/s, 8B 모델 ~9 tok/s 달성
- Orion에서 GPT-2 학습까지 가능
- 하이브리드 ANE prefill + GPU decode 전략 검증됨

**제약사항**:
- CoreML 변환 필요 (추가 파이프라인)
- 큰 모델(30B+)에서의 ANE 성능 미검증
- Metal ↔ CoreML 컨텍스트 전환 오버헤드

**판정**: 적합하지만 심층 검증 필요. 먼저 소형 모델(1-8B)에서 프로토타입 후 스케일업.

### 2.3 MLX 프레임워크

**M4 Pro 호환성**: 완벽
- MLX 0.31.0 이미 설치됨
- Metal 4 완전 지원
- Unified memory zero-copy 최적화

**성능 기대치** (M4 Pro 20-core GPU 기준):
- 7B 4-bit: ~30-40 tok/s
- 14B 4-bit: ~15-25 tok/s
- 32B 4-bit: ~10-18 tok/s
- M4 Max (40 GPU) 대비 약 50-70% 성능 예상 (메모리 대역폭이 주 병목)

**M4 Pro 메모리 대역폭**: 273 GB/s
- 토큰 생성 속도 이론적 상한: `bandwidth / model_size_bytes`
- 7B 4-bit (3.7GB): ~73 tok/s 이론 상한
- 32B 4-bit (16.8GB): ~16 tok/s 이론 상한

**판정**: 매우 적합. 즉시 활용 가능.

### 2.4 HQQ 양자화

**M4 Pro 호환성**: 호환
- Python/PyTorch 기반, 하드웨어 비의존적
- 캘리브레이션 데이터 불필요
- 7B 모델 60초 내 양자화

**48GB 제약**:
- 양자화 과정에서 원본 모델 + 양자화 모델 동시 메모리 필요
- 7B FP16 (14GB) + 양자화 버퍼: ~20GB → 가능
- 70B FP16 (140GB): 불가능 → 청크 단위 처리 필요

**판정**: 매우 적합. 소형-중형 모델은 즉시 적용, 대형 모델은 청크 처리.

### 2.5 AQLM 양자화

**M4 Pro 호환성**: 부분 호환
- 학습 기반 양자화 → GPU 필요
- CUDA 최적화가 주력이나 PyTorch 기반이므로 MPS 백엔드로 동작 가능
- 추론 시 MLX 호환 가능

**판정**: 적합. 양자화 자체는 느릴 수 있으나 사전 양자화된 모델 활용 가능.

### 2.6 QuIP#

**M4 Pro 호환성**: 제한적
- CUDA 커널에 크게 의존
- Apple Silicon 네이티브 구현 부족
- Hadamard 변환 자체는 Metal로 구현 가능

**판정**: 조건부 적합. MLX용 커스텀 구현 필요하며 난이도 높음.

### 2.7 PagedAttention

**M4 Pro 호환성**: 호환
- vllm-mlx에서 실험적 지원 (paged attention)
- Unified memory에서 페이지 관리가 비교적 단순

**48GB KV 캐시 절약 예시**:
- 32B 모델, 8k context: 기본 ~8GB KV → PagedAttention으로 ~2GB 절약
- 절약된 메모리로 더 긴 컨텍스트 또는 더 큰 모델 가능

**판정**: 적합. vllm-mlx 참조 구현 활용 가능.

### 2.8 Speculative Decoding

**M4 Pro 호환성**: 호환
- Draft 모델 (1-3B) + Target 모델 동시 로딩 필요
- 48GB에서: 32B target(4-bit, 17GB) + 1B draft(4-bit, 0.5GB) = ~28GB → 가능

**예상 효과**:
- 단순 greedy: 1x (baseline)
- Speculative (수락률 70%): ~1.5-2x 향상
- Tree-based (Medusa): ~2-2.5x 향상

**제약**: 메모리에 두 모델 동시 로딩 필요 → 큰 target 모델에서는 불가

**판정**: 적합. 32B 이하 모델에서 효과적.

### 2.9 Expert 캐싱 (Local Routing Consistency)

**M4 Pro 호환성**: 완전 호환
- 소프트웨어 수준 최적화, 하드웨어 의존성 없음
- OS 페이지 캐시와 상호보완적

**48GB 활용**:
- K=4 active experts × 2 = 8 experts 캐시
- Expert당 7MB (4-bit) × 8 = 56MB 추가 메모리만 필요

**판정**: 매우 적합. 최소 메모리로 I/O 병목 완화 가능.

### 2.10 Prompt 캐싱

**M4 Pro 호환성**: 완전 호환
- mlx-lm의 `cache_prompt` 기능 즉시 사용 가능
- SafeTensors 포맷으로 프롬프트 캐시 저장/로딩

**판정**: 매우 적합. 즉시 사용 가능.

## 3. 종합 권장 구현 순서

| 순서 | 기법 | 이유 |
|------|------|------|
| 1 | MLX 기반 추론 엔진 | 이미 설치됨, 즉시 시작 가능, 기반 인프라 |
| 2 | 자동 파이프라인 (분석+변환+배포) | 핵심 차별점, 기존 도구 조합 |
| 3 | HQQ 양자화 통합 | 캘리브레이션 불필요, 빠른 적용 |
| 4 | Prompt 캐싱 | mlx-lm 내장, 즉시 활용 |
| 5 | Expert 캐싱 | 최소 메모리, I/O 병목 완화 |
| 6 | PagedAttention | 컨텍스트 확장 가능 |
| 7 | Speculative Decoding | 토큰 생성 속도 향상 |
| 8 | ANE 하이브리드 | 전력 효율 + 성능, 프로토타입 필요 |

## 4. 메모리 예산 시뮬레이션 (48GB)

```
사용 가능 메모리: 48GB
- macOS 오버헤드: -8GB
- Python/MLX 런타임: -2GB
= 순수 모델용: 38GB

시나리오 A: 32B 4-bit (sweet spot)
- 모델 가중치: 16.8GB
- KV 캐시 (8k ctx): 8GB
- 활성화 메모리: 0.3GB
- 여유: 12.9GB (→ 컨텍스트 확장 또는 draft 모델 로딩)

시나리오 B: 70B 2-bit + Speculative
- 모델 가중치: 18.4GB
- KV 캐시 (4k ctx): 6.7GB
- Draft 모델 (1B 4-bit): 0.5GB
- 활성화 메모리: 0.1GB
- 여유: 12.3GB

시나리오 C: MoE 8x7B 4-bit
- 모델 가중치: 24.7GB
- KV 캐시 (4k ctx): 3.4GB
- Expert 캐시 (8 experts): 0.06GB
- 여유: 9.9GB
```
