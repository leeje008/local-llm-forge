# 심층 리서치: ANE 하이브리드 추론 전략

> 리서치 일자: 2026-03-25
> 목적: Apple Neural Engine을 활용한 LLM 추론 최적화 방안 조사

## 1. Apple Neural Engine (ANE) 개요

| 항목 | M4 Pro |
|------|--------|
| 연산 성능 | 38 TOPS |
| 전력 소비 | ~2W |
| 비교 (GPU) | GPU ~20W (10배 전력 차이) |
| 지원 포맷 | CoreML (.mlmodelc) |
| 정밀도 | FP16, INT8 |

## 2. ANE 기반 LLM 프로젝트 분석

### 2.1 Orion (arXiv:2603.06728)

**성과**:
- GPT-2 124M: 170+ tok/s (ANE 전용)
- 110M 모델 학습까지 가능
- ANE의 특성을 체계적으로 분석한 첫 연구

**핵심 발견**:
- ANE는 고정 함수 가속기로, 특정 연산 패턴에 최적화
- 배치 크기가 클수록 ANE 활용도 높아짐 (prefill에 적합)
- 단일 토큰 생성(decode)에서는 GPU가 더 효율적

**하이브리드 전략 제안**:
- ANE: prefill (입력 프롬프트 처리, 대량 배치)
- GPU (또는 SME): decode (토큰 하나씩 생성)

### 2.2 ANEMLL (github.com/Anemll/Anemll)

**성과**:
- 1B 모델: 47-62 tok/s
- 8B 모델: ~9 tok/s

**구현 방식**:
- CoreML로 모델 변환
- ANE 전용 추론 파이프라인
- 메모리 효율적인 청크 처리

**제약사항**:
- 큰 모델(30B+)에서의 성능 미검증
- CoreML 변환이 모든 아키텍처를 지원하지는 않음

## 3. 하이브리드 ANE + GPU 전략

### 3.1 이론적 배경

LLM 추론은 두 단계:
1. **Prefill**: 입력 토큰 전체를 한 번에 처리 → **Compute-bound** → ANE 적합
2. **Decode**: 토큰 하나씩 생성 → **Memory-bound** → GPU 적합 (높은 대역폭)

### 3.2 제안 아키텍처

```
[입력 프롬프트]
      │
      ▼
┌─────────────┐
│  ANE Prefill │ ← CoreML 모델
│  (배치 처리)  │    높은 TOPS/W, 대량 토큰 처리
└──────┬──────┘
       │ KV 캐시 전달
       ▼
┌─────────────┐
│  GPU Decode  │ ← MLX/Metal
│  (단일 토큰)  │    높은 메모리 대역폭, 단일 토큰 최적
└──────┬──────┘
       │
       ▼
  [출력 토큰]
```

### 3.3 구현 과제

| 과제 | 난이도 | 설명 |
|------|--------|------|
| CoreML 변환 | 중간 | 모든 모델이 깔끔하게 변환되지 않음 |
| KV 캐시 공유 | 높음 | ANE와 GPU 간 KV 캐시 포맷 호환 필요 |
| 컨텍스트 전환 | 중간 | ANE ↔ GPU 전환 오버헤드 |
| 모델 분할 | 높음 | 어떤 레이어를 ANE/GPU에 배치할지 결정 |
| 메모리 관리 | 중간 | 두 런타임의 메모리 사용 조율 |

### 3.4 실현 가능 시나리오

**시나리오 A: ANE prefill only (실현 가능성 높음)**
- Prefill만 ANE, 전체 decode는 GPU
- KV 캐시를 한 번만 전달
- 구현 복잡도 낮음

**시나리오 B: 레이어 분할 (실현 가능성 중간)**
- 특정 레이어는 ANE, 나머지는 GPU
- 레이어 간 데이터 전달 오버헤드

**시나리오 C: 완전 하이브리드 (실현 가능성 낮음)**
- 연산 유형별로 ANE/GPU 동적 분배
- 가장 높은 효율이지만 구현 복잡

## 4. M4 Pro 적용 판단

### 4.1 장점
- 38 TOPS ANE → prefill 가속 가능
- 배터리 수명 크게 개선 (2W vs 20W)
- Unified memory로 ANE ↔ GPU 데이터 전달 효율적

### 4.2 단점
- CoreML 변환 파이프라인 구축 필요
- 큰 모델에서의 성능 불확실
- MLX와 CoreML의 KV 캐시 포맷 불일치 가능

### 4.3 권장 접근법

**Phase 1 (프로토타입)**:
1. ANEMLL을 참고하여 1B 모델에서 ANE prefill 테스트
2. MLX decode와 결합하여 하이브리드 파이프라인 프로토타입
3. 성능 측정: ANE prefill + GPU decode vs GPU only

**Phase 2 (스케일업)**:
1. 8B 모델에서 하이브리드 검증
2. KV 캐시 공유 메커니즘 구현
3. 자동 전략 선택에 ANE 옵션 추가

**Phase 3 (최적화)**:
1. 프로파일 기반으로 ANE/GPU 분배 최적화
2. 배터리 모드 vs 성능 모드 전환
3. 큰 모델(14B+)에서의 가능성 탐색

## 5. 참고 자료

- Orion: https://arxiv.org/html/2603.06728v1
- ANEMLL: https://github.com/Anemll/Anemll
- CoreML Tools: https://coremltools.readme.io/
- WWDC 2025 MLX: https://developer.apple.com/videos/play/wwdc2025/298/
