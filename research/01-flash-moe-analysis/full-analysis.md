# flash-moe 레퍼지토리 전수 분석

> 분석 대상: https://github.com/danveloper/flash-moe
> 분석 일자: 2026-03-25

## 1. 프로젝트 개요

- **목표**: Qwen3.5-397B-A17B (397B 파라미터 MoE 모델)을 MacBook Pro 48GB에서 4.4+ tok/s로 구동
- **핵심 성과**: 서버 클러스터가 필요하던 모델을 노트북에서 인터랙티브 속도로 구동
- **모델-DRAM 비율**: 4x (209GB 모델 → 48GB RAM)

## 2. 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | Objective-C (59.4%), C (13.6%), Metal (7.4%), Python (8.7%), TeX (9.7%) |
| GPU | Apple Metal (M3 Max, 40코어 GPU) |
| 빌드 | Makefile + Clang (-O2, -fobjc-arc) |
| 의존성 | Accelerate framework, pthreads, compression |
| Python | safetensors, numpy, torch (가중치 변환용) |

## 3. 디렉토리 구조

```
flash-moe/
├── metal_infer/
│   ├── infer.m          # 핵심 추론 엔진 (~7000줄)
│   ├── shaders.metal    # GPU 커널 (~1200줄)
│   ├── chat.m           # 인터랙티브 채팅 + 도구 호출
│   ├── main.m           # 벤치마크/테스트
│   ├── tokenizer.h      # BPE 토크나이저 (단일 헤더 C)
│   ├── extract_weights.py    # 가중치 변환
│   ├── repack_experts_2bit.py # 2-bit 양자화 변환
│   ├── train_predictor.py     # Expert 라우팅 분석
│   ├── Makefile
│   ├── model_weights.bin (5.5GB)
│   ├── vocab.bin, tokenizer.bin
├── docs/                # 기술 문서
├── paper/               # 연구 논문 (LaTeX)
├── CLAUDE.md
├── expert_index.json
├── results.tsv
└── progress.py
```

## 4. 구현된 최적화 기법 상세

### 4.1 양자화 & 압축

**4-bit per-group affine 양자화**
- group-64 블록 단위로 scale/bias 공유
- 가중치: 4-bit unsigned int로 저장
- 역양자화: `value = scale * quantized + bias`

**2-bit 양자화**
- Expert 크기 44% 감소 (7.08MB → 3.93MB)
- RMSE 손실 매우 작음 (0.001-0.003)
- 단점: JSON 생성 품질 저하 (도구 호출 불가)

**FMA 최적화 역양자화**
- `scale×x + bias×x` 를 사전 계산하여 fused multiply-add 활용
- 12% 성능 향상

**LUT 기반 최적화**
- 그룹당 16-entry 룩업 테이블
- uint→float 타입 변환 완전 제거
- Metal 커널 `dequant_matvec_4bit_v5`에 적용

### 4.2 I/O 아키텍처

**병렬 pread() Expert 스트리밍**
- 8개 pthread가 SSD에서 expert 가중치를 GPU 버퍼로 직접 로딩
- 매 추론 스텝당 ~600MB 데이터 이동
- Expert당 7MB (4-bit) / 3.93MB (2-bit)

**OS 페이지 캐시 신뢰 전략**
- 커스텀 LRU 캐시 제거 → 38% 성능 향상
- 이유: 커스텀 캐시가 macOS 메모리 압축기 thrashing 유발
- 핵심 철학: "Trust the OS"

**2MB 정렬 버퍼**
- 메모리 정렬 최적화로 5% 개선
- 기본 정렬 대비 3.6배 속도 향상

### 4.3 GPU 파이프라인 최적화

**3-stage Command Buffer Fusion**
레이어당 GPU dispatch를 ~25개에서 3개로 축소:

| 스테이지 | 연산 내용 |
|---------|----------|
| CMD1 | Attention 입력 프로젝션 (Q, K, V matmul) |
| CMD2 | 출력 프로젝션 + residual + 정규화 + 라우팅 |
| CMD3 | K개 expert forward + shared expert + GPU측 합산 + 정규화 |

- 레이어당 0.83ms 절약
- CMD3를 비동기 제출하여 다음 레이어 CMD1과 오버랩

**Double-buffered I/O**
- Set A: GPU 연산용
- Set B: 다음 스텝 프리페치용

**Batched Matmul Encoding**
- N개 독립 matmul을 단일 command buffer로 통합

### 4.4 Metal 커널 최적화

| 커널/기법 | 설명 |
|----------|------|
| `dequant_matvec_4bit_v5` | LUT 최적화 역양자화 커널 |
| Threadgroup shared memory | 입력 벡터를 한 번 로드 후 reduction에서 재사용 |
| Coalesced access | 인접 스레드가 인접 128-bit(uint4) 또는 32-bit(uint32) 읽기 |
| SIMD reduction | simd_sum() + threadgroup 집계 |
| FMA 명령어 | 역양자화+곱셈을 단일 GPU 명령어로 |
| 커널 퓨전 | Gate+Up+SwiGLU, O+RMSNorm+Routing, MoE+Residual |
| Multi-head 병렬성 | Grid Y-dimension으로 expert 배칭 |

### 4.5 메모리 관리

- **Unified memory**: `MTLResourceStorageModeShared`로 CPU-GPU 일관성
- **재사용 버퍼 풀**: K=8 expert용 사전 할당, scratch 버퍼
- **Memory-mapped weights**: ~6GB 할당, 나머지 42GB를 OS 페이지 캐시용으로 확보
- **K=4 Expert 프루닝**: 기본 K=10 대비 2.6배 속도 향상

### 4.6 Attention 최적화

- **RoPE**: GPU 측 계산 (CPU 오버헤드 제거)
- **KV 캐싱**: GPU 버퍼에 key-value 캐시 유지
- **GQA (Grouped-Query Attention)**: 다수 query head가 하나의 KV head 공유
- **Linear attention (GatedDeltaNet)**: 처음 45레이어에 적용 (recurrent state 기반)
- **Full attention**: 마지막 15레이어에 적용

### 4.7 MoE (Mixture of Experts)

- 레이어당 512 experts
- K=4 활성 expert + 1 shared expert
- Weighted expert combination (GPU측 fusion + residual + shared expert gating)

## 5. 시도 후 폐기한 기법들

| 기법 | 결과 | 폐기 이유 |
|------|------|----------|
| LZ4 압축 | -13% | 15-24% 빠른 읽기지만 0.68ms 압축 해제 오버헤드가 이득 상쇄 |
| 커스텀 Metal LRU 캐시 (500-3000 entry) | -38% | 메모리 압축기 thrashing 유발 |
| F_NOCACHE / F_RDADVISE 프리페치 | +73% GPU 메모리 지연 | 메모리 컨트롤러 경합 |
| GPU 커널 변형 (LUT dequant, vector loads) | -2~3% | 레지스터 압박 |
| dispatch_io | -70% | pread 대비 심각한 성능 저하 |
| aio_read | -7% | pread 대비 열등 |
| 투기적 라우팅 예측 | 0.4% 성공률 | 실용 불가 (25.6% 적중률도 부족) |
| 추가 커널 퓨전 시도 | -2% | 오히려 성능 저하 |

## 6. 성능 벤치마크

### 최종 성능
| 설정 | 속도 | 비고 |
|------|------|------|
| 4-bit 양자화 | 4.36 tok/s | 우수한 출력 품질, 도구 호출 가능 |
| 2-bit 양자화 | 5.74 tok/s | JSON 생성 불량, 도구 호출 불가 |
| 웜 캐시 피크 | 7.05 tok/s | 단일 토큰 기준 |

### 토큰당 성능 분포 (5.74 tok/s 기준)
- Expert I/O: 47%
- GPU 연산 (attention): 30%
- GPU 연산 (라우팅/shared expert): 15%
- CPU attention: 5%
- 기타: 3%

### 스케일링 전망
- 콜드 캐시: 3.90 tok/s
- 웜 캐시: 7.05 tok/s
- OS 페이지 캐시 적중률: ~71%
- M4 Max 예상: ~8 tok/s
- 차세대 SSD: ~11 tok/s

## 7. 아키텍처 설계 원칙

1. **하드웨어 인식 설계**: Apple Silicon의 GPU/SSD 동시 실행 불가 특성을 수용, 직렬 실행 최적화
2. **OS 우선 설계**: macOS 페이지 캐시를 애플리케이션 캐시보다 신뢰
3. **GPU 파이프라인 스테이징**: 3-stage command buffer로 round-trip 최소화
4. **병렬 I/O**: 다중 pthread로 concurrent expert 로딩
5. **스트리밍 아키텍처**: Expert를 사전 로딩이 아닌 on-demand 스트리밍

## 8. 모델 아키텍처 (Qwen3.5-397B-A17B)

- 60 Transformer 레이어 (45 linear attention + 15 full attention)
- 레이어당 512 experts, K=4 + 1 shared
- 4096차원 hidden layer
- 248,320 어휘 크기
- Gate-Up-Down MoE 프로젝션 구조

## 9. 핵심 한계점

1. **단일 모델 고정**: Qwen3.5-397B에 완전히 하드코딩, 다른 모델 미지원
2. **수동 최적화**: 모든 튜닝이 수작업, 자동화 없음
3. **Objective-C 기반**: 접근성 낮음, Python/MLX 대비 개발 생산성 저하
4. **ANE 미활용**: Metal GPU만 사용, Apple Neural Engine 완전 무시
5. **MLX 미활용**: Apple의 ML 프레임워크를 활용하지 않음
6. **고급 양자화 미적용**: 기본 per-group 양자화만 사용 (AQLM, HQQ, QuIP# 미적용)
