# 구현 단계별 계획

## 실측 벤치마크 결과 (Qwen2.5-7B 4-bit, M4 Pro 48GB)

| 설정 | 속도 (tok/s) | 배수 | 비고 |
|------|-------------|------|------|
| Baseline | 55.7 | 1.00x | MLX 4-bit 기본 |
| + Speculative (0.5B draft) | 67.6 | 1.21x | 가장 큰 향상 |
| + KV-bits 8 (FP8 KV) | 53.8 | ~1.0x | 메모리 50% 절약 |
| + Spec + KV-bits 8 | 61.5 | 1.10x | 속도 + 메모리 절약 |
| HQQ 3-bit (0.5B) | 308.6 | — | 작은 모델 참고 |

---

## Phase 1: MVP — 모델 분석 + 기본 최적화 (완료)

### 목표
`forge analyze <model>` → `forge optimize <model>` → 동작하는 최적화 모델

### 구현 항목

**1.1 프로젝트 기초**
- [x] 디렉토리 구조 생성
- [x] research/ 리서치 자료 문서화
- [x] docs/ 설계 문서 작성
- [ ] CLAUDE.md 생성
- [ ] pyproject.toml (의존성: mlx, mlx-lm, transformers, click, pyyaml)
- [ ] `src/forge/__init__.py` 패키지 초기화

**1.2 하드웨어 프로파일러** (`src/forge/analyzer/hardware_profiler.py`)
- [ ] CPU 정보 감지 (sysctl)
- [ ] GPU 코어 수 감지 (system_profiler)
- [ ] 메모리 총량 감지
- [ ] MLX/Ollama 설치 여부 확인
- [ ] HardwareProfile 데이터클래스 반환

**1.3 모델 인스펙터** (`src/forge/analyzer/model_inspector.py`)
- [ ] AutoConfig로 모델 메타데이터 추출
- [ ] Dense/MoE 자동 판별
- [ ] Attention 타입 감지 (MHA/GQA/MQA)
- [ ] 파라미터 수 추정
- [ ] ModelProfile 데이터클래스 반환

**1.4 메모리 계산기** (`src/forge/analyzer/memory_calculator.py`)
- [ ] 가중치 메모리 계산 (양자화별)
- [ ] KV 캐시 메모리 계산 (컨텍스트별)
- [ ] 활성화 메모리 계산
- [ ] 총 메모리 예산 계산
- [ ] MemoryBudget 데이터클래스 반환

**1.5 전략 선택기** (`src/forge/optimizer/strategy_selector.py`)
- [ ] 양자화 수준 자동 결정
- [ ] 포맷/런타임 자동 선택
- [ ] 컨텍스트 길이 계산
- [ ] OptimizationStrategy 반환

**1.6 변환기** (`src/forge/pipeline/converter.py`)
- [ ] HuggingFace 모델 다운로드
- [ ] mlx-lm convert 래핑
- [ ] GGUF 변환 (폴백)

**1.7 CLI** (`src/forge/cli.py`)
- [ ] `forge analyze <model>` 커맨드
- [ ] `forge optimize <model>` 커맨드
- [ ] 결과 출력 포매팅

### 검증
```bash
forge analyze meta-llama/Llama-3-8B
forge optimize meta-llama/Llama-3-8B --auto
mlx_lm.generate --model ./optimized/Llama-3-8B-q4 --prompt "Hello"
```

---

## Phase 2: 고급 양자화 + 프로파일링 (목표: +2주)

### 구현 항목

**2.1 HQQ 양자화 통합** (`src/forge/optimizer/quantizer.py`)
- [ ] HQQ 설치 및 통합
- [ ] 2-3bit 양자화 파이프라인
- [ ] HQQ → MLX 가중치 변환

**2.2 프로파일러** (`src/forge/optimizer/profiler.py`)
- [ ] 짧은 벤치마크 실행 (100 토큰)
- [ ] TTFT, TPS, 메모리 측정
- [ ] 파라미터 자동 조정 (컨텍스트, 배치)

**2.3 벤치마크 스위트** (`src/forge/pipeline/benchmarker.py`)
- [ ] 표준화된 벤치마크 프롬프트 세트
- [ ] TTFT, TPS, 품질 메트릭
- [ ] 결과 비교 테이블 출력

**2.4 CLI 확장**
- [ ] `forge bench <model>` 커맨드
- [ ] `forge optimize --quant hqq` 옵션
- [ ] 프로파일 결과 표시

### 검증
```bash
forge optimize meta-llama/Llama-3-70B --quant hqq --bits 3
forge bench ./optimized/Llama-3-70B-hqq-3bit
```

---

## Phase 3: 배포 + Ollama 통합 (목표: +1주)

### 구현 항목

**3.1 배포 관리** (`src/forge/pipeline/deployer.py`)
- [ ] MLX 서버 시작 (OpenAI 호환 API)
- [ ] Ollama Modelfile 자동 생성
- [ ] GGUF 배포 경로

**3.2 Prompt 캐싱**
- [ ] mlx-lm cache_prompt 통합
- [ ] 시스템 프롬프트 캐시 관리

**3.3 CLI 확장**
- [ ] `forge deploy <model>` 커맨드
- [ ] `forge list` (최적화된 모델 목록)

### 검증
```bash
forge deploy ./optimized/Llama-3-8B-q4 --port 8080
curl http://localhost:8080/v1/chat/completions -d '{"messages":[...]}'
```

---

## Phase 4: 추론 엔진 최적화 (목표: +4주)

### 구현 항목

**4.1 Speculative Decoding** (`src/forge/engine/speculative.py`)
- [ ] Draft 모델 자동 선택
- [ ] Tree-based speculative 구현
- [ ] 수락률 모니터링

**4.2 KV 캐시 최적화** (`src/forge/engine/kv_cache.py`)
- [ ] PagedAttention 구현
- [ ] FP8 KV 캐시 압축
- [ ] 캐시 프리페치

**4.3 Expert 캐싱** (MoE 전용)
- [ ] Local Routing Consistency 기반 캐싱
- [ ] 캐시 크기 자동 조정

### 검증
```bash
forge optimize meta-llama/Llama-3-70B --speculative --draft SmolLM-360M
forge bench ./optimized/ --compare baseline,speculative
```

---

## Phase 5: ANE 하이브리드 + 고급 기능 (목표: +8주)

### 구현 항목

**5.1 ANE 하이브리드** (`src/forge/engine/attention.py`)
- [ ] CoreML 변환 파이프라인
- [ ] ANE prefill + GPU decode
- [ ] KV 캐시 공유 메커니즘

**5.2 커스텀 Metal 커널** (`src/forge/metal/kernels.metal`)
- [ ] 병목 구간 프로파일링
- [ ] 최적화 커널 구현

### 검증
```bash
forge optimize <model> --ane-hybrid
forge bench --compare gpu-only,ane-hybrid
```
