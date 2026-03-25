# 고도화 논문 서베이 (10회 리서치 종합)

> 조사 일자: 2026-03-26
> 분석 논문: 60+ papers (2024-2026)

## 양자화

| 논문 | ID | 비트 | 핵심 | MLX |
|------|-----|------|------|-----|
| SpinQuant | 2405.16406 | 4-bit W/A/KV | Rotation으로 outlier 제거, QuaRot 45% 개선 | 코드有, MLX 미포팅 |
| ParetoQ | 2502.02631 | 1-4bit | 2-3bit 임계점, ternary Pareto frontier | 코드有 |
| SliM-LLM | 2405.14917 | 2-bit | Salience-driven mixed-precision, 48% ppl 개선 | 가능 |
| any4 | 2507.04610 | 4-bit | 학습된 숫자 표현, INT4/FP4/NF4 초과 | FB코드有 |
| AQLM | 2401.06118 | 2-3bit | Additive quantization, sub-3bit SOTA | 코드有 |
| QuIP# | 2402.04396 | 2-3bit | Hadamard+E8 lattice codebook | 코드有 |
| CRVQ | 2412.09282 | <2bit | Channel-relaxed VQ, 38.9% 개선 | 코드有 |
| LittleBit | 2506.13771 | 0.1bit | Latent factorization, 31x 압축 | QAT필요 |
| Sparsity+Quant | 2405.20935 | - | S→Q 순서 증명 (ICLR Spotlight) | 코드有 |

## Speculative Decoding

| 논문 | ID | 속도향상 | 핵심 | MLX |
|------|-----|---------|------|-----|
| ReDrafter | 2403.09919 | 1.37-2.3x | Apple Research, MLX 검증 | **검증됨** |
| Mirror SD | 2510.13161 | 2.8-5.8x | GPU+ANE 병렬 (Apple Research) | 설계만 |
| EAGLE-3 | 2503.01840 | 6.5x | Training-time test, NeurIPS 2025 | vLLM有 |
| Judge Decoding | ICLR 2025 | 9x | Judge head로 드래프트 평가 | 이론 |
| Self-Speculative | 2510.04147 | 3.46x | 드래프트 모델 불필요 | 가능 |
| PEARL | 2408.11850 | 4.43x | Pre/post-verify 오버랩 | 가능 |
| OmniDraft | 2507.02659 | 1.5-2x | 단일 드래프트로 모든 타겟 | 온디바이스 |

## KV 캐시

| 논문 | ID | 절약 | 핵심 | MLX |
|------|-----|------|------|-----|
| KIVI | 2402.02750 | 2.6x 메모리 | 비대칭 2-bit KV | kv_bits 확장 필요 |
| StreamingLLM | 2309.17453 | 고정 메모리 | Attention sink + 슬라이딩 윈도우 | max_kv_size |
| H2O | 2306.14048 | 80% 감소 | Heavy-hitter 20% 유지 | 커스텀 필요 |
| LazyLLM | 2407.14057 | 2.34x prefill | Apple Research 토큰 프루닝 | 가능 |
| CommonKV | 2508.16134 | 98% (조합) | Cross-layer KV 공유 | SVD 기반 |
| Coupled Quant | 2405.03917 | 1-bit KV | 채널 간 의존성 활용 | 이론 |

## MoE Expert

| 논문 | ID | 효과 | 핵심 | MLX |
|------|-----|------|------|-----|
| REAP | 2510.13999 | 50% 무손실 | Router-weighted 프루닝 | 가능 |
| Sub-MoE | 2506.23266 | 96%@25% | Joint SVD expert 병합 | 가능 |
| MoE-SVD | ICML 2025 | 60% 압축 | 분해 기반 1.5x 속도 | 가능 |
| PreScope | 2509.23638 | 141% throughput | Layer-aware prefetch | 가능 |
| MoE-SpeQ | 2511.14102 | 2.34x | Speculative + expert cache | 가능 |
| Local Routing | 2505.16056 | 분석 도구 | SRP/SCH offloading 메트릭 | 가능 |
| DeepSeekMoE | 2401.06066 | 분석 | Expert specialization 패턴 | 분석용 |
| Aux-Loss-Free | 2408.15664 | 로드밸런스 | Dynamic bias (DeepSeek-V3) | 가능 |
