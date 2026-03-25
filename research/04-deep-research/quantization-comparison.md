# 심층 리서치: 양자화 기법 비교

> 리서치 일자: 2026-03-25
> 목적: M4 Pro 48GB 환경에서 최적의 양자화 기법 선택을 위한 비교 분석

## 1. 양자화 기법 종합 비교

| 기법 | 비트 | 캘리브레이션 | 속도(양자화) | 품질(vs FP16) | Apple Silicon | MLX 호환 |
|------|------|------------|-----------|-------------|-------------|---------|
| 기본 INT4 | 4 | 불필요 | 즉시 | 97-99% | O | O (내장) |
| GPTQ | 4 | 필요 (128샘플) | 느림 | 97-99% | △ | △ |
| AWQ | 4 | 필요 (128-256샘플) | 중간 | 95-99% | △ | △ |
| HQQ | 1-8 | **불필요** | **매우 빠름** | 96-99% | O | △ |
| AQLM | 2-4 | 필요 | 느림 | **≤4bit SOTA** | △ | X |
| QuIP# | 2-3 | 필요 | 느림 | **2-3bit SOTA** | X | X |
| AutoRound | 2-8 | 최소 | 빠름 | 97-99% | △ | △ |

## 2. 상세 분석

### 2.1 MLX 내장 양자화 (기본 INT4/INT8)

**방식**: 단순 per-group affine 양자화
```python
# mlx-lm에서 사용
mlx_lm.convert --model <id> -q --q-bits 4 --q-group-size 128
```

**장점**:
- MLX에 완전 통합, 추가 의존성 없음
- zero-copy 역양자화 커널 최적화
- Mixed quantization 지원: `mixed_2_6`, `mixed_3_4`, `mixed_4_6`
- 가장 빠른 추론 속도 (네이티브 커널)

**단점**:
- 고급 양자화 기법 대비 낮은 품질 (특히 2-3bit)
- 캘리브레이션 기반 최적화 없음

**권장 사용**: 기본 파이프라인, 4-bit 이상에서 사용

### 2.2 HQQ (Half-Quadratic Quantization)

**방식**: Robust optimization으로 최적 양자화 파라미터 탐색
```python
from hqq.core.quantize import HQQLinear, HQQBackend
HQQLinear.set_backend(HQQBackend.PYTORCH)
# 캘리브레이션 데이터 없이 양자화
```

**장점**:
- **캘리브레이션 데이터 완전 불필요**
- GPTQ 대비 50x 빠른 양자화 속도 (Llama-2-70B < 5분)
- 1-bit까지 지원 (연구용)
- HuggingFace Transformers 통합 (HqqConfig)
- 다양한 비트폭 지원 (8, 4, 3, 2, 1)

**단점**:
- MLX 네이티브 커널 미지원 → PyTorch MPS 백엔드 사용
- 추론 속도가 MLX 내장 양자화보다 느릴 수 있음

**M4 Pro 적용 방안**:
1. HQQ로 양자화 수행
2. 양자화된 가중치를 MLX 포맷으로 변환
3. MLX 네이티브 커널로 추론

**권장 사용**: 2-3bit 양자화가 필요할 때, 캘리브레이션 없이 빠르게 양자화

### 2.3 AQLM (Additive Quantization of Language Models)

**방식**: 8-16개 가중치를 그룹으로 묶어 다중 벡터 코드의 합으로 표현

**장점**:
- ≤4bit에서 SOTA 성능
- 8x 메모리 감소 (14GB → 1.75GB for 7B)
- ICML 2024 게재

**단점**:
- 학습 기반 → GPU 집중적 양자화 과정
- CUDA 최적화 주력
- Apple Silicon 네이티브 지원 부족

**M4 Pro 적용 방안**:
- 사전 양자화된 AQLM 모델을 HuggingFace에서 다운로드
- 또는 MPS 백엔드로 로컬 양자화 (느림)

**권장 사용**: 극한 압축 필요 시 사전 양자화 모델 활용

### 2.4 AutoRound (Intel)

**방식**: Signed gradient descent로 rounding과 clipping 동시 최적화

**장점**:
- **AutoScheme API**: 자동으로 최적 다중 비트 스킴 선택
- 다양한 내보내기 포맷 (AutoRound, AutoGPTQ, AutoAWQ, GGUF)
- INT2에서도 강력한 성능 (enable_alg_ext)

**단점**:
- Intel 최적화 중심
- Apple Silicon 최적화 부족

**M4 Pro 적용 방안**:
- AutoRound로 양자화 → GGUF 내보내기 → llama.cpp/Ollama 사용
- 또는 AutoRound → AutoGPTQ 내보내기 → 변환

## 3. 비트별 최적 기법 권장

| 비트 | 1차 권장 | 2차 권장 | 이유 |
|------|---------|---------|------|
| 8-bit | MLX 내장 | HQQ | 네이티브 최적화 |
| 4-bit | MLX 내장 | HQQ | 품질 충분, 속도 최적 |
| 3-bit | HQQ | MLX mixed_3_4 | HQQ가 3bit에서 더 나은 품질 |
| 2-bit | HQQ | AQLM (사전양자화) | 캘리브레이션 없이 빠른 적용 |

## 4. forge 양자화 파이프라인 설계

```python
def select_quantization(model_config, target_bits, hardware):
    if target_bits >= 4:
        # 4bit 이상: MLX 내장 양자화 (최고 추론 속도)
        return MLXQuantizer(bits=target_bits, group_size=128)

    elif target_bits == 3:
        # 3bit: MLX mixed 또는 HQQ
        if hardware.has_mlx:
            return MLXQuantizer(recipe='mixed_3_6')  # 중요 레이어 6bit, 나머지 3bit
        return HQQQuantizer(bits=3)

    elif target_bits == 2:
        # 2bit: HQQ 우선, AQLM 사전양자화 차선
        if check_pretrained_available(model_config, 'aqlm', 2):
            return PretrainedQuantizer(source='aqlm')
        return HQQQuantizer(bits=2)

    else:
        raise ValueError(f"Unsupported bits: {target_bits}")
```

## 5. 참고 자료

- HQQ: https://github.com/dropbox/hqq
- AQLM: https://github.com/Vahe1994/AQLM
- AutoRound: https://github.com/intel/auto-round
- MLX Quantization: https://huggingface.co/docs/hub/en/mlx
- AWQ: https://arxiv.org/abs/2306.00978
- Quantization Handbook: https://towardsdatascience.com/the-ultimate-handbook-for-llm-quantization-88bb7cb0d9d7/
