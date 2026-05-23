# PROJECT_CONTEXT.md
> 모델: Qwen2.5-3B-Instruct | 최종 업데이트: 2026-05-23
> **Source of truth**: 이윤제(P1) 결과 기준 (paper_draft_v1.tex와 동기화)
> **표기 변경**: 기존 F (DALR) → **E** 로 통일, 기존 Two-stage E는 폐기

---

## 프로젝트 개요

**목표**: Qwen2.5-3B-Instruct 소형 모델의 **한국어 수학 추론 성능**을 Knowledge Distillation + 이중언어 라우팅으로 향상

**핵심 가설**: Gemini teacher의 영어/한국어 CoT를 student에 전이하면 한국어 수학 추론 능력이 향상된다

**평가 기준**
- 🇰🇷 **주평가**: HRM8K-KO (1,319문제, 사람 검토 고품질 번역)
- 🇺🇸 **보조평가**: GSM8K-EN (1,319문제)

---

## 실험 환경 (P1 setup)

| 항목 | 값 |
|------|-----|
| 모델 | Qwen/Qwen2.5-3B-Instruct |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Teacher | Gemini |
| LoRA r / α | 32 / 64 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Epochs | 3 |
| LR | 2e-4 |
| Effective batch | 8 (per_device=1, grad_accum=8) |
| max_seq_length | 1024 |
| Inference | Greedy (T=0, rep_penalty=1.1), XLSC T=0.7 N=3 |

---

## 학습 데이터 구성

| Setup | 설명 | Train 샘플 | 구성 |
|-------|------|----------:|------|
| **A** | Zero-shot | — | 학습 없음 |
| **B** | EN CoT SFT | 7,283 | Gemini EN CoT (전체 valid) |
| **C** | KO CoT SFT | 6,407 | Gemini KO CoT (전체 valid) |
| **D** | Bilingual SFT | 12,814 | EN + KO 50:50 mix (min×2) |
| **E** (구 F, DALR) | 난이도 기반 언어 라우팅 | 7,327 | KO 6,407 + EN bridge 920 |
| **E_random** (구 F_random) | DALR ablation | 7,327 | KO 6,407 + EN random 920 (같은 EN을 쉬운 문제에) |

**E (DALR) 구성 로직**
```
각 문제 i에 대해:
  cot_ko[i].valid == True
      → KO 시스템 프롬프트 + KO 질문 + KO CoT       (6,407개)
  cot_ko[i].valid == False + cot_en[i].valid == True
      → EN 시스템 프롬프트 + KO 질문 + EN CoT        (920개, bridge)
  둘 다 invalid
      → 폐기                                         (146개)
```

**E_random 구성 로직 (ablation)**
```
E와 동일한 EN bridge 920개를 사용하되,
원래 의도(어려운 문제)와 무관하게 랜덤하게 쉬운 문제(KO valid)에 배치
→ 같은 양의 EN 데이터, 다른 routing
→ "routing이 본질인가, data scaling이 본질인가" 분리 검증
```

**Teacher CoT 유효율**
- EN CoT: 7,283 / 7,473 = **97.5%** valid
- KO CoT: 6,407 / 7,473 = **85.7%** valid

---

## 추론 기법

| 기법 | 설명 |
|------|------|
| **Greedy** | 단일 생성, temperature=0, rep_penalty=1.1 |
| **XLSC** | 동일 E(DALR) 모델로 KO×3 + EN×3 생성 → 6-way 다수결 |
| **Cascade XLSC** | XLSC 3:3 동률 케이스에 EN 1회 추가 → 7-way 홀수 투표 |

**XLSC 핵심 아이디어**
- 동일 모델, 동일 한국어 문제
- KO 시스템 프롬프트로 N번 생성 (한국어 추론)
- EN 시스템 프롬프트로 N번 생성 (E의 bridge 능력 활용 → 영어 추론)
- 2N표 다수결 → KO/EN 오류 패턴 decorrelation 활용

**Cascade XLSC 동기**
- XLSC 3:3 동률 케이스 약 20% 발생 (preliminary)
- 동률 케이스의 정확도가 전체 평균보다 현저히 낮음
- EN tiebreaker: 동률 내 EN 다수결 정답이 KO 다수결보다 약간 우세

---

## 실험 결과 (HRM8K-KO 1,319 / GSM8K-EN 1,319)

| Setup | HRM8K-KO | 95% CI (KO) | GSM8K-EN |
|-------|:--------:|:-----------:|:--------:|
| **A** Zero-shot | 53.90 | [51.2, 56.6] | 71.65 |
| **B** EN CoT SFT | 54.44 | [51.7, 57.2] | 74.22 |
| **C** KO CoT SFT | 59.51 | [56.8, 62.2] | 70.74 |
| **D** Bilingual SFT | 58.98 | [56.2, 61.8] | 74.98 |
| **E** (DALR) | **61.79** 🥇 | [59.1, 64.4] | 69.37 |
| **E_random** (ablation) | 58.83 | [56.2, 61.6] | 72.71 |
| **E + XLSC** | 🔄 실행 중 | TBD | TBD |
| **E + Cascade XLSC** | ⏳ XLSC 후 자동 | TBD | TBD |

### 통계 검정 결과 (1,319 기준)

| 비교 | Test | Result | 의의 |
|------|------|--------|------|
| **E vs E_random** (KO) ⭐ | McNemar | **p = 0.030 \*** | Routing > data scaling 입증 |
| E vs C (KO) | McNemar | TBD | DALR이 단순 KO 학습보다 유의한가 |
| E vs D (KO) | McNemar | TBD | DALR이 bilingual mix보다 유의한가 |
| E + XLSC vs E (KO) | McNemar | TBD | XLSC 효과 |
| Cascade vs XLSC (KO) | McNemar | TBD | Cascade 추가 효과 |

---

## 주요 발견

**1. 영어 CoT(B) 단독은 한국어 학습에 미미**
- B KO 54.44% vs A KO 53.90% (+0.54%p)
- 영어 추론만 학습하면 한국어 문제 풀이 능력 향상 거의 없음

**2. 한국어 CoT(C)가 KO 성능 향상에 가장 직접적인 baseline**
- C KO 59.51% (A 대비 +5.61%p)
- EN도 70.74%로 유지

**3. DALR(E)이 단순 혼합(D)보다 KO에서 우위**
- E 61.79% > D 58.98% (+2.81%p)
- D는 데이터량 12,814 (E의 1.75배)이지만 E가 우세
- → 데이터 양이 아니라 routing 전략이 효과적임을 시사

**4. E vs E_random ablation: "routing이 본질"** ⭐ (메인 주장)
- E 61.79% > E_random 58.83% (+2.96%p)
- McNemar p = 0.030 *
- **같은 920개 EN bridge라도 어디 두느냐가 결정적**
- E_random은 C(59.51) 보다도 낮음 → 무작위 EN 주입은 오히려 해로움

**5. E의 EN 성능 modest trade-off**
- E EN 69.37% vs C EN 70.74% (−1.37%p)
- E EN 69.37% vs D EN 74.98% (−5.61%p)
- 주평가는 KO이므로 허용 가능한 수준
- **EN trade-off framing** (TBD — 논문 작성 시 결정):
  - **옵션 A**: 솔직 인정 — "주평가는 HRM8K-KO. EN drop은 알고 있는 trade-off"
  - **옵션 B**: 의도된 design — "EN bridge는 KO 질문 풀이용으로 routing되므로 by design"

---

## 현재 진행 상태

| 작업 | 상태 | 결과 |
|------|------|------|
| A~D, E, E_random greedy 평가 | ✅ 완료 | 위 표 참조 |
| Bootstrap 95% CI | ✅ 완료 | 위 표 참조 |
| McNemar (E vs E_random) | ✅ 완료 | p=0.030 |
| McNemar 나머지 비교 | ⏳ 추가 계산 필요 | TBD |
| **E + XLSC** (KO×3+EN×3) | 🔄 **이제 시작** | — |
| **E + Cascade XLSC** | ⏳ XLSC 완료 후 | — |
| Llama 3.2-3B robustness | ⏳ 시간 되면 | TBD |
| EXAONE 3.5-2.4B robustness | ⏳ 시간 되면 | TBD |

---

## 파일 경로

```
korean_cot_distill/
├── src/
│   ├── data/
│   │   ├── make_dalr_data.py        E (DALR) 학습 데이터 생성
│   │   └── make_dalr_random_data.py E_random ablation 데이터 생성
│   ├── train/sft.py                 LoRA SFT 학습 (a~f 모두 지원)
│   ├── eval/
│   │   ├── evaluate.py              Greedy 평가 (SETUP_ADAPTER 정의)
│   │   ├── self_consistency.py      단일 모델 SC (참고용)
│   │   ├── xlsc.py                  ⭐ 새로 작성 (XLSC: KO×N + EN×N)
│   │   └── cascade_xlsc.py          ⭐ 새로 작성 (Cascade tie-breaking)
│   └── analysis/
│       └── statistical_tests.py     Bootstrap + McNemar
├── weights/
│   ├── setup_b/, setup_c/, setup_d/  baseline LoRA adapters
│   ├── setup_f/                      ⭐ E (DALR) adapter (파일명은 setup_f 유지)
│   └── setup_f_random/               ⭐ E_random adapter
├── data/
│   ├── eval/hrm8k_ko.jsonl           HRM8K (1,319)
│   ├── eval/gsm8k_test_en.jsonl      GSM8K (1,319)
│   └── teacher_cot/cot_{en,ko}.jsonl Gemini CoT
└── results/
    ├── setup_*_hrm8k.json            greedy 결과
    ├── setup_*_gsm8k.json
    ├── xlsc_e_hrm8k.json             ⭐ 곧 생성
    └── cascade_xlsc_e_hrm8k.json     ⭐ 곧 생성
```

**주의**: 파일시스템에는 `setup_f` 그대로 둠 (rename 안 함). 논문에서만 **E**로 표기.

---

## 다음 액션

1. `src/eval/xlsc.py` 작성 (KO×3 + EN×3 voting)
2. XLSC 실행: `python -m src.eval.xlsc --setup f --bench hrm8k --n 3 --temp 0.7`
3. `src/eval/cascade_xlsc.py` 작성 (tie-breaking)
4. Cascade 실행
5. 통계 검정 추가 계산
6. paper_draft_v1.tex 표 1 채우기 (F→E 표기 일괄 변경)
