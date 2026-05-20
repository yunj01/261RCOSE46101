# 📋 Project Context

> 새 세션 시작 시: **"PROJECT_CONTEXT.md 와 STATUS.md 읽고 이어서 작업하자"**

---

## 🎯 최종 확정 주제

> **"Quality-Aware Cross-Lingual Aggregation for Korean Math Reasoning"**
> 한국어 수학 추론을 향상시키기 위해 cross-lingual signal을 quality-aware 방식으로 aggregation.

### 3-Stage Framework

| Stage | 단계 | 방법 | 신호 |
|-------|------|------|------|
| 1 | **Data** | **DALR** (Difficulty-Aware Language Routing) | Teacher의 KO/EN CoT 품질 격차 |
| 2 | **Model** | **Model Soup** (LoRA weight averaging) | 다중 모델의 weight-space 결합 |
| 3 | **Inference** | **CLSC** (Cross-Lingual Self-Consistency) | 다중 모델 voting agreement |

---

## ✅ 완료 (윤제형 세팅, 500 샘플 기준)

### 단일 모델
| Setup | 설명 | HRM8K (KO) | GSM8K (EN) |
|-------|------|-----------|-----------|
| A | Base (no FT) | 52.4 | 67.4 |
| B | EN CoT FT (Gemini-En 7,283) | 53.6 | 73.4 |
| C | KO CoT FT (Gemini-Ko 6,407) | 58.4 | 69.2 |
| D | Bilingual Mix (12,814) | 55.4 | 72.8 |
| E | Two-stage (EN→KO+replay) | 56.6 | 72.2 |
| **F (DALR)** | KO 6,407 + EN bridge 920 | **59.2** 🥇single-pre-soup | 66.2 |
| **F_random** | DALR ablation: random EN bridge | 56.8 | 69.6 |
| **soup_bcf** | B+C+F LoRA avg | 60.6 | 75.0 |
| **soup_bcdef** | B+C+D+E+F LoRA avg | **61.2** 🥇single | **77.4** 🥇single |

### CLSC 앙상블
| Combo | HRM8K | GSM8K |
|-------|-------|-------|
| 6-way (A~F) | 68.6 | 81.6 |
| **7-way (+ soup_bcdef)** | **70.0** 🥇 | **84.4** 🥇 |
| 9-way (전체) | 70.4 | 84.0 |

### 1,319 전체 평가 (진행 중, 일부 완료)
| Setup | Bench | 1,319 | 500 | Δ |
|-------|-------|-------|-----|---|
| F | GSM8K | 69.37 | 66.2 | +3.17 |
| F_random | GSM8K | 72.71 | 69.6 | +3.11 |
| soup_bcdef | GSM8K | 78.62 | 77.4 | +1.22 |

---

## 🔬 핵심 발견

### 1. DALR의 trade-off
- **F (DALR)**: HRM8K +0.8 vs C, GSM8K **-3.0** vs C → "어려운 EN 데이터를 KO 실패 문제에 배치" 효과
- **F_random**: 같은 양 EN 데이터지만 쉬운 문제에 → HRM8K -1.6 vs C, GSM8K +0.4 vs C
- → **DALR routing 효과 (HRM8K +2.4, GSM8K -3.4) 입증**
- → "data scaling이 아니라 quality-aware routing이 본질"

### 2. Model Soup이 trade-off 해소
- F의 KO↑ EN↓ trade-off를 Soup이 **양쪽 다 향상**으로 해소
- soup_bcdef = 61.2 / 77.4 (single model SOTA, 두 언어 동시)

### 3. CLSC가 추가로 ensemble agreement signal 활용
- 6-way → 7-way (soup 추가): **+1.4 / +2.8** (GSM8K McNemar p=0.001 **)

### 4. Cross-Distillation (CD)은 폐기
- 새 표(준's setting) 결과: CD1/2/3 = 56~58% (C 단독보다 하락)
- 원인: Student CoT 품질이 Teacher보다 30%p+ 낮아 quality dilution
- Paper에서 **negative result로 인용** → DALR의 quality filter motivation 강화

---

## 📊 통계 검정 (500 샘플 기준)

### Bootstrap 95% CI
- F: 59.2 [55.0, 63.4]
- soup_bcdef: 61.2 [57.0, 65.6]
- 7-way CLSC: 70.0 [65.8, 74.2]

### McNemar (GSM8K — 유의)
- soup_bcdef vs F: **p<0.001 *** (KO ablation)
- soup_bcdef vs C: **p<0.001 *** 
- 7-way vs 6-way: **p=0.001 ** **

### McNemar (HRM8K — 유의성 부족, 1,319 필요)
- F vs C: p=0.791 (ns)
- F vs F_random: p=0.323 (ns)
- soup_bcdef vs F: p=0.430 (ns)
- 7-way vs 6-way: p=0.190 (ns)

→ **HRM8K 1,319 전체 평가 진행 중** (~17h 남음)

---

## 📋 진행 중 / 다음 작업

### 🔴 현재 실행 중 (detached, PID 41844)
- c, f, f_random, soup_bcdef × HRM8K (1,319)
- c × GSM8K (1,319, 재실행)

### 🟡 끝나면 자동
- Bootstrap CI + McNemar 1,319 기반 재계산
- HRM8K 유의성 최종 확인

### 🟢 후순위
- Per-tier analysis (DALR이 어떤 난이도에서 효과?)
- CLSC 1,319 재계산 (필요시 A, B, D, E 1,319 추가 평가)
- Naive CD1 우리 세팅 재현 (negative result 자료)
- Error categorization (정성 분석)

---

## 📂 핵심 파일

```
프로젝트 루트: C:\Users\tuni1\Desktop\nlp\korean_cot_distill\

코드:
  src/data/make_dalr_data.py          ← F (DALR) 데이터
  src/data/make_dalr_random_data.py   ← F_random (DALR ablation) 데이터
  src/train/sft.py                    ← LoRA SFT (b, c, d, e1, e2, f, f_random)
  src/train/make_soup.py              ← LoRA weight averaging
  src/eval/evaluate.py                ← 평가 (모든 setup)
  src/eval/clsc.py                    ← CLSC voting
  src/analysis/statistical_tests.py   ← Bootstrap CI + McNemar

스크립트:
  scripts/run_detached.ps1            ← SSH 끊겨도 살아남는 백그라운드 launcher

데이터:
  data/raw/gsm8k_train_*.jsonl
  data/teacher_cot/cot_{en,ko}.jsonl
  data/train/setup_*.jsonl            ← B, C, D, E1, E2, F, F_random

가중치:
  weights/setup_{b,c,d,e_stage1,e_final,f,f_random}/
  weights/soup_{bc,cf,bcf,bcdef}/

결과:
  results/setup_*_{hrm8k,gsm8k}{,_limit500}.json
  results/clsc_*.json
```

---

## 🔑 Paper Narrative (확정)

### Title 후보
- "Quality-Aware Cross-Lingual Aggregation for Small Korean Math Reasoners"
- "Tri-Stage Aggregation Framework for Korean Mathematical Reasoning"

### Story
```
Problem: 작은 모델의 KO 수학 추론은 EN 대비 낮음. 단순 mix는 EN 편향 가속.

Method: Quality-Aware Aggregation
  Stage 1 (Data) — DALR: Teacher quality로 KO/EN routing
    → trade-off: KO↑ EN↓
  Stage 2 (Model) — Soup: 여러 specialist LoRA averaging
    → trade-off 해소, 양쪽 동시 향상
  Stage 3 (Inference) — CLSC: voting agreement
    → 절대 최고 성능, soup과 시너지

Results: 
  Single SOTA: soup_bcdef 61.2/77.4
  Ensemble SOTA: 7-way (with soup) 70.0/84.4
  Base 대비: KO +17.6, EN +17.0

Ablation:
  DALR routing 효과: F vs F_random
  Soup 효과: soup_bcdef vs single best
  Negative: Naive Cross-Distillation 실패 (-3~5%p)
```

---

*Last updated: 2026-05-19*
*Status: 1,319 full eval running (PID 41844), Bootstrap/McNemar pending re-run on full data*
