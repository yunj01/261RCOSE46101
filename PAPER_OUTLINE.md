# 📄 Paper Writing Guide — Team Collaboration Doc

> **목적**: 3명이 분담해서 논문 쓰기 위한 통합 outline.
> 각자 자기 섹션 살붙이기 전에 이 문서로 구조 합의.

> **Format**: NeurIPS 2020 (`paper_template/template.tex` 기반)
> **Target length**: 본문 7-8 페이지 + 참고문헌 + 부록
> **Overleaf**: https://ko.overleaf.com/project/6a0d7eb5c5eecf6926fc5ef0

---

## ✅ 최종 확정 사항

### Method (2-stage)
| Stage | Method | 책임 |
|-------|--------|------|
| 1 — Data | **DALR** (Difficulty-Aware Language Routing) | 윤제 |
| 2 — Inference | **XLSC + Cascade** (Cross-Lingual Self-Consistency) | TBD |

### Models
| Role | Model | 비고 |
|------|-------|------|
| **Main** | Qwen2.5-3B-Instruct | 모든 분석 중심 |
| Robustness | Llama-3.2-3B | DALR/XLSC 일반화 입증 |
| Robustness | EXAONE-3.5-2.4B | 한국어 특화 모델 비교 |
| ~~Drop~~ | ~~Phi-3.5-mini~~ | 성능 너무 낮아 유기 |

### Benchmarks
- **HRM8K** (KO, 1,319 문제) ⭐ 메인
- **GSM8K** (EN, 1,319 문제) — robustness/forgetting 측정

### Dropped (논문에서 빼야 할 것)
- ❌ Model Soup
- ❌ CLSC (5-model ensemble)
- ❌ PHLW (Post-hoc Logprob Weighted)
- ❌ BAR / PivotCoT / 검증게이팅 / Self-Distill

---

## 📐 논문 구조 (확정)

```
1. Introduction              ~1 page   [공동]
2. Related Work              ~0.5 page [공동]
3. Approach                  ~2 pages  [DALR 담당 + XLSC 담당 분담]
   3.1 Overview (+ Figure 1)
   3.2 DALR
   3.3 XLSC
   3.4 Cascade XLSC
   3.5 Baselines
4. Experiments               ~1.5 pages [실험 담당]
   4.1 Data
   4.2 Models
   4.3 Implementation
   4.4 Evaluation Protocol
5. Results                   ~1.5 pages [공동, 표 중심]
   5.1 Main Results (Qwen)
   5.2 Robustness (Llama, EXAONE)
   5.3 DALR Ablation (F vs F_random)
6. Analysis                  ~1.5 pages [분석 담당]
   6.1 Per-difficulty Analysis
   6.2 XLSC Tie Statistics + Cascade Effect
   6.3 Cross-Lingual Error Decorrelation
   6.4 Statistical Tests
7. Conclusion                ~0.5 page [공동]
References
Appendix
   A. Team Contributions
   B. Hyperparameters
   C. Additional Tables
```

---

## 👥 3인 분담

### 👤 Person 1 (DALR 담당) — 윤제
**책임 섹션**:
- 3.2 DALR (method 상세)
- 5.1 Main Results (Qwen) — DALR 결과 부분
- 5.3 DALR Ablation (F vs F_random)
- 6.1 Per-difficulty Analysis
- 6.4 Statistical Tests (McNemar, Bootstrap CI)

**핵심 메시지**:
> "DALR의 효과는 'routing'에 있고 'data scaling'이 아니다 (F vs F_random)"

**필요한 자료**:
- F 결과 (HRM8K 1,319 전체)
- F_random 결과
- McNemar p-value, Bootstrap CI
- Per-difficulty breakdown

---

### 👤 Person 2 (XLSC + Cascade 담당)
**책임 섹션**:
- 3.3 XLSC
- 3.4 Cascade XLSC
- 5.1 Main Results (Qwen) — XLSC 결과 부분
- 6.2 XLSC Tie Statistics + Cascade Effect
- 6.3 Cross-Lingual Error Decorrelation

**핵심 메시지**:
> "DALR으로 학습된 모델은 KO/EN 양쪽에서 신뢰 가능한 답 생성 → cross-lingual voting에 최적화"
> "Cascade는 동률 케이스(19.8%)만 타겟해서 효율적으로 해결"

**필요한 자료**:
- XLSC 결과 (KO×3 + EN×3)
- Cascade 결과
- Tie 비율, tie 케이스 정확도
- KO/EN 다수결 일치/불일치 분석

---

### 👤 Person 3 (Experiments + Robustness 담당)
**책임 섹션**:
- 4.1-4.4 전체 (Data, Models, Implementation, Evaluation)
- 5.2 Robustness (Llama, EXAONE 표)
- (선택) 6.4 Statistical Tests 일부

**핵심 메시지**:
> "Qwen에서 발견한 패턴이 Llama, EXAONE에서도 재현 → 방법론의 일반성"

**필요한 자료**:
- Llama 전체 결과 (A, B, C, D, DALR, XLSC, Cascade)
- EXAONE 전체 결과
- 학습 hyperparameter (LoRA, lr, epoch 등)
- 평가 protocol 세부사항

---

### 🤝 공동 작업
- **Abstract**: 마지막에 모두 함께 (현재 template에 있는 것 수정)
- **Introduction**: 윤제가 초안, 모두 review
- **Related Work**: 각자 자기 method 관련 인용 추가
- **Conclusion**: 모두 함께
- **References.bib**: 인용한 사람이 즉시 추가

---

## 📊 핵심 결과 (Qwen 메인)

### Table 1: Single Model Results (HRM8K / GSM8K)

| Setup | HRM8K (KO) | GSM8K (EN) | 비고 |
|-------|-----------|-----------|------|
| A (Base) | 55.8 | 71.4 | Zero-shot baseline |
| B (EN CoT) | 53.3 | 75.1 | Cross-lingual transfer |
| C (KO CoT) | 60.0 | 73.2 | KO single-language baseline |
| D (Bilingual mix) | 60.5 | 75.9 | Naive bilingual |
| **DALR (F)** | 58.9 / 61.79† | 66.9 / 69.37† | ⭐ Stage 1 method |
| F_random (ablation) | 56.8 | 69.6 | DALR routing 효과 ablation |

†1,319 전체 평가 수치

### Table 2: With Inference-Time Aggregation

| Setup | HRM8K (KO) |
|-------|-----------|
| DALR alone | 58.9 |
| DALR + **XLSC** | **69.4** ⭐ Stage 2 (메인 결과) |
| DALR + **XLSC Cascade** | (TBD, Llama 기준 +0.5%p 추가) |

### Table 3: Robustness across Models

| Model | A | C | DALR | DALR+XLSC | DALR+Cascade |
|-------|---|---|------|-----------|--------------|
| Qwen2.5-3B | 55.8 | 60.0 | 58.9 | **69.4** | TBD |
| Llama-3.2-3B | 34.8 | 53.3 | 55.1 | 66.6 | 67.1 |
| EXAONE-3.5-2.4B | 56.1 | 57.7 | 57.4 | N/A | TBD |

---

## 📈 필요한 Figure (제안 4개)

1. **Figure 1 — Pipeline Overview**
   - 2-stage framework 다이어그램
   - Stage 1 (DALR data routing) → Stage 2 (XLSC inference voting)
   - 책임: 윤제 또는 P2

2. **Figure 2 — DALR Routing Logic**
   - Per-problem routing 결정 트리
   - KO valid → KO CoT | KO fail + EN valid → EN bridge | else discard
   - 책임: P1 (윤제)

3. **Figure 3 — XLSC + Cascade Flow**
   - KO×3 + EN×3 voting → tie 발견 → EN+1로 cascade
   - 책임: P2

4. **Figure 4 — Per-Difficulty Analysis**
   - HRM8K 문제를 난이도별로 나눠 DALR vs baselines 비교
   - 책임: P1

---

## 🔬 통계 검정 (Section 6.4)

| 비교 | Test | Expected | 책임 |
|------|------|----------|------|
| F vs F_random (HRM8K) | McNemar | p=0.030 * | P1 |
| F vs C (HRM8K) | McNemar | TBD (1,319 평가 후) | P1 |
| XLSC vs DALR alone | McNemar | TBD | P2 |
| Cascade vs XLSC | McNemar | TBD | P2 |
| 모든 setup | Bootstrap 95% CI | TBD | P1 또는 P3 |

---

## 📝 글쓰기 가이드라인

### 스타일
- **명확하고 간결하게** (NeurIPS 스타일)
- 1인칭 복수 ("We propose...") 사용
- Bold는 핵심 용어 첫 도입 시만
- 인용은 `\citep{}` 사용 (괄호 안)

### 표/그림
- 모든 표/그림에는 caption 필수
- 모든 표/그림은 본문에서 참조 (`Table~\ref{...}`)
- 숫자는 항상 출처 명시 (어느 실험 결과인지)

### 한국어 → 영어 작성
- 모든 글은 영어로 작성
- 한국어로 먼저 쓴 후 번역 OK, 그러나 최종은 자연스러운 영어

### 인용 추가 시
1. `references.bib`에 BibTeX entry 추가
2. 본문에서 `\citep{key}` 또는 `\citet{key}` 사용
3. 중복 확인

---

## 📋 진행 상태 트래킹

각 섹션 상태를 다음으로 표시:
- ⬜ 미작성
- 🟡 초안 작성 중
- 🟢 1차 완료, review 대기
- ✅ 최종 확정

### 현재 상태 (2026-05-20)
| 섹션 | 담당 | 상태 |
|------|------|------|
| Abstract | 공동 | 🟡 초안 (Soup/CLSC 언급 수정 필요) |
| 1. Introduction | 공동 | 🟡 초안 (수정 필요) |
| 2. Related Work | 공동 | 🟡 초안 (XLSC 관련 추가) |
| 3.1 Overview | P1 | ⬜ |
| 3.2 DALR | P1 | 🟢 1차 완료 (template에 있음) |
| 3.3 XLSC | P2 | ⬜ |
| 3.4 Cascade XLSC | P2 | ⬜ |
| 3.5 Baselines | P3 | 🟡 |
| 4. Experiments | P3 | 🟡 |
| 5.1 Main Results (Qwen) | P1+P2 | 🟡 (표 업데이트 필요) |
| 5.2 Robustness | P3 | ⬜ |
| 5.3 DALR Ablation | P1 | 🟡 |
| 6.1 Per-difficulty | P1 | ⬜ |
| 6.2 XLSC Tie | P2 | ⬜ |
| 6.3 Error Decorrelation | P2 | ⬜ |
| 6.4 Statistical | P1/P3 | 🟡 |
| 7. Conclusion | 공동 | 🟡 |
| Appendix A — Team contrib | 공동 | ⬜ |

---

## 🚫 Template에서 제거할 내용

1. **Section 3.3 "Model Soup" 전체 삭제**
2. **Section 3.4 "Cross-Lingual Ensemble"** → "XLSC + Cascade"로 교체
3. **Abstract의 "model soup"/"cross-lingual ensemble" 언급** → "XLSC + Cascade"로 변경
4. **Introduction의 "three-level pipeline"** → "two-level"로 수정
5. **Table 1의 "Model soup (B+C+D+E+F)" 행** → 제거, "DALR + XLSC" 추가
6. **References.bib의 `wortsman2022soup`** → 삭제 가능 (인용 안 함)

---

## 📦 새로 추가할 인용

`references.bib`에 추가 필요:
- GSM8K (Cobbe et al., 2021)
- HRM8K (HAERAE-HUB)
- Qwen2.5 technical report
- LoRA (Hu et al., 2021)
- xCoT / MGSM (Shi et al., 2022) — cross-lingual reasoning
- Magister 2022 — CoT distillation
- (선택) cross-lingual self-consistency 관련

---

## ⏱️ 일정 (제안)

| Day | Task |
|-----|------|
| Day 1 (오늘) | Outline 확정 + template 갈아엎기 + 각자 담당 섹션 확인 |
| Day 2-3 | 각자 초안 작성 |
| Day 4 | 1차 통합 + cross-review |
| Day 5 | 통계 검정 결과 반영 + 표 확정 |
| Day 6 | 그림 추가 + 영어 다듬기 |
| Day 7 | 최종 정리 + 제출 |

---

## 💬 합의 필요한 사항

1. **저자 순서** (3명 누구부터?)
2. **Title 최종 결정**:
   - Option A: "Difficulty-Aware Language Routing for Korean Mathematical Reasoning" (현재)
   - Option B: "Cross-Lingual Distillation and Self-Consistency for Korean Math Reasoning"
   - Option C: "Bridging the English-Korean Gap in Small Math Reasoners via Difficulty-Aware Routing"
3. **각자 담당 섹션 confirm** (위 분담 OK?)
4. **Cascade XLSC를 main result에 포함 vs analysis로?**
   - 현재 안: main에 포함 (Table 2)
   - 대안: main에는 XLSC만, Cascade는 6.2 analysis

---

*Last updated: 2026-05-20*
*Next action: 위 outline confirm → template.tex 갈아엎기 시작*
