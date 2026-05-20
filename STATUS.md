# 📊 Project Status

> **Topic:** Two-Stage Cross-lingual CoT Distillation for Small Korean Reasoning Models
> **Method:** Stage 1 (English CoT) → Stage 2 (Korean CoT + 10% replay, LR/4) + DALR + CLSC
> **Student model:** Qwen/Qwen2.5-3B-Instruct
> **Teacher model:** gemini-3-flash-preview (단일 teacher, 통제 변인)

---

## ✅ Phase 1: Setup + Data Download (완료)

### Environment
- [x] Python venv (`venv/`)
- [x] Dependencies install (torch+CUDA, transformers, unsloth, google-genai 등)
- [x] GEMINI_API_KEY 환경변수 (영구 등록)
- [x] WANDB API key 등록 (project: `korean-cot-distill`)
- [x] xformers 제거 → PyTorch 네이티브 SDPA 사용 (RTX 5060 Ti Blackwell SM12.0 대응)

### Data
```
data/raw/
  gsm8k_train_en.jsonl   (7,473)
  gsm8k_train_ko.jsonl   (7,473, parallel)
data/eval/
  gsm8k_test_en.jsonl    (1,319)
  gsm8k_test_ko.jsonl    (1,319)
  hrm8k_ko.jsonl         (600)   ⭐ 메인 평가
  kmmlu_subset.jsonl     (500)
```

---

## ✅ Phase 2: Teacher CoT Generation (완료)

| Lang | Total | Valid | 통과율 |
|------|-------|-------|--------|
| EN | 7,473 | 7,283 | 97.5% |
| KO | 7,473 | 6,407 | 85.7% |

```
data/teacher_cot/
  cot_en.jsonl   (7,283 valid)
  cot_ko.jsonl   (6,407 valid)
```

---

## ✅ Phase 3: Training Data Format (완료)

| Setup | 파일 | 샘플 수 | 구성 |
|-------|------|---------|------|
| B | `setup_b_english_cot.jsonl` | 7,283 | EN 100% |
| C | `setup_c_korean_cot.jsonl` | 6,407 | KO 100% |
| D | `setup_d_bilingual_mix.jsonl` | 12,814 | EN 50% + KO 50% |
| E.1 | `setup_e_stage1_english.jsonl` | 7,283 | EN 100% |
| E.2 | `setup_e_stage2_korean_with_replay.jsonl` | 7,119 | KO 90% + EN 10% |

---

## ✅ Phase 4: Fine-tuning (완료)

| Setup | 설명 | train_loss | 시간 | 어댑터 |
|-------|------|-----------|------|--------|
| B | English CoT FT (lr=2e-4, 3ep) | 0.2157 | 2h03m | `weights/setup_b/` |
| C | Korean CoT FT (lr=2e-4, 3ep) | 0.4558 | 2h12m | `weights/setup_c/` |
| D | Bilingual Mix FT (lr=2e-4, 3ep) | 0.3779 | 4h10m | `weights/setup_d/` |
| E1 | Two-stage Stage 1 English (lr=2e-4, 3ep) | 0.2154 | 2h06m | `weights/setup_e_stage1/` |
| E2 | Two-stage Stage 2 KO+replay (lr=5e-5, 2ep) | 0.5290 | 1h35m | `weights/setup_e_final/` |
| F | DALR: KO CoT (6,407) + EN bridge (920) (lr=2e-4, 3ep) | 0.4645 | 2h13m | `weights/setup_f/` |

**총 학습 시간:** ~14h19m  
**비고:** F는 KO CoT 실패 문제 920개에 EN CoT를 bridge로 활용 (총 7,327 샘플)

---

## ✅ Phase 5: Evaluation (완료)

6 setup (A~F) × 2 benchmark + CLSC 앙상블
Single-sample greedy inference, system prompt (zero-shot CoT), repetition_penalty=1.1, 500 samples each

### 단일 모델 결과 매트릭스

| Setup | HRM8K (Korean) ⭐ | GSM8K (English) |
|-------|------------------|-----------------|
| A – Base (no FT) | 52.4% (262/500) | 67.4% (337/500) |
| B – EN CoT FT | 53.6% (268/500) | 73.4% (367/500) |
| C – KO CoT FT | 58.4% (292/500) | 69.2% (346/500) |
| D – Bilingual Mix FT | 55.4% (277/500) | 72.8% (364/500) |
| E – Two-stage FT | 56.6% (283/500) | 72.2% (361/500) |
| **F – DALR (신규)** | **59.2%** (296/500) | 66.2% (331/500) |

### CLSC 앙상블 결과 (Cross-Lingual Self-Consistency)

| Combo | HRM8K ⭐ | GSM8K |
|-------|---------|-------|
| B+C (2-way) | 58.4% | 73.4% |
| F+B (DALR+EN) | 59.2% | 73.4% |
| F+C (DALR+KO) | 58.4% | 66.2% |
| B+C+D (3-way) | 58.4% | 76.8% |
| F+B+C (3-way) | 58.4% | 77.2% |
| B+C+D+E (4-way) | 63.0% | 79.6% |
| **A+B+C+D+E+F (6-way)** | **68.6%** 🚀 | **81.6%** 🚀 |

### 핵심 지표
- **단일 모델 Korean SOTA**: F (DALR) 59.2% — 기존 최고(C 58.4%) +0.8%p
- **CLSC 6-way Korean SOTA**: 68.6% — base 대비 **+16.2%p** 향상
- **CLSC 6-way English SOTA**: 81.6% — base 대비 **+14.2%p** 향상
- **Forgetting metric (E)**: 72.2% / 73.4% = 98.4% retention ✅
- **CLSC forgetting 해소**: 6-way에서 Korean·English 동시 최고 → forgetting 개념 자체 소멸
- **F+B 조합**: HRM8K 59.2% + GSM8K 73.4% — F의 Korean강점 + B의 English강점 상호보완
- 결과 파일: `results/setup_*_limit500.json`, `results/clsc_*.json`

---

## ⏳ Phase 6: Analysis (진행 예정)

### Quantitative
- [x] 5 × 2 결과 매트릭스 표 (Phase 5에서 완료)
- [x] **Forgetting metric:** E GSM8K(72.2%) / B GSM8K(73.4%) = 98.4% ✅
- [x] **Cross-lingual transfer:** B HRM8K(53.6%) vs C HRM8K(58.4%) 확인됨
- [ ] Bar chart 시각화 (setup별 HRM8K/GSM8K 비교)
- [ ] Forgetting curve (E stage1 → E stage2 변화)

### Qualitative
- [ ] 대표 sample 10-20개 비교 (correct/wrong 케이스)
- [ ] Error categorization (단위오류, 수식오류, 언어혼재, step누락)
- [ ] Error distribution 시각화

---

## 🎯 가이드라인 매칭

| 가이드라인 | 우리 답변 |
|-----------|----------|
| Type 1 (Application) | ✅ Korean math reasoning with small model |
| Type 3 (Variant Method) | ✅ DALR (quality-aware CoT routing) + CLSC (cross-lingual ensemble) |
| Type 4 (Analysis) | ✅ Forgetting quantification + Error analysis + Ablation |
| Critical thinking? | ✅ 단순 mix 대비 정량/정성 입증, DALR vs E 비교 |
| Novel contribution? | ✅ DALR (teacher quality signal로 언어 선택) + CLSC (언어 다양성 앙상블) |
| Realistic evaluation? | ✅ Bidirectional (한/영) + 6 single setups + 7 CLSC combos |
| Appropriate baselines? | ✅ A(zero-shot), B(En), C(Ko), D(Mix), E(Two-stage) |

---

## ⏱️ 전체 일정

| Phase | 시간 | 상태 |
|-------|------|------|
| Phase 1 (Setup + Data) | ~2h | ✅ |
| Phase 2 (Gemini CoT) | ~4h | ✅ |
| Phase 3 (데이터 포맷) | 30분 | ✅ |
| Phase 4 (5 FT) | 12h06m | ✅ |
| Phase 5 (평가 A~E) | ~10h (500샘플×10콤보) | ✅ |
| Phase 5b (DALR F + CLSC) | ~5h (학습2.2h+eval2.5h+CLSC5min) | ✅ |
| Phase 6 (분석) | ~3h | ⏳ |
