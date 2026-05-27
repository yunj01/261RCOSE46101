# Difficulty-Aware Language Routing and Cross-Lingual Self-Consistency for Korean Mathematical Reasoning

**Korea University COSE461 — Final Project · Team 20**

Improving Korean mathematical reasoning in a small language model
(Qwen2.5-3B) by injecting cross-lingual signal at **two stages**:
training (data routing) and inference (cross-lingual voting).

> **이 repo는 팀 공유용 단일 source of truth입니다.**
> 모든 실험은 이 코드/데이터로 재현해주세요. 다른 setup으로 돌린 결과는
> 합치지 마세요. 자세한 협업 규칙은 [`TEAM_RULES.md`](TEAM_RULES.md) 참고.

---

## Overview

Small models reason far better in English than in Korean on math problems.
Even the teacher (Gemini) produces lower-quality Korean CoT (85.7%
correct) than English (97.5%), so naive Korean distillation transfers a
degraded signal. We address this with **two cross-lingual mechanisms**:

| Stage | Method | Idea |
|-------|--------|------|
| Data | **DALR** (ours) | Per-problem routing: KO CoT if teacher Korean is correct, EN CoT as a *bridge* otherwise. |
| Inference | **XLSC + Cascade** (ours) | Sample N×KO + N×EN from the DALR model and vote; resolve 3:3 ties with one extra EN sample. |

We also keep an ablation, **E_random**, with the same number of EN
bridges placed on randomly chosen *easy* problems instead of hard ones,
to isolate the effect of routing from data scaling.

### Key results (HRM8K Korean / GSM8K English, accuracy %)

Numbers on the full 1,319-problem test sets, Qwen2.5-3B-Instruct.

| Setup | HRM8K | GSM8K |
|-------|------:|------:|
| A (base) | 53.90 | 71.65 |
| B (English CoT) | 54.44 | 74.22 |
| C (Korean CoT) | 59.51 | 70.74 |
| D (Bilingual mix) | 58.98 | 74.98 |
| **E (DALR)** | **61.79** | 69.37 |
| E_random (ablation) | 58.83 | 72.71 |
| **E + XLSC** | **68.69** | — |
| **E + Cascade XLSC** | **69.98** | — |

E vs. E_random on HRM8K: **+2.96 pts**, McNemar **p = 0.030** —
gains come from *routing*, not from added data.

> **Setup naming.** Code, data, weights, and results are all aligned on
> **E** / **E_random** (matching the paper). The dropped legacy
> "two-stage E" setup and `setup_f*` naming have been removed.
> Run scripts with `--setup e` / `--setup e_random`.

---

## Repository structure

```
src/
  data/       # download, teacher CoT generation, DALR data construction
  train/      # LoRA SFT
  eval/       # greedy evaluation, single-model SC, XLSC, Cascade XLSC
  analysis/   # bootstrap CI, McNemar tests
config/       # base.yaml — all hyperparameters
scripts/      # run scripts (ps1/bat; local-only)
paper_template/  # NeurIPS-2020 LaTeX template
data/
  raw/        # GSM8K (EN, KO machine-translated) — tracked
  eval/       # HRM8K, GSM8K test, KMMLU, KoBEST — tracked
  teacher_cot/# Gemini CoT (cot_en.jsonl, cot_ko.jsonl) — tracked (~15MB)
  train/      # per-setup SFT datasets (setup_{b,c,d,e_dalr,e_random}.jsonl)
results/      # eval JSONs — tracked (team source of truth)
weights/      # LoRA adapters — NOT tracked (large; share via Drive)
PROJECT_CONTEXT.md  # current state, decisions, TBD list
PAPER_OUTLINE.md    # paper section ownership, deliverables
TEAM_RULES.md       # collaboration rules (read first!)
paper_draft_v1.tex  # working paper draft
```

---

## Setup (per team member)

```powershell
# 1. Clone
git clone https://github.com/yunj01/261RCOSE46101.git
cd 261RCOSE46101

# 2. Python venv + deps
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Set GEMINI_API_KEY only if you need to regenerate teacher CoT
$env:GEMINI_API_KEY = "..."   # PowerShell
```

---

## Reproduce

### From scratch (full pipeline, only needed if regenerating data)

```bash
# Data: download + teacher CoT + DALR construction
python -m src.data.download
python -m src.data.generate_cot              # needs GEMINI_API_KEY ($)
python -m src.data.make_dalr_data            # setup E
python -m src.data.make_dalr_random_data     # setup E_random (ablation)

# Train
python -m src.train.sft --setup b
python -m src.train.sft --setup c
python -m src.train.sft --setup d
python -m src.train.sft --setup e            # DALR
python -m src.train.sft --setup e_random     # ablation
```

### From shared weights (typical for teammates)

```bash
# Greedy evaluation (single model)
python -m src.eval.evaluate --setup e --bench hrm8k --limit 0
python -m src.eval.evaluate --setup e --bench gsm8k --limit 0

# Single-model self-consistency (sanity check baseline)
python -m src.eval.self_consistency --setup e --bench hrm8k --n 6 --temp 0.7

# Cross-Lingual Self-Consistency (XLSC)
python -m src.eval.xlsc --setup e --bench hrm8k --n 3 --temp 0.7

# Cascade XLSC (tie-breaking with one extra EN sample)
python -m src.eval.cascade_xlsc --setup e --bench hrm8k

# Statistics (bootstrap CI + McNemar)
python -m src.analysis.statistical_tests --limit 0
```

---

## Hardware
Training/eval was distributed across team members:
- 이윤제: RTX 5060 Ti (8GB) — Qwen2.5-3B (main pipeline)
- 김상준: RTX 5070 Ti (16GB) — EXAONE-3.5-2.4B robustness
- 원준서: RTX 5070 (12GB) — Llama-3.2-3B robustness

All runs use 4-bit quantization, LoRA r=32 α=64.
Single-model 1,319-problem evaluation: ~30 min.
XLSC (KO×3 + EN×3): ~3 hours per benchmark.

---

## Team 20
| Name | ID | Role |
|------|----|------|
| 이윤제 | 2022320317 | DALR design/training, statistical analysis, paper §3.2 §5 |
| 김상준 | 2022320306 | XLSC/Cascade, tie analysis, paper §3.3 §3.4 |
| 원준서 | 2022320302 | Baselines (A–D), robustness (Llama, EXAONE), paper §4 |

See [`TEAM_RULES.md`](TEAM_RULES.md) for collaboration rules and
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for current state.
