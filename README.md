# Difficulty-Aware Language Routing for Korean Mathematical Reasoning

**Korea University COSE461 — Final Project**

Improving Korean mathematical reasoning in a small language model
(Qwen2.5-3B) by leveraging stronger English reasoning, via
**quality-aware cross-lingual aggregation** at three levels: data, model
weights, and inference.

## Overview

Small models reason far better in English than Korean on math problems.
Even the teacher (Gemini) produces lower-quality Korean CoT (85.7%
correct) than English (97.5%), so naive Korean distillation transfers a
degraded signal. We address this with three components:

| Level | Method | Idea |
|-------|--------|------|
| Data | **DALR** (ours) | Per problem, route to a Korean or English teacher CoT by teacher correctness; English CoT used as a *bridge* only on Korean-failed problems. |
| Model | **Model Soup** [Wortsman et al., 2022] | Average LoRA weights of specialist students into one model (1× inference cost). |
| Inference | **Cross-Lingual Ensemble** | Majority-vote across models reasoning in different languages. |

### Key results (HRM8K Korean / GSM8K English, accuracy %)

| Setup | HRM8K | GSM8K |
|-------|-------|-------|
| A (base) | 53.9 | — |
| C (Korean CoT) | 59.5 | 70.7 |
| **F (DALR)** | **61.8** | 69.4 |
| F_random (ablation) | 58.8 | 72.7 |
| Model soup | 60.1 | **78.6** |
| Ensemble (6 + soup) | **70.0** | **84.4** |

*Numbers on the full 1,319 test set (ensemble preliminary). DALR vs.
F_random on HRM8K: +3.0 pts, McNemar p=0.030 — gains come from routing,
not data scaling.*

## Repository structure

```
src/
  data/    # data download, teacher CoT generation, DALR data construction
  train/   # LoRA SFT, model soup
  eval/    # evaluation, cross-lingual ensemble (CLSC)
  analysis/# statistical tests, final report
scripts/   # run scripts
paper/         # paper draft (ACL-style)
paper_template/# course template (NeurIPS 2020) filled with our content
results/   # evaluation outputs (gitignored; regenerate via eval)
weights/   # LoRA adapters (gitignored; too large)
data/      # datasets (gitignored; regenerate via src/data)
```

## Reproduce

```bash
# 1. Environment
python -m venv venv && venv/Scripts/activate   # Windows
pip install -r requirements.txt

# 2. Data: download + teacher CoT + DALR construction
python -m src.data.download
python -m src.data.generate_cot          # needs GEMINI_API_KEY
python -m src.data.make_dalr_data        # setup F
python -m src.data.make_dalr_random_data # setup F_random (ablation)

# 3. Train (example)
python -m src.train.sft --setup f
python -m src.train.make_soup --adapters b c d e_final f --out soup_bcdef

# 4. Evaluate
python -m src.eval.evaluate --setup f --bench hrm8k --limit 0
python -m src.eval.clsc --bench all --limit 0

# 5. Analysis
python -m src.analysis.final_report
```

## Hardware
- RTX 5060 Ti (8GB), Qwen2.5-3B-Instruct (4-bit), LoRA r=32 α=64.

## Team
- (member 1), (member 2), (member 3) — see report appendix.
