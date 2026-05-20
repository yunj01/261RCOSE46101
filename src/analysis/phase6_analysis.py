"""
Phase 6: Analysis & Visualization

Outputs (saved to analysis/):
  1. fig1_main_bar.png        — A~F 단일 모델 HRM8K/GSM8K 비교
  2. fig2_clsc_bar.png        — CLSC 앙상블 조합별 비교
  3. fig3_tradeoff.png        — HRM8K vs GSM8K scatter (trade-off)
  4. fig4_error_venn.png      — F vs C vs B 오답 overlap (HRM8K)
  5. fig5_dalr_benefit.png    — DALR이 C 대비 추가로 맞춘/틀린 문제 분석
  6. analysis_report.txt      — 텍스트 요약 리포트
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).resolve().parent.parent.parent
RESULTS  = PROJECT / "results"
ANALYSIS = PROJECT / "analysis"
ANALYSIS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "a": "#9E9E9E",   # grey  – base
    "b": "#1976D2",   # blue  – EN CoT
    "c": "#388E3C",   # green – KO CoT
    "d": "#7B1FA2",   # purple– Bilingual
    "e": "#F57C00",   # orange– Two-stage
    "f": "#C62828",   # red   – DALR
}
CLSC_COLOR = "#00838F"   # teal for CLSC bars

# ── load results ────────────────────────────────────────────────────────────
def load_result(setup: str, bench: str):
    path = RESULTS / f"setup_{setup}_{bench}_limit500.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_clsc(combo: str, bench: str):
    tag  = combo.replace("+", "_")
    path = RESULTS / f"clsc_{tag}_{bench}_limit500.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

SETUPS = {
    "a": "A\n(Base)",
    "b": "B\n(EN CoT)",
    "c": "C\n(KO CoT)",
    "d": "D\n(Bilingual)",
    "e": "E\n(Two-stage)",
    "f": "F\n(DALR)",
}
BENCHES = {"hrm8k": "HRM8K (Korean) ⭐", "gsm8k": "GSM8K (English)"}

single = {}
for s in SETUPS:
    single[s] = {}
    for b in ["hrm8k", "gsm8k"]:
        d = load_result(s, b)
        single[s][b] = {"acc": d["accuracy"], "correct": d["correct"], "details": d["details"]}

CLSC_COMBOS = ["B+C", "F+B", "F+C", "B+C+D", "F+B+C", "B+C+D+E", "A+B+C+D+E+F"]
clsc = {}
for combo in CLSC_COMBOS:
    clsc[combo] = {}
    for b in ["hrm8k", "gsm8k"]:
        d = load_clsc(combo, b)
        if d:
            clsc[combo][b] = {"acc": d["accuracy"], "correct": d["correct"]}

print("Data loaded.")

# ════════════════════════════════════════════════════════════════════════════
# Fig 1: Main bar chart — single models
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
fig.suptitle("Single-Model Accuracy: A–F Setups", fontweight="bold", y=1.01)

for ax, bench, bench_label in zip(axes, ["hrm8k", "gsm8k"],
                                   ["HRM8K (Korean Math) ⭐", "GSM8K (English Math)"]):
    keys  = list(SETUPS.keys())
    vals  = [single[s][bench]["acc"] for s in keys]
    cols  = [COLORS[s] for s in keys]
    bars  = ax.bar(list(SETUPS.values()), vals, color=cols, width=0.6, zorder=3)
    base  = single["a"][bench]["acc"]
    ax.axhline(base, color="#9E9E9E", lw=1.4, ls="--", label=f"Base ({base}%)")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f"{v}%",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_title(bench_label, pad=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, 78)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.legend(fontsize=9)

plt.tight_layout()
out = ANALYSIS / "fig1_main_bar.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out.name}")

# ════════════════════════════════════════════════════════════════════════════
# Fig 2: CLSC bar chart
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("CLSC Ensemble Accuracy (majority voting)", fontweight="bold", y=1.01)

all_combos = list(SETUPS.keys()) + CLSC_COMBOS
combo_labels = list(SETUPS.values()) + [c.replace("+", "+\n") for c in CLSC_COMBOS]
is_clsc = [False]*len(SETUPS) + [True]*len(CLSC_COMBOS)

for ax, bench, bench_label in zip(axes, ["hrm8k", "gsm8k"],
                                   ["HRM8K (Korean Math) ⭐", "GSM8K (English Math)"]):
    vals, cols = [], []
    for i, key in enumerate(all_combos):
        if not is_clsc[i]:
            vals.append(single[key][bench]["acc"])
            cols.append(COLORS[key])
        else:
            acc = clsc.get(key, {}).get(bench, {}).get("acc", 0)
            vals.append(acc)
            cols.append(CLSC_COLOR)

    bars = ax.bar(combo_labels, vals, color=cols, width=0.65, zorder=3)
    best_single = max(single[s][bench]["acc"] for s in SETUPS)
    ax.axhline(best_single, color="#C62828", lw=1.4, ls="--",
               label=f"Best single ({best_single}%)")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v}%",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_title(bench_label, pad=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, 88)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.tick_params(axis="x", labelsize=7.5)

    single_patch = mpatches.Patch(color="#555555", label="Single model")
    clsc_patch   = mpatches.Patch(color=CLSC_COLOR, label="CLSC ensemble")
    ax.legend(handles=[single_patch, clsc_patch,
                        mpatches.Patch(color="#C62828", label=f"Best single ({best_single}%)")],
              fontsize=8.5)

plt.tight_layout()
out = ANALYSIS / "fig2_clsc_bar.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out.name}")

# ════════════════════════════════════════════════════════════════════════════
# Fig 3: Trade-off scatter (HRM8K vs GSM8K)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Korean–English Accuracy Trade-off", fontweight="bold")

# Single models
for s, label in SETUPS.items():
    x = single[s]["gsm8k"]["acc"]
    y = single[s]["hrm8k"]["acc"]
    ax.scatter(x, y, s=180, color=COLORS[s], zorder=5)
    offset = (0.3, 0.4)
    if s == "d": offset = (0.3, -1.0)
    if s == "e": offset = (-3.5, 0.4)
    ax.annotate(label.replace("\n", " "), (x, y),
                xytext=(x + offset[0], y + offset[1]), fontsize=9, fontweight="bold",
                color=COLORS[s])

# CLSC key combos
clsc_highlight = {
    "B+C+D+E":     ("B+C+D+E", "^"),
    "A+B+C+D+E+F": ("All-6", "★"),
}
for combo, (lbl, marker) in clsc_highlight.items():
    x = clsc.get(combo, {}).get("gsm8k", {}).get("acc", 0)
    y = clsc.get(combo, {}).get("hrm8k", {}).get("acc", 0)
    if x and y:
        ax.scatter(x, y, s=280, color=CLSC_COLOR, marker="*", zorder=6)
        ax.annotate(lbl, (x, y), xytext=(x + 0.3, y + 0.4),
                    fontsize=9, fontweight="bold", color=CLSC_COLOR)

ax.set_xlabel("GSM8K Accuracy (English) →  higher is better")
ax.set_ylabel("HRM8K Accuracy (Korean) ⭐  →  higher is better")
ax.yaxis.grid(True, alpha=0.3)
ax.xaxis.grid(True, alpha=0.3)

single_patch = mpatches.Patch(color="#555555", label="Single model (A–F)")
clsc_patch   = mpatches.Patch(color=CLSC_COLOR, label="CLSC ensemble")
ax.legend(handles=[single_patch, clsc_patch], fontsize=10)

plt.tight_layout()
out = ANALYSIS / "fig3_tradeoff.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out.name}")

# ════════════════════════════════════════════════════════════════════════════
# Fig 4: DALR benefit analysis (HRM8K: F vs C — what changed?)
# ════════════════════════════════════════════════════════════════════════════
f_det = {d["id"]: d for d in single["f"]["hrm8k"]["details"]}
c_det = {d["id"]: d for d in single["c"]["hrm8k"]["details"]}
b_det = {d["id"]: d for d in single["b"]["hrm8k"]["details"]}

common = set(f_det) & set(c_det) & set(b_det)

f_only   = [pid for pid in common if f_det[pid]["correct"] and not c_det[pid]["correct"]]
c_only   = [pid for pid in common if c_det[pid]["correct"] and not f_det[pid]["correct"]]
both_ok  = [pid for pid in common if f_det[pid]["correct"] and c_det[pid]["correct"]]
both_bad = [pid for pid in common if not f_det[pid]["correct"] and not c_det[pid]["correct"]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("DALR (F) vs Korean CoT (C): HRM8K Problem Analysis", fontweight="bold")

# Left: venn-style bar
ax = axes[0]
categories = ["F only\n(DALR gain)", "C only\n(DALR loss)", "Both correct", "Both wrong"]
counts      = [len(f_only), len(c_only), len(both_ok), len(both_bad)]
bar_cols    = ["#C62828", "#388E3C", "#1976D2", "#9E9E9E"]
bars = ax.bar(categories, counts, color=bar_cols, width=0.55, zorder=3)
for bar, v in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1, str(v),
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Number of problems")
ax.set_title("Outcome breakdown (F vs C, HRM8K)")
ax.yaxis.grid(True, alpha=0.35, zorder=0)

# Right: gold answer length distribution for F-only vs c-only
ax = axes[1]
def ans_len(pids, det):
    return [len(det[p]["gold"]) for p in pids if p in det]

f_gain_lens = ans_len(f_only, f_det)
c_gain_lens = ans_len(c_only, c_det)

bins = range(1, 8)
ax.hist(f_gain_lens, bins=bins, alpha=0.65, color="#C62828", label=f"F gains ({len(f_only)})", density=True)
ax.hist(c_gain_lens, bins=bins, alpha=0.65, color="#388E3C", label=f"C gains ({len(c_only)})", density=True)
ax.set_xlabel("Answer digit length")
ax.set_ylabel("Density")
ax.set_title("Answer complexity: where each method wins")
ax.legend()
ax.yaxis.grid(True, alpha=0.35)

plt.tight_layout()
out = ANALYSIS / "fig4_dalr_vs_c.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out.name}")

# ════════════════════════════════════════════════════════════════════════════
# Fig 5: Per-setup correct count on HRM8K (stacked view)
# ════════════════════════════════════════════════════════════════════════════
# How many problems does each setup uniquely solve?
all_dets = {s: {d["id"]: d["correct"] for d in single[s]["hrm8k"]["details"]} for s in SETUPS}
pids = list(all_dets["a"].keys())

unique_correct = {}
for s in SETUPS:
    others = [k for k in SETUPS if k != s]
    unique = [p for p in pids
              if all_dets[s].get(p) and not any(all_dets[o].get(p) for o in others)]
    unique_correct[s] = len(unique)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Problems uniquely solved by each setup (HRM8K)", fontweight="bold")
keys = list(SETUPS.keys())
vals = [unique_correct[s] for s in keys]
cols = [COLORS[s] for s in keys]
bars = ax.bar(list(SETUPS.values()), vals, color=cols, width=0.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("# problems only this setup solved")
ax.yaxis.grid(True, alpha=0.35, zorder=0)
total_solvable = sum(1 for p in pids if any(all_dets[s].get(p) for s in SETUPS))
ax.set_xlabel(f"(Total problems solvable by at least 1 setup: {total_solvable}/{len(pids)})")
plt.tight_layout()
out = ANALYSIS / "fig5_unique_solved.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out.name}")

# ════════════════════════════════════════════════════════════════════════════
# Text report
# ════════════════════════════════════════════════════════════════════════════
report_lines = []
def rpt(*args): report_lines.append(" ".join(str(a) for a in args))

rpt("="*70)
rpt("PHASE 6 ANALYSIS REPORT")
rpt("="*70)

rpt("\n[1] SINGLE MODEL RESULTS (n=500 each)")
rpt(f"{'Setup':<20} {'HRM8K':>10} {'GSM8K':>10}")
rpt("-"*42)
for s, lbl in SETUPS.items():
    lbl2 = lbl.replace("\n", " ")
    rpt(f"{lbl2:<20} {single[s]['hrm8k']['acc']:>9.1f}% {single[s]['gsm8k']['acc']:>9.1f}%")

rpt("\n[2] CLSC ENSEMBLE RESULTS")
rpt(f"{'Combo':<22} {'HRM8K':>10} {'GSM8K':>10}")
rpt("-"*44)
for combo in CLSC_COMBOS:
    h = clsc.get(combo,{}).get("hrm8k",{}).get("acc","N/A")
    g = clsc.get(combo,{}).get("gsm8k",{}).get("acc","N/A")
    rpt(f"{combo:<22} {str(h)+' %':>10} {str(g)+' %':>10}")

rpt("\n[3] DALR (F) vs KO CoT (C) — HRM8K breakdown")
rpt(f"  Problems F gets right, C gets wrong (DALR gain) : {len(f_only)}")
rpt(f"  Problems C gets right, F gets wrong (DALR loss)  : {len(c_only)}")
rpt(f"  Both correct                                     : {len(both_ok)}")
rpt(f"  Both wrong                                       : {len(both_bad)}")
rpt(f"  Net DALR gain: +{len(f_only)-len(c_only)} problems "
    f"({(len(f_only)-len(c_only))/len(common)*100:.1f}%p)")

rpt("\n[4] UNIQUE CONTRIBUTION per setup (HRM8K)")
for s, lbl in SETUPS.items():
    rpt(f"  {lbl.replace(chr(10),' '):<20}: {unique_correct[s]} problems uniquely solved")

rpt(f"\n  Total solvable by ≥1 setup : {total_solvable}/{len(pids)} "
    f"({total_solvable/len(pids)*100:.1f}%)")

rpt("\n[5] KEY FINDINGS")
best_s_hrm = max(SETUPS, key=lambda s: single[s]["hrm8k"]["acc"])
best_s_gsm = max(SETUPS, key=lambda s: single[s]["gsm8k"]["acc"])
best_clsc_hrm = max(CLSC_COMBOS, key=lambda c: clsc.get(c,{}).get("hrm8k",{}).get("acc",0))
best_clsc_gsm = max(CLSC_COMBOS, key=lambda c: clsc.get(c,{}).get("gsm8k",{}).get("acc",0))

rpt(f"  Best single HRM8K : {best_s_hrm.upper()} ({single[best_s_hrm]['hrm8k']['acc']}%)")
rpt(f"  Best single GSM8K : {best_s_gsm.upper()} ({single[best_s_gsm]['gsm8k']['acc']}%)")
rpt(f"  Best CLSC HRM8K   : {best_clsc_hrm} ({clsc[best_clsc_hrm]['hrm8k']['acc']}%)")
rpt(f"  Best CLSC GSM8K   : {best_clsc_gsm} ({clsc[best_clsc_gsm]['gsm8k']['acc']}%)")
base_hrm = single["a"]["hrm8k"]["acc"]
base_gsm = single["a"]["gsm8k"]["acc"]
rpt(f"  HRM8K gain vs base: +{clsc[best_clsc_hrm]['hrm8k']['acc']-base_hrm:.1f}%p (CLSC)")
rpt(f"  GSM8K gain vs base: +{clsc[best_clsc_gsm]['gsm8k']['acc']-base_gsm:.1f}%p (CLSC)")
forget = single["e"]["gsm8k"]["acc"] / single["b"]["gsm8k"]["acc"] * 100
rpt(f"  Forgetting (E vs B GSM8K): {forget:.1f}% retention")

rpt("\n" + "="*70)

report_text = "\n".join(report_lines)
rpt_path = ANALYSIS / "analysis_report.txt"
rpt_path.write_text(report_text, encoding="utf-8")
print(f"Saved: {rpt_path.name}")
print("\n" + report_text)
