"""
Matched-budget grid analysis: XLSC vs single-language self-consistency.

Reproduces the paper's main XLSC number (HRM8K, Qwen2.5-3B, setup E).

For each of the 20 (3 of 6 KO) x (3 of 3 EN) subset compositions, computes
the majority-vote accuracy. Reports mean and bootstrap 95% CI, alongside
the matched-budget monolingual baseline (KO x 6 self-consistency).

Inputs (already in results/):
  - sc_e_n6_t0.7_hrm8k_promptKO.json   : 6 KO samples per problem
  - xlsc_e_n3_t0.7_hrm8k.json          : 3 EN samples per problem

Output:
  - results/matched_budget_grid.json   : per-subset accuracies and summary

Usage:
  python -m src.analysis.matched_budget_grid
"""
import json
import statistics
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np


def majority(samples):
    """Plurality vote with first-occurrence tiebreak. Matches self_consistency.py."""
    nonempty = [x for x in samples if x != ""]
    if not nonempty:
        return ""
    counts = Counter(nonempty)
    top = max(counts.values())
    for x in nonempty:
        if counts[x] == top:
            return x
    return nonempty[0]


def is_correct(voted, gold):
    if voted == "":
        return False
    try:
        return abs(float(voted) - float(gold)) < 1e-2
    except (ValueError, TypeError):
        return voted.strip() == str(gold).strip()


def bootstrap_ci(values, n_iter=1000, seed=42, alpha=0.05):
    """Bootstrap percentile CI on the mean of values, returned in percent."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = np.fromiter(
        (arr[rng.integers(0, n, n)].mean() * 100 for _ in range(n_iter)),
        dtype=float, count=n_iter,
    )
    means.sort()
    lo = float(means[int(n_iter * alpha / 2)])
    hi = float(means[int(n_iter * (1 - alpha / 2))])
    return lo, hi


def main():
    project = Path(__file__).resolve().parent.parent.parent
    results_dir = project / "results"

    ko6_path  = results_dir / "sc_e_n6_t0.7_hrm8k_promptKO.json"
    xlsc_path = results_dir / "xlsc_e_n3_t0.7_hrm8k.json"
    out_path  = results_dir / "matched_budget_grid.json"

    ko6  = json.load(open(ko6_path,  encoding="utf-8"))
    xlsc = json.load(open(xlsc_path, encoding="utf-8"))
    ko_by_id   = {d["id"]: d for d in ko6["details"]}
    xlsc_by_id = {d["id"]: d for d in xlsc["details"]}
    ids = sorted(set(ko_by_id) & set(xlsc_by_id))
    print(f"Common problems: {len(ids)}")

    n_ko_voted, n_en_voted = 3, 3
    n_ko_pool, n_en_pool = 6, 3

    subsets = []
    per_problem_correct = [0] * len(ids)
    for ko_sub in combinations(range(n_ko_pool), n_ko_voted):
        for en_sub in combinations(range(n_en_pool), n_en_voted):
            c = 0
            for i, pid in enumerate(ids):
                ko_s = [ko_by_id[pid]["samples"][k]      for k in ko_sub]
                en_s = [xlsc_by_id[pid]["en_samples"][k] for k in en_sub]
                if is_correct(majority(ko_s + en_s), ko_by_id[pid]["gold"]):
                    c += 1
                    per_problem_correct[i] += 1
            acc = c / len(ids) * 100
            subsets.append({
                "ko_idx":   list(ko_sub),
                "en_idx":   list(en_sub),
                "correct":  c,
                "total":    len(ids),
                "accuracy": round(acc, 4),
            })

    accs = [s["accuracy"] for s in subsets]
    n_subsets = len(subsets)
    mean = statistics.mean(accs)
    std  = statistics.stdev(accs)

    # Bootstrap 95% CI on the per-problem expected correctness (averaged across subsets)
    exp_correct = [c / n_subsets for c in per_problem_correct]
    ci_lo, ci_hi = bootstrap_ci(exp_correct)

    # KO x 6 (matched-budget monolingual) baseline
    ko6_correct = [int(d["correct"]) for d in ko6["details"]]
    ko6_acc = sum(ko6_correct) / len(ko6_correct) * 100
    ko6_ci_lo, ko6_ci_hi = bootstrap_ci(ko6_correct)

    # Paired sign test: how many subsets exceed pure KO x 6
    n_above = sum(1 for a in accs if a > ko6_acc)
    # one-sided binomial p under H0: P(XLSC > KO x 6) = 0.5
    p_one_sided = sum(comb(n_subsets, k) for k in range(n_above, n_subsets + 1)) / (2 ** n_subsets)

    summary = {
        "description": (
            "Matched-budget (N=6) grid analysis: XLSC vs monolingual KO x 6 "
            "self-consistency on HRM8K, Qwen2.5-3B E (DALR). Reproduces the "
            "XLSC accuracy reported in Table 1 of the paper."
        ),
        "inputs": {
            "ko_pool": str(ko6_path.relative_to(project)),
            "en_pool": str(xlsc_path.relative_to(project)),
        },
        "composition": {
            "n_ko_pool": n_ko_pool, "n_en_pool": n_en_pool,
            "n_ko_voted": n_ko_voted, "n_en_voted": n_en_voted,
        },
        "n_problems":        len(ids),
        "n_subsets":         n_subsets,
        "xlsc_subset_mean":  round(mean, 4),
        "xlsc_subset_std":   round(std, 4),
        "xlsc_bootstrap_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "ko6_baseline_acc":    round(ko6_acc, 4),
        "ko6_bootstrap_95ci": [round(ko6_ci_lo, 4), round(ko6_ci_hi, 4)],
        "cross_lingual_lift_pp": round(mean - ko6_acc, 4),
        "sign_test": {
            "subsets_above_ko6":   n_above,
            "subsets_total":       n_subsets,
            "p_one_sided":         p_one_sided,
        },
        "subsets": subsets,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(f"  KO x 6 (matched-budget monolingual)     : {ko6_acc:6.2f}  CI [{ko6_ci_lo:.2f}, {ko6_ci_hi:.2f}]")
    print(f"  XLSC (3+3, mean over {n_subsets:>2} subsets)        : {mean:6.2f} ± {std:.2f}  CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Cross-lingual lift                       : {mean - ko6_acc:+6.2f} pts")
    print(f"  Subsets above KO x 6                    : {n_above}/{n_subsets}  (one-sided sign test p = {p_one_sided:.2e})")
    print(f"  -> {out_path.relative_to(project)}")


if __name__ == "__main__":
    main()
