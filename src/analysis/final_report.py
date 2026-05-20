"""
Final paper report: when all 1,319 eval results are in, generate the full table set.

Generates:
  1. Main accuracy table (all setups × bench × {500, 1319})
  2. Bootstrap 95% CI on 1,319
  3. McNemar paired tests on 1,319
  4. Per-difficulty breakdown (problems stratified by how many setups got it right)
  5. CoT language usage breakdown

Usage:
  python -m src.analysis.final_report
"""

import json
import math
import random
from pathlib import Path
from collections import defaultdict


SETUPS_SINGLE = ["a", "b", "c", "d", "e", "f", "f_random", "soup_bcf", "soup_bcdef"]
CLSC_COMBOS   = ["A_B_C_D_E_F", "A_B_C_D_E_F_SOUP_BCDEF"]

KEY_PAIRS = [
    ("f", "c",          "F (DALR) vs C (KO-only)"),
    ("f", "f_random",   "F (DALR) vs F_random (random EN bridge)"),
    ("soup_bcdef", "f", "soup_bcdef vs F"),
    ("soup_bcdef", "c", "soup_bcdef vs C"),
]
CLSC_PAIRS = [
    ("A_B_C_D_E_F_SOUP_BCDEF", "A_B_C_D_E_F", "7-way (with soup) vs 6-way"),
]


def chi2_sf_df1(x: float) -> float:
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2))


def load_details(path: Path) -> dict[str, bool] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("partial"):
            return None
        return {item["id"]: bool(item["correct"]) for item in d["details"]}
    except Exception:
        return None


def bootstrap_ci(correct: list[bool], n_resample: int = 1000, seed: int = 42):
    rng = random.Random(seed)
    n = len(correct)
    accs = []
    for _ in range(n_resample):
        sample = [correct[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(sample) / n)
    accs.sort()
    lo = accs[int(0.025 * n_resample)] * 100
    hi = accs[int(0.975 * n_resample)] * 100
    return sum(correct) / n * 100, lo, hi


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict:
    common = set(a) & set(b)
    win_a = win_b = 0
    for pid in common:
        if a[pid] and not b[pid]: win_a += 1
        elif not a[pid] and b[pid]: win_b += 1
    disagree = win_a + win_b
    if disagree == 0:
        return {"win_a": 0, "win_b": 0, "chi2": 0.0, "p": 1.0, "n": len(common)}
    chi2 = (abs(win_a - win_b) - 1) ** 2 / disagree
    return {"win_a": win_a, "win_b": win_b, "chi2": chi2, "p": chi2_sf_df1(chi2), "n": len(common)}


def fmt_p(p: float) -> str:
    if p < 0.001: return "***p<0.001"
    if p < 0.01:  return f"** p={p:.3f}"
    if p < 0.05:  return f"*  p={p:.3f}"
    if p < 0.10:  return f"~  p={p:.3f}"
    return f"   p={p:.3f}"


def per_difficulty_breakdown(details_by_setup: dict[str, dict[str, bool]], focus_setup: str = "f"):
    """For each problem, count how many setups got it right. Then show focus_setup's accuracy by bucket."""
    if focus_setup not in details_by_setup:
        return None
    all_ids = set(details_by_setup[focus_setup].keys())
    for s, d in details_by_setup.items():
        all_ids &= set(d.keys())

    n_models = len(details_by_setup)
    buckets = defaultdict(lambda: [0, 0])  # n_correct -> [focus_correct, total]

    for pid in all_ids:
        n_correct = sum(1 for s in details_by_setup if details_by_setup[s][pid])
        focus_ok = details_by_setup[focus_setup][pid]
        buckets[n_correct][0] += int(focus_ok)
        buckets[n_correct][1] += 1

    return buckets, n_models


def main():
    project = Path(__file__).resolve().parent.parent.parent
    results_dir = project / "results"

    print("=" * 86)
    print("  FINAL REPORT - Quality-Aware Cross-Lingual Aggregation for Korean Math Reasoning")
    print("=" * 86)

    # ── 1. Main table: accuracy on 1,319 (with 500 as fallback) ────────────
    print("\n[1] Main Accuracy Table\n" + "-" * 86)
    print(f"{'Setup':<14} {'HRM8K_1319':<14} {'HRM8K_500':<14} {'GSM8K_1319':<14} {'GSM8K_500':<14}")

    details_full = {b: {} for b in ["hrm8k", "gsm8k"]}
    for setup in SETUPS_SINGLE:
        row = f"{setup:<14}"
        for bench in ["hrm8k", "gsm8k"]:
            for limit in ["", "_limit500"]:
                path = results_dir / f"setup_{setup}_{bench}{limit}.json"
                partial = False
                if path.exists():
                    try:
                        with open(path, encoding="utf-8") as f:
                            d = json.load(f)
                        partial = d.get("partial", False)
                        marker = "*" if partial else ""
                        acc = d["accuracy"]
                        n = d["total"]
                        cell = f"{acc:5.2f}({n}){marker}"
                    except Exception:
                        cell = "(err)"
                else:
                    cell = "-"
                row += f" {cell:<14}"
                if limit == "" and not partial and path.exists():
                    dd = load_details(path)
                    if dd is not None:
                        details_full[bench][setup] = dd
        print(row)

    print()
    for combo in CLSC_COMBOS:
        row = f"{combo:<14}"
        for bench in ["hrm8k", "gsm8k"]:
            for limit in ["", "_limit500"]:
                path = results_dir / f"clsc_{combo}_{bench}{limit}.json"
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        d = json.load(f)
                    cell = f"{d['accuracy']:5.2f}({d['total']})"
                else:
                    cell = "-"
                row += f" {cell:<14}"
        print(row)

    # ── 2. Bootstrap CI on 1,319 ────────────────────────────────────────────
    print("\n[2] Bootstrap 95% CI (1,319 samples)\n" + "-" * 86)
    print(f"{'Setup':<14} {'HRM8K':<26} {'GSM8K':<26}")
    for setup in SETUPS_SINGLE:
        row = f"{setup:<14}"
        for bench in ["hrm8k", "gsm8k"]:
            d = details_full[bench].get(setup)
            if d:
                acc, lo, hi = bootstrap_ci(list(d.values()))
                cell = f"{acc:5.2f} [{lo:5.2f},{hi:5.2f}]"
            else:
                cell = "-"
            row += f" {cell:<26}"
        print(row)

    # ── 3. McNemar on 1,319 ─────────────────────────────────────────────────
    print("\n[3] McNemar Paired Test (1,319 samples)\n" + "-" * 86)
    for bench in ["hrm8k", "gsm8k"]:
        print(f"\n[{bench.upper()}]")
        for a, b, label in KEY_PAIRS:
            da = details_full[bench].get(a)
            db = details_full[bench].get(b)
            if da and db:
                r = mcnemar(da, db)
                print(f"  {label:<42}  A>B={r['win_a']:4d}  B>A={r['win_b']:4d}  {fmt_p(r['p'])}")
            else:
                print(f"  {label:<42}  (one or both missing)")

    # ── 4. Per-difficulty breakdown (HRM8K) ────────────────────────────────
    print("\n[4] Per-Difficulty Breakdown - does F help on harder problems? (HRM8K)\n" + "-" * 86)
    n_singles = sum(1 for s in SETUPS_SINGLE if s in details_full["hrm8k"])
    if "f" in details_full["hrm8k"] and n_singles >= 3:
        result = per_difficulty_breakdown(details_full["hrm8k"], focus_setup="f")
        if result:
            buckets, n_models = result
            print(f"{'# models correct':<20} {'F acc':<12} {'(F correct/total)':<20}")
            for k in sorted(buckets.keys()):
                f_correct, total = buckets[k]
                acc = 100 * f_correct / total if total else 0
                print(f"  {k:>3}/{n_models:<13}  {acc:5.2f}%       ({f_correct}/{total})")
        # Same for soup_bcdef
        if "soup_bcdef" in details_full["hrm8k"]:
            print()
            result = per_difficulty_breakdown(details_full["hrm8k"], focus_setup="soup_bcdef")
            if result:
                buckets, n_models = result
                print(f"  (same buckets, soup_bcdef as focus)")
                for k in sorted(buckets.keys()):
                    c, t = buckets[k]
                    acc = 100 * c / t if t else 0
                    print(f"  {k:>3}/{n_models:<13}  {acc:5.2f}%       ({c}/{t})")
    else:
        print("  (Need at least 3 setups with 1,319 results to compute)")

    print("\n" + "=" * 86)
    print("  Legend: *** p<0.001  ** p<0.01  * p<0.05  ~ p<0.10")
    print("=" * 86)


if __name__ == "__main__":
    main()
