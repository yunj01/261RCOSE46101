"""
Statistical tests for the main results (DALR pipeline).

  Bootstrap 95% CI:
    For every setup x bench, resample with replacement (n=1000) to get
    accuracy distribution. Report 2.5% and 97.5% percentiles.

  McNemar test (paired):
    For each key pair (A vs B), compute the 2x2 table:
        b = A correct, B wrong
        c = A wrong, B correct
    Statistic: chi2 = (|b - c| - 1)^2 / (b + c)   (continuity correction)
    p-value: chi2 with 1 dof.
    Two-tailed.

Usage:
  python -m src.analysis.statistical_tests --limit 0     # full 1,319 eval
  python -m src.analysis.statistical_tests --limit 500   # 500-sample subset
"""

import argparse
import json
import math
import random
from pathlib import Path


def chi2_sf_df1(x: float) -> float:
    # Survival function (1 - CDF) of chi-square with 1 dof.
    # chi2_1 = Z^2 with Z ~ N(0,1), so p = 2 * (1 - Phi(sqrt(x))) = erfc(sqrt(x/2)).
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2))


SETUPS = ["a", "b", "c", "d", "e", "e_random"]

KEY_PAIRS = [
    ("e", "c",        "DALR vs KO-only (C)"),
    ("e", "d",        "DALR vs bilingual mix (D)"),
    ("e", "e_random", "DALR vs random EN bridge (ablation)"),
]


def load_details(path: Path) -> dict[str, bool]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("partial"):
        raise ValueError(f"partial result: {path.name}")
    return {item["id"]: bool(item["correct"]) for item in d["details"]}


def bootstrap_ci(correct: list[bool], n_resample: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(correct)
    accs = []
    for _ in range(n_resample):
        sample = [correct[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(sample) / n)
    accs.sort()
    lo = accs[int(0.025 * n_resample)]
    hi = accs[int(0.975 * n_resample)]
    mean = sum(correct) / n
    return mean * 100, lo * 100, hi * 100


def mcnemar(a_correct: dict[str, bool], b_correct: dict[str, bool]) -> dict:
    common = set(a_correct.keys()) & set(b_correct.keys())
    b = c = 0
    for pid in common:
        a_ok = a_correct[pid]
        b_ok = b_correct[pid]
        if a_ok and not b_ok:
            b += 1
        elif not a_ok and b_ok:
            c += 1
    n_disagree = b + c
    if n_disagree == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0, "n_disagree": 0}
    chi2 = (abs(b - c) - 1) ** 2 / n_disagree  # with continuity correction
    p = chi2_sf_df1(chi2)
    return {"b": b, "c": c, "chi2": chi2, "p": p, "n_disagree": n_disagree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=str, default="0",
                    help="0 = full 1,319 eval (default); otherwise loads results/setup_*_{bench}_limit{N}.json")
    ap.add_argument("--n-resample", type=int, default=1000)
    args = ap.parse_args()

    project = Path(__file__).resolve().parent.parent.parent
    results_dir = project / "results"
    suffix = f"_limit{args.limit}" if args.limit and args.limit != "0" else ""

    benches = ["hrm8k", "gsm8k"]

    # ── Bootstrap CI for every (setup, bench) ─────────────────────────────
    print("=" * 78)
    print(f"  Bootstrap 95% CI  (n_resample = {args.n_resample})")
    print("=" * 78)
    print(f"{'Setup':<14}", end="")
    for b in benches:
        print(f"{b.upper():>30}", end="")
    print()
    print("-" * 78)

    setup_details = {b: {} for b in benches}
    for setup in SETUPS:
        row = f"{setup:<14}"
        for b in benches:
            path = results_dir / f"setup_{setup}_{b}{suffix}.json"
            if not path.exists():
                row += f"{'—':>30}"
                continue
            try:
                det = load_details(path)
                setup_details[b][setup] = det
                correct = list(det.values())
                acc, lo, hi = bootstrap_ci(correct, args.n_resample)
                row += f"  {acc:5.2f} [{lo:5.2f},{hi:5.2f}]      "
            except Exception:
                row += f"{'(err)':>30}"
        print(row)

    # ── McNemar paired tests ──────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  McNemar Test  (paired, continuity-corrected, two-tailed)")
    print("=" * 78)

    def fmt_p(p):
        if p < 0.001:
            return "p<0.001 ***"
        if p < 0.01:
            return f"p={p:.3f} **"
        if p < 0.05:
            return f"p={p:.3f} *"
        if p < 0.10:
            return f"p={p:.3f} ~"
        return f"p={p:.3f} (ns)"

    for bench in benches:
        print(f"\n[{bench.upper()}]")
        for a, b, label in KEY_PAIRS:
            if a not in setup_details[bench] or b not in setup_details[bench]:
                continue
            r = mcnemar(setup_details[bench][a], setup_details[bench][b])
            sig = fmt_p(r["p"])
            print(f"  {label:<38} | A>B={r['b']:3d}, B>A={r['c']:3d} | chi2={r['chi2']:5.2f} | {sig}")

    print()
    print("=" * 78)
    print("Legend: A>B = A correct, B wrong (favors A) | B>A = B correct, A wrong (favors B)")
    print("        *** p<0.001  ** p<0.01  * p<0.05  ~ p<0.10  (ns) p>=0.10")
    print("=" * 78)


if __name__ == "__main__":
    main()
