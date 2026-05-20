"""
Cross-Lingual Self-Consistency (CLSC) evaluation.

All setups use the same seed-42 sub-sample, so problem IDs align perfectly.
This script loads per-problem predictions from multiple setups and applies
majority voting to boost accuracy.

Voting combos tried (whichever result files are available):
  B+C          — English CoT vs Korean CoT  (core CLSC)
  B+C+D        — add Bilingual as tiebreaker
  F+C          — DALR vs Korean CoT
  F+B          — DALR vs English CoT
  F+B+C        — DALR + both mono-lingual adapters
  ALL          — all available setups

Usage:
  python -m src.eval.clsc --bench all --limit 500
"""

import json
import argparse
from pathlib import Path
from collections import Counter


# ── helpers ──────────────────────────────────────────────────────────────────

def load_result(path: Path) -> dict[str, dict]:
    """Returns {problem_id: {predicted, gold, correct}}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("partial"):
        raise ValueError("partial result — skip")
    return {
        d["id"]: {
            "predicted": str(d.get("predicted", "")).strip(),
            "gold":      str(d.get("gold", "")).strip(),
            "correct":   bool(d.get("correct", False)),
        }
        for d in data["details"]
    }


def majority_vote(
    pred_dicts: list[dict],
    setup_names: list[str],
    tiebreak_idx: int,
) -> tuple[list[dict], float, int, int]:
    """Vote across setups, tiebreak by tiebreak_idx setup's answer."""
    all_ids = list(pred_dicts[0].keys())

    details = []
    for pid in all_ids:
        gold   = pred_dicts[0][pid]["gold"]
        votes  = [p[pid]["predicted"] for p in pred_dicts if pid in p]
        counts = Counter(votes)
        max_c  = max(counts.values())
        winners = [ans for ans, c in counts.items() if c == max_c]

        if len(winners) == 1:
            chosen = winners[0]
        else:
            # Tiebreak: prefer the designated setup's answer
            chosen = pred_dicts[tiebreak_idx][pid]["predicted"]

        try:
            ok = abs(float(chosen) - float(gold)) < 0.01
        except (ValueError, TypeError):
            ok = chosen == gold

        details.append({
            "id":         pid,
            "gold":       gold,
            "voted":      chosen,
            "correct":    ok,
            "all_votes":  votes,
            "setups":     setup_names,
        })

    correct  = sum(d["correct"] for d in details)
    total    = len(details)
    accuracy = round(correct / total * 100, 2) if total else 0.0
    return details, accuracy, correct, total


# ── main ─────────────────────────────────────────────────────────────────────

def run_clsc(bench: str, limit_suffix: str, results_dir: Path):
    print(f"\n{'='*60}")
    print(f"  CLSC  bench={bench}")
    print(f"{'='*60}")

    # Load all available completed setup results
    preds: dict[str, dict] = {}
    for setup in ["a", "b", "c", "d", "e", "f", "f_random", "soup_bcf", "soup_bcdef"]:
        path = results_dir / f"setup_{setup}_{bench}{limit_suffix}.json"
        if not path.exists():
            continue
        try:
            preds[setup] = load_result(path)
            print(f"  loaded setup_{setup}: {len(preds[setup])} problems")
        except Exception as e:
            print(f"  skip setup_{setup}: {e}")

    available = list(preds.keys())
    if len(available) < 2:
        print("  Need ≥2 setups — skip")
        return {}

    # Tiebreak preference (index into the list passed to majority_vote)
    # For Korean bench → prefer C; for English bench → prefer B
    def tiebreak_for(combo_setups: list[str]) -> int:
        preferred = "c" if bench == "hrm8k" else "b"
        if preferred in combo_setups:
            return combo_setups.index(preferred)
        return 0

    # Define combos
    def make_combo(name_setups: list[str]) -> tuple[str, list[str]] | None:
        if all(s in preds for s in name_setups):
            return ("+".join(s.upper() for s in name_setups), name_setups)
        return None

    candidates = [
        # original combos
        make_combo(["b", "c"]),
        make_combo(["b", "c", "d"]),
        make_combo(["f", "c"]),
        make_combo(["f", "b"]),
        make_combo(["f", "b", "c"]),
        make_combo(["b", "c", "d", "e"]),
        # NEW: combos with soup_bcdef
        make_combo(["soup_bcdef", "f"]),
        make_combo(["soup_bcdef", "b"]),
        make_combo(["soup_bcdef", "c"]),
        make_combo(["soup_bcdef", "f", "b"]),
        make_combo(["soup_bcdef", "f", "c"]),
        make_combo(["soup_bcdef", "b", "c"]),
        make_combo(["soup_bcdef", "f", "b", "c"]),
        make_combo(["soup_bcdef", "soup_bcf"]),
        # NEW: combos with soup_bcf
        make_combo(["soup_bcf", "f"]),
        make_combo(["soup_bcf", "b", "c"]),
        # full ALL (with new setups)
        ("+".join(s.upper() for s in available), available) if len(available) >= 3 else None,
        # ALL except soup_bcf (since bcf ⊂ bcdef-ish, avoid double counting)
        make_combo(["a", "b", "c", "d", "e", "f", "soup_bcdef"]),
        # 6-way original (for reference / regression check)
        make_combo(["a", "b", "c", "d", "e", "f"]),
    ]
    combos = list({c[0]: c for c in candidates if c is not None}.values())

    bench_summary = {}
    for combo_name, combo_setups in combos:
        pred_list = [preds[s] for s in combo_setups]
        tb        = tiebreak_for(combo_setups)
        details, acc, correct, total = majority_vote(pred_list, combo_setups, tb)

        print(f"  [{combo_name:<12}]  {acc}%  ({correct}/{total})")
        bench_summary[combo_name] = {"accuracy": acc, "correct": correct, "total": total}

        out = {
            "method":   "CLSC",
            "combo":    combo_name,
            "bench":    bench,
            "total":    total,
            "correct":  correct,
            "accuracy": acc,
            "details":  details,
        }
        tag = combo_name.replace("+", "_")
        out_path = results_dir / f"clsc_{tag}_{bench}{limit_suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"    → {out_path.name}")

    return bench_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench",  choices=["hrm8k", "gsm8k", "all"], default="all")
    parser.add_argument("--limit",  type=str, default="500",
                        help="Limit suffix used in eval filenames (0 = no suffix)")
    args = parser.parse_args()

    project     = Path(__file__).resolve().parent.parent.parent
    results_dir = project / "results"
    limit_suffix = f"_limit{args.limit}" if args.limit and args.limit != "0" else ""

    benches      = ["hrm8k", "gsm8k"] if args.bench == "all" else [args.bench]
    all_results  = {}

    for bench in benches:
        all_results[bench] = run_clsc(bench, limit_suffix, results_dir)

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  CLSC SUMMARY")
    print(f"{'='*60}")

    combos_seen = sorted(
        {k for b in all_results.values() for k in b},
        key=lambda x: (len(x), x),
    )
    header = f"{'Combo':<14}" + "".join(f"{b:>14}" for b in benches)
    print(header)
    print("-" * len(header))
    for combo in combos_seen:
        row = f"{combo:<14}"
        for b in benches:
            acc = all_results.get(b, {}).get(combo, {}).get("accuracy", "N/A")
            row += f"{str(acc)+' %':>14}"
        print(row)
    print("="*60)

    # Save summary
    summary_path = results_dir / f"clsc_summary{limit_suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] CLSC summary → {summary_path}")


if __name__ == "__main__":
    main()
