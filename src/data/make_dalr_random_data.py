"""
Setup F_random: DALR ablation (random English bridges).

Control for setup F (DALR). Same total size as F (6,407 KO + 920 EN),
but the 920 EN bridges are sampled from problems where Korean CoT *succeeded*,
not from the hard problems where Korean CoT failed.

Hypothesis test:
  F  > F_random  → "EN on hard problems" matters (DALR claim holds)
  F == F_random  → just data scaling, no real routing effect
  F  < F_random  → DALR is actively harmful (unlikely)

Output: data/train/setup_f_random.jsonl

Usage:
  python -m src.data.make_dalr_random_data [--seed 42] [--n-bridge 920]
"""

import argparse
import json
import random
from pathlib import Path


def extract_suffix(id_str: str) -> str:
    return id_str.rsplit("_", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bridge", type=int, default=920,
                    help="number of EN bridges (match setup F)")
    args = ap.parse_args()
    random.seed(args.seed)

    project = Path(__file__).resolve().parent.parent.parent

    ko_questions: dict[str, str] = {}
    with open(project / "data/raw/gsm8k_train_ko.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ko_questions[extract_suffix(rec["id"])] = rec["question"]
    print(f"[F_random] Korean questions loaded : {len(ko_questions):,}")

    en_cot: dict[str, str] = {}
    with open(project / "data/teacher_cot/cot_en.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("valid"):
                en_cot[extract_suffix(rec["id"])] = rec["cot"]
    print(f"[F_random] English CoT valid       : {len(en_cot):,}")

    ko_cot: dict[str, str] = {}
    with open(project / "data/teacher_cot/cot_ko.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("valid"):
                ko_cot[extract_suffix(rec["id"])] = rec["cot"]
    print(f"[F_random] Korean CoT valid        : {len(ko_cot):,}")

    # KO-success pool for random EN bridge sampling (the control set)
    ko_success_with_en = [s for s in ko_cot if s in en_cot]
    print(f"[F_random] KO-success ∩ EN-valid   : {len(ko_success_with_en):,}")

    if len(ko_success_with_en) < args.n_bridge:
        raise ValueError(
            f"Not enough KO-success problems with EN CoT for {args.n_bridge} bridges."
        )

    bridge_ids = set(random.sample(ko_success_with_en, args.n_bridge))

    # Match F's structure: every KO-success problem keeps its KO CoT,
    # plus 920 random KO-success problems ALSO get an EN CoT entry (duplicated input, different CoT).
    # Total = 6,407 KO + 920 EN = 7,327 (same as F).
    records = []
    n_ko, n_bridge, n_skip = 0, 0, 0

    for suffix, q_ko in ko_questions.items():
        if suffix in ko_cot:
            records.append({
                "question": q_ko,
                "cot":      ko_cot[suffix],
                "routing":  "ko_cot",
            })
            n_ko += 1
        else:
            n_skip += 1

        if suffix in bridge_ids:
            records.append({
                "question": q_ko,
                "cot":      en_cot[suffix],
                "routing":  "en_random_bridge",
            })
            n_bridge += 1

    out_path = project / "data/train/setup_f_random.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[F_random] Dataset created (seed={args.seed}):")
    print(f"  Korean CoT                       : {n_ko:,}")
    print(f"  English CoT (random bridge)      : {n_bridge:,}")
    print(f"  Skipped (KO failed, not sampled) : {n_skip:,}")
    print(f"  Total                            : {len(records):,}")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
