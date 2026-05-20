"""
Setup F: DALR (Difficulty-Aware Language Routing) training data.

Logic per problem (matched by numeric ID suffix):
  - Korean CoT valid   → Korean Q + Korean CoT        (natural reasoning)
  - Korean CoT invalid, English CoT valid
                       → Korean Q + English CoT        (bridge for hard problems)
  - Both invalid       → skip

Output: data/train/setup_f_dalr.jsonl

Usage:
  python -m src.data.make_dalr_data
"""

import json
from pathlib import Path


def extract_suffix(id_str: str) -> str:
    """'gsm8k_en_01458' → '01458'"""
    return id_str.rsplit("_", 1)[1]


def main():
    project = Path(__file__).resolve().parent.parent.parent

    # ── 1. Korean questions (source of Q text for all problems) ──────────────
    ko_questions: dict[str, str] = {}
    with open(project / "data/raw/gsm8k_train_ko.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            suffix = extract_suffix(rec["id"])
            ko_questions[suffix] = rec["question"]
    print(f"[DALR] Korean questions loaded : {len(ko_questions):,}")

    # ── 2. English CoT (valid only) ──────────────────────────────────────────
    en_cot: dict[str, str] = {}
    with open(project / "data/teacher_cot/cot_en.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("valid"):
                suffix = extract_suffix(rec["id"])
                en_cot[suffix] = rec["cot"]
    print(f"[DALR] English CoT valid       : {len(en_cot):,}")

    # ── 3. Korean CoT (valid only) ───────────────────────────────────────────
    ko_cot: dict[str, str] = {}
    with open(project / "data/teacher_cot/cot_ko.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("valid"):
                suffix = extract_suffix(rec["id"])
                ko_cot[suffix] = rec["cot"]
    print(f"[DALR] Korean CoT valid        : {len(ko_cot):,}")

    # ── 4. Build DALR dataset ────────────────────────────────────────────────
    records = []
    n_ko, n_bridge, n_skip = 0, 0, 0

    for suffix, q_ko in ko_questions.items():
        if suffix in ko_cot:
            # Easy problem: Korean reasoning quality is sufficient
            records.append({
                "question": q_ko,
                "cot":      ko_cot[suffix],
                "routing":  "ko_cot",
            })
            n_ko += 1
        elif suffix in en_cot:
            # Hard problem: Korean CoT failed → use high-quality English CoT
            records.append({
                "question": q_ko,
                "cot":      en_cot[suffix],
                "routing":  "en_bridge",
            })
            n_bridge += 1
        else:
            n_skip += 1

    # ── 5. Save ──────────────────────────────────────────────────────────────
    out_path = project / "data/train/setup_f_dalr.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[DALR] Dataset created:")
    print(f"  Korean CoT   (easy problems)   : {n_ko:,}")
    print(f"  English CoT  (hard/bridge)     : {n_bridge:,}")
    print(f"  Skipped (both invalid)         : {n_skip:,}")
    print(f"  Total                          : {len(records):,}")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
