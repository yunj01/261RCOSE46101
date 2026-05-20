"""
Phase 3: Format teacher CoT into 5 training setups.

Inputs:
  data/teacher_cot/cot_en.jsonl   (7,283 valid)
  data/teacher_cot/cot_ko.jsonl   (6,407 valid)

Outputs (data/train/):
  setup_b_english_cot.jsonl                  # EN only
  setup_c_korean_cot.jsonl                   # KO only
  setup_d_bilingual_mix.jsonl                # EN + KO shuffled (50:50 balanced)
  setup_e_stage1_english.jsonl               # = Setup B (alias)
  setup_e_stage2_korean_with_replay.jsonl    # KO 90% + EN 10% replay

Record schema (per line):
  {
    "id":       original id,
    "lang":     "en" | "ko",
    "question": str,
    "cot":      str,         # full chain-of-thought ending with answer line
    "answer":   str,         # gold numeric
  }
"""

import json
import random
from pathlib import Path


SEED = 42
REPLAY_RATIO = 0.10   # Setup E Stage 2: English replay fraction


def load_valid(path: Path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("valid"):
                continue
            out.append({
                "id":       r["id"],
                "lang":     r["lang"],
                "question": r["question"],
                "cot":      r["cot"],
                "answer":   r["gold"],
            })
    return out


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [OK] {len(records):,} -> {path}")


def main():
    project = Path(__file__).resolve().parent.parent.parent
    cot_dir = project / "data" / "teacher_cot"
    out_dir = project / "data" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Loading valid CoT]")
    en = load_valid(cot_dir / "cot_en.jsonl")
    ko = load_valid(cot_dir / "cot_ko.jsonl")
    print(f"  EN valid: {len(en):,}")
    print(f"  KO valid: {len(ko):,}")

    rng = random.Random(SEED)

    # ---- Setup B: English CoT only ----
    print("\n[Setup B] English CoT only")
    b = en.copy()
    rng.shuffle(b)
    save_jsonl(b, out_dir / "setup_b_english_cot.jsonl")

    # ---- Setup C: Korean CoT only ----
    print("\n[Setup C] Korean CoT only")
    c = ko.copy()
    rng.shuffle(c)
    save_jsonl(c, out_dir / "setup_c_korean_cot.jsonl")

    # ---- Setup D: Bilingual Mix (50:50 balanced) ----
    # Match smaller language count so EN/KO is exact 50:50 (controlled comparison)
    print("\n[Setup D] Bilingual Mix (50:50)")
    n = min(len(en), len(ko))
    en_sample = rng.sample(en, n)
    ko_sample = rng.sample(ko, n)
    d = en_sample + ko_sample
    rng.shuffle(d)
    save_jsonl(d, out_dir / "setup_d_bilingual_mix.jsonl")

    # ---- Setup E Stage 1: English (same as Setup B) ----
    print("\n[Setup E Stage 1] English (alias of Setup B)")
    e1 = en.copy()
    rng.shuffle(e1)
    save_jsonl(e1, out_dir / "setup_e_stage1_english.jsonl")

    # ---- Setup E Stage 2: KO 90% + EN 10% replay ----
    print("\n[Setup E Stage 2] Korean 90% + English 10% replay")
    n_ko = len(ko)
    n_replay = int(round(n_ko * REPLAY_RATIO / (1 - REPLAY_RATIO)))
    n_replay = min(n_replay, len(en))
    en_replay = rng.sample(en, n_replay)
    e2 = ko.copy() + en_replay
    rng.shuffle(e2)
    print(f"  KO: {n_ko:,}, EN replay: {n_replay:,}, total: {len(e2):,}")
    print(f"  Actual ratio: KO={n_ko/len(e2)*100:.1f}%, EN={n_replay/len(e2)*100:.1f}%")
    save_jsonl(e2, out_dir / "setup_e_stage2_korean_with_replay.jsonl")

    print("\n" + "=" * 50)
    print("[DONE] Phase 3 complete - 5 setups formatted")
    print("=" * 50)


if __name__ == "__main__":
    main()
