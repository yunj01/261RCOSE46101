"""
Cross-Lingual Self-Consistency (XLSC).

For each test problem, sample N chains with a Korean system prompt AND N chains
with an English system prompt from the SAME (DALR-trained) model, then majority-
vote over all 2N answers.

Why this works with DALR: setup E is trained on Korean questions paired with
either Korean CoT (when teacher Korean succeeded) or English CoT (when teacher
Korean failed). So the model has learned to produce reasonable answers in EITHER
language for Korean questions; their errors are partially decorrelated.

We also record per-problem tie information so that `cascade_xlsc.py` can re-run
ONE additional English sample on tied problems.

Usage:
  python -m src.eval.xlsc --setup e --bench hrm8k --n 3 --temp 0.7
  python -m src.eval.xlsc --setup e --bench all   --n 3 --temp 0.7 --limit 0
"""

from unsloth import FastLanguageModel

import json
import argparse
from pathlib import Path
from collections import Counter

import re
import torch
from tqdm import tqdm

from src.eval.evaluate import (
    MODEL_NAME, MAX_SEQ_LEN, MAX_NEW_TOKENS, SAMPLE_SEED,
    SETUP_ADAPTER, BENCH_FILE,
    SYSTEM_KO, SYSTEM_EN,
    load_bench, normalize_gold,
)


# ─── XLSC-specific helpers ──────────────────────────────────────────────────

def make_prompt_xlsc(record, system_lang: str, tokenizer):
    """Build chat prompt with a chosen system-prompt language.
    system_lang ∈ {'ko', 'en'}.  The user question is kept in its original
    language (i.e. unchanged from the benchmark)."""
    q = record.get("question", record.get("problem", ""))
    system = SYSTEM_KO if system_lang == "ko" else SYSTEM_EN
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": q},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


_RE_EN = re.compile(r"answer is\s*(?:\$)?(-?[\d,]+\.?\d*)", re.IGNORECASE)
_RE_KO = re.compile(r"답은\s*(?:\$)?(-?[\d,]+\.?\d*)")
_RE_NUM = re.compile(r"-?\d+\.?\d*")


def extract_answer_bilingual(text: str) -> str:
    """Bilingual answer extraction: try both EN ('answer is X') and KO ('답은 X')
    patterns, then fall back to the last number in the text."""
    m = _RE_EN.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    m = _RE_KO.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _RE_NUM.findall(text)
    return nums[-1] if nums else ""


def vote(preds: list[str]) -> tuple[str, dict, bool]:
    """Majority vote with tie detection.

    Returns (voted_answer, counts_dict, is_tied).
    - is_tied = True iff there exist >=2 distinct answers tied for the top count.
    - On a tie, voted_answer is still returned (first-occurrence among the tied
      group) but the caller should treat it as unresolved.
    """
    nonempty = [p for p in preds if p != ""]
    if not nonempty:
        return "", {}, True
    counts = Counter(nonempty)
    top = max(counts.values())
    tied = [a for a, c in counts.items() if c == top]
    is_tied = len(tied) > 1
    if is_tied:
        # first-occurrence tiebreak (not authoritative; cascade may overwrite)
        for p in nonempty:
            if counts[p] == top:
                return p, dict(counts), True
    # clear winner
    for p in nonempty:
        if counts[p] == top:
            return p, dict(counts), False
    return nonempty[0], dict(counts), False


def is_correct(pred: str, gold: str) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-2
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


# ─── XLSC runner ────────────────────────────────────────────────────────────

def run_xlsc(setup: str, bench: str, project: Path, n: int, temp: float, limit: int):
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    bench_path = project / BENCH_FILE[bench]
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{limit}" if limit else ""
    out_path = results_dir / f"xlsc_{setup}_n{n}_t{temp}_{bench}{suffix}_exaone.json"

    # Resume: skip if a completed (non-partial) result already exists.
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("total", 0) > 0 and not existing.get("partial", False):
                print(f"[SKIP] {out_path.name} done: {existing['accuracy']}%")
                return existing
        except Exception:
            pass

    print(f"\n{'='*60}\n  XLSC: setup={setup}  bench={bench}  N={n}  temp={temp}\n{'='*60}")

    # ── Load model ───────────────────────────────────────────────────────
    src_name = str(adapter_path) if (adapter_path and adapter_path.exists()) else MODEL_NAME
    if adapter_path and not adapter_path.exists():
        print(f"  [WARN] Adapter not found: {adapter_path}, using base model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=src_name,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    FastLanguageModel.for_inference(model)

    # ── Load benchmark ──────────────────────────────────────────────────
    records = load_bench(bench_path)
    if limit and 0 < limit < len(records):
        import random as _r
        _r.Random(SAMPLE_SEED).shuffle(records)
        records = records[:limit]
    print(f"  Loaded {len(records)} problems")

    correct = total = 0
    n_tied = 0
    details = []

    for i, rec in enumerate(tqdm(records, desc=f"XLSC {setup}/{bench}")):
        gold = normalize_gold(rec)
        sample_pairs = []  # list of (lang, answer) for all 2N samples

        for system_lang in ("ko", "en"):
            prompt = make_prompt_xlsc(rec, system_lang, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            inp_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    num_return_sequences=n,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for j in range(out.shape[0]):
                gen = tokenizer.decode(out[j][inp_len:], skip_special_tokens=True).strip()
                ans = extract_answer_bilingual(gen)
                sample_pairs.append((system_lang, ans))

        ko_samples = [a for lang, a in sample_pairs if lang == "ko"]
        en_samples = [a for lang, a in sample_pairs if lang == "en"]
        all_samples = ko_samples + en_samples

        voted, counts, tied = vote(all_samples)
        ko_voted, _, _ = vote(ko_samples)
        en_voted, _, _ = vote(en_samples)

        ok = is_correct(voted, gold)
        correct += int(ok)
        total += 1
        if tied:
            n_tied += 1

        details.append({
            "id":          rec.get("id", f"{i}"),
            "question":    rec.get("question", rec.get("problem", "")),
            "gold":        gold,
            "ko_samples":  ko_samples,
            "en_samples":  en_samples,
            "ko_majority": ko_voted,
            "en_majority": en_voted,
            "voted":       voted,
            "is_tied":     tied,
            "vote_counts": counts,
            "correct":     ok,
        })

        # Incremental save every 50 problems
        if (i + 1) % 50 == 0:
            partial = {
                "setup": setup, "bench": bench, "n": n, "temp": temp,
                "method": "xlsc",
                "total": total, "correct": correct,
                "accuracy": round(correct / total * 100, 2),
                "n_tied": n_tied,
                "tied_rate": round(n_tied / total * 100, 2),
                "partial": True, "details": details,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, indent=2)

    acc = round(correct / total * 100, 2) if total else 0.0
    summary = {
        "setup": setup, "bench": bench, "n": n, "temp": temp,
        "method": "xlsc",
        "total": total, "correct": correct, "accuracy": acc,
        "n_tied": n_tied,
        "tied_rate": round(n_tied / total * 100, 2) if total else 0.0,
        "details": details,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  [RESULT] XLSC {setup}/{bench} N={n}: {acc}%  ({correct}/{total})")
    print(f"           ties = {n_tied} ({summary['tied_rate']}%)")
    print(f"  -> {out_path.name}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=list(SETUP_ADAPTER.keys()))
    ap.add_argument("--bench", default="hrm8k", choices=["all", "hrm8k", "gsm8k"])
    ap.add_argument("--n",     type=int,   default=3, help="samples per language (total votes = 2n)")
    ap.add_argument("--temp",  type=float, default=0.7)
    ap.add_argument("--limit", type=int,   default=0,
                    help="0 = use full benchmark; otherwise subsample N problems (seed=42)")
    args = ap.parse_args()

    project = Path(__file__).resolve().parent.parent.parent
    benches = ["hrm8k", "gsm8k"] if args.bench == "all" else [args.bench]
    for b in benches:
        run_xlsc(args.setup, b, project, args.n, args.temp, args.limit)


if __name__ == "__main__":
    main()
