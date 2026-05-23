"""
Single-model Self-Consistency (SC).

Sample N chains from ONE model (temperature > 0) and majority-vote the answer.
Contrast with CLSC (clsc.py), which votes across multiple DISTINCT models.

This lets us test: does the gain come from "more samples" (single-model SC)
or from "cross-lingual diversity" (multi-model CLSC)? Compare at equal N.

Usage:
  python -m src.eval.self_consistency --setup f --bench hrm8k --n 5 --temp 0.7 --limit 0
  python -m src.eval.self_consistency --setup f --bench all --n 5 --limit 500
"""

from unsloth import FastLanguageModel

import json
import argparse
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm

# Reuse the exact same prompting / extraction as the main eval (keeps setting identical)
from src.eval.evaluate import (
    MODEL_NAME, MAX_SEQ_LEN, MAX_NEW_TOKENS, SAMPLE_SEED,
    SETUP_ADAPTER, BENCH_FILE,
    load_bench, make_prompt, extract_answer, normalize_gold,
)


def majority_answer(preds: list[str]) -> str:
    """Most common non-empty prediction; ties broken by first occurrence."""
    nonempty = [p for p in preds if p != ""]
    if not nonempty:
        return ""
    counts = Counter(nonempty)
    top = max(counts.values())
    for p in nonempty:               # first-occurrence tiebreak
        if counts[p] == top:
            return p
    return nonempty[0]


def run_sc(setup: str, bench: str, project: Path, n: int, temp: float, limit: int):
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    bench_path = project / BENCH_FILE[bench]
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{limit}" if limit else ""
    out_path = results_dir / f"sc_{setup}_n{n}_t{temp}_{bench}{suffix}.json"

    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("total", 0) > 0 and not existing.get("partial", False):
                print(f"[SKIP] {out_path.name} done: {existing['accuracy']}%")
                return existing
        except Exception:
            pass

    print(f"\n{'='*60}\n  SC: setup={setup} bench={bench} N={n} temp={temp}\n{'='*60}")

    if adapter_path and adapter_path.exists():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path), max_seq_length=MAX_SEQ_LEN,
            dtype=None, load_in_4bit=True, attn_implementation="sdpa")
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN,
            dtype=None, load_in_4bit=True, attn_implementation="sdpa")
    FastLanguageModel.for_inference(model)

    records = load_bench(bench_path)
    if limit and 0 < limit < len(records):
        import random as _r
        _r.Random(SAMPLE_SEED).shuffle(records)
        records = records[:limit]
    print(f"  Loaded {len(records)} problems")

    correct = total = 0
    details = []
    for i, rec in enumerate(tqdm(records, desc=f"SC {setup}/{bench}")):
        prompt = make_prompt(rec, bench, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        sample_preds = []
        inp_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            # Generate all N samples in ONE batched call (parallel on GPU).
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=temp, top_p=0.95,
                num_return_sequences=n,
                repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
            for j in range(out.shape[0]):
                gen = tokenizer.decode(out[j][inp_len:], skip_special_tokens=True).strip()
                sample_preds.append(extract_answer(gen, bench))

        voted = majority_answer(sample_preds)
        gold = normalize_gold(rec)
        try:
            ok = abs(float(voted) - float(gold)) < 0.01
        except (ValueError, TypeError):
            ok = voted.strip() == gold.strip()
        correct += int(ok); total += 1
        details.append({"id": rec.get("id", f"{i}"), "gold": gold,
                        "voted": voted, "samples": sample_preds, "correct": ok})

        if (i + 1) % 50 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"setup": setup, "bench": bench, "n": n, "temp": temp,
                           "total": total, "correct": correct,
                           "accuracy": round(correct/total*100, 2),
                           "partial": True, "details": details}, f, ensure_ascii=False, indent=2)

    acc = round(correct/total*100, 2) if total else 0.0
    summary = {"setup": setup, "bench": bench, "n": n, "temp": temp,
               "method": "single-model-SC", "total": total, "correct": correct,
               "accuracy": acc, "details": details}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  [RESULT] SC {setup}/{bench} N={n}: {acc}% ({correct}/{total})\n  -> {out_path.name}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=list(SETUP_ADAPTER.keys()))
    ap.add_argument("--bench", default="all", choices=["all", "hrm8k", "gsm8k"])
    ap.add_argument("--n", type=int, default=5, help="number of samples to vote over")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    project = Path(__file__).resolve().parent.parent.parent
    benches = ["hrm8k", "gsm8k"] if args.bench == "all" else [args.bench]
    for b in benches:
        run_sc(args.setup, b, project, args.n, args.temp, args.limit)


if __name__ == "__main__":
    main()
