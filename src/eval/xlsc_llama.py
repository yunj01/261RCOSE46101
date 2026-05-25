"""
XLSC with raw transformers + 4bit BnB (matched-precision evaluation).

Loads `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit` directly (the same
base used during QLoRA training) and applies PEFT adapter on top.

Output filename: xlsc_{setup}_n{n}_t{temp}_{bench}_4bit.json

Usage:
  python -m src.eval.xlsc_4bit --setup e --bench hrm8k --n 3 --temp 0.7 --batch_size 4
"""
import json, argparse, re
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval.evaluate import (
    MAX_SEQ_LEN, MAX_NEW_TOKENS, SAMPLE_SEED,
    SETUP_ADAPTER, BENCH_FILE,
    SYSTEM_KO, SYSTEM_EN,
    load_bench, normalize_gold,
)
# Reuse helpers from FP16 xlsc
from src.eval.xlsc import (
    make_prompt_xlsc, extract_answer_bilingual, vote, is_correct, generate_batch,
)


PREQUANT_BASE = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
FP16_TOK      = "unsloth/Llama-3.2-3B-Instruct"


def run_xlsc(setup, bench, project, n, temp, limit, batch_size=4):
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    bench_path = project / BENCH_FILE[bench]
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{limit}" if limit else ""
    out_path = results_dir / f"xlsc_{setup}_n{n}_t{temp}_{bench}{suffix}_4bit.json"

    # Resume
    correct = total = n_tied = 0
    details, done_ids = [], set()
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("total",0) > 0 and not existing.get("partial",False):
                print(f"[SKIP] {out_path.name} done: {existing['accuracy']}%")
                return existing
            elif existing.get("partial") and existing.get("total",0) > 0:
                details  = existing["details"]
                correct  = existing["correct"]
                total    = existing["total"]
                n_tied   = existing.get("n_tied", 0)
                done_ids = {d["id"] for d in details}
                print(f"[RESUME] {out_path.name}: {total} done, {len(done_ids)} ids skipped")
        except Exception:
            pass

    print(f"\n{'='*60}\n  XLSC 4bit: setup={setup} bench={bench} N={n} temp={temp} bs={batch_size}\n{'='*60}")

    # Load model: 4bit base + PEFT
    tokenizer = AutoTokenizer.from_pretrained(FP16_TOK, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"  Loading pre-quantized base: {PREQUANT_BASE}")
    model = AutoModelForCausalLM.from_pretrained(
        PREQUANT_BASE,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if adapter_path and adapter_path.exists():
        print(f"  Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    elif adapter_path:
        print(f"  [WARN] Adapter not found: {adapter_path}")
    model.eval()

    # Load benchmark
    records = load_bench(bench_path)
    if limit and 0 < limit < len(records):
        import random as _r
        _r.Random(SAMPLE_SEED).shuffle(records)
        records = records[:limit]
    if done_ids:
        records = [r for i,r in enumerate(records) if r.get("id", str(i)) not in done_ids]
        print(f"  Resuming: {len(records)} problems remaining")
    else:
        print(f"  Loaded {len(records)} problems  |  bs={batch_size}  n={n}")

    last_saved = total
    for batch_start in tqdm(range(0, len(records), batch_size), desc=f"XLSC4 {setup}/{bench}"):
        batch_recs = records[batch_start:batch_start + batch_size]
        ko_prompts = [make_prompt_xlsc(rec, "ko", tokenizer) for rec in batch_recs]
        en_prompts = [make_prompt_xlsc(rec, "en", tokenizer) for rec in batch_recs]
        ko_results = generate_batch(model, tokenizer, ko_prompts, n, temp)
        en_results = generate_batch(model, tokenizer, en_prompts, n, temp)
        for j, rec in enumerate(batch_recs):
            gold = normalize_gold(rec)
            ko_samples = ko_results[j]
            en_samples = en_results[j]
            voted, counts, tied = vote(ko_samples + en_samples)
            ok = is_correct(voted, gold)
            correct += int(ok); total += 1
            if tied: n_tied += 1
            details.append({
                "id": rec.get("id", f"{batch_start + j}"),
                "question": rec.get("question", ""),
                "gold": gold,
                "ko_samples": ko_samples, "en_samples": en_samples,
                "voted": voted, "counts": counts, "is_tied": tied, "correct": ok,
            })
        if total - last_saved >= 25:
            partial = {
                "setup": setup, "bench": bench, "n": n, "temp": temp,
                "loader": "raw4bit",
                "total": total, "correct": correct, "n_tied": n_tied,
                "accuracy": round(correct/total*100, 2), "partial": True, "details": details,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, indent=2)
            last_saved = total

    accuracy = correct/total if total else 0.0
    summary = {
        "setup": setup, "bench": bench, "n": n, "temp": temp,
        "loader": "raw4bit",
        "total": total, "correct": correct, "n_tied": n_tied,
        "accuracy": round(accuracy*100, 2), "details": details,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  [RESULT XLSC 4bit] {setup}/{bench} = {accuracy*100:.2f}% ({correct}/{total}) tied={n_tied}")
    print(f"  Saved -> {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", default="e", choices=list(SETUP_ADAPTER.keys()))
    p.add_argument("--bench", default="hrm8k", choices=list(BENCH_FILE.keys()))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()
    project = Path(__file__).resolve().parent.parent.parent
    run_xlsc(args.setup, args.bench, project, args.n, args.temp, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
