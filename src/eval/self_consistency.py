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


def run_sc(setup: str, bench: str, project: Path, n: int, temp: float, limit: int,
           prompt_lang: str = "auto", batch_size: int = 1):
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    bench_path = project / BENCH_FILE[bench]
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Override system-prompt + extractor language (for KO*N / EN*N controls
    # on the same Korean test set; ablation for XLSC's cross-lingual claim).
    if prompt_lang == "ko":
        prompt_bench = "hrm8k"
    elif prompt_lang == "en":
        prompt_bench = "gsm8k"
    else:
        prompt_bench = bench
    prompt_suffix = f"_prompt{prompt_lang.upper()}" if prompt_lang != "auto" else ""
    suffix = f"_limit{limit}" if limit else ""
    out_path = results_dir / f"sc_{setup}_n{n}_t{temp}_{bench}{prompt_suffix}{suffix}.json"

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

    # Setup tokenizer for left-padded batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    correct = total = 0
    details = []
    pbar = tqdm(total=len(records), desc=f"SC {setup}/{bench} B={batch_size}")
    last_saved = 0
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start:batch_start + batch_size]
        prompts = [make_prompt(rec, prompt_bench, tokenizer) for rec in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        inp_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            # Generate (batch_size * n) sequences in one call.
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=temp, top_p=0.95,
                num_return_sequences=n,
                repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
        # out shape: [B*N, inp_len + max_new]. Reshape to [B, N, ...]
        out = out.view(len(batch), n, -1)

        for bi, rec in enumerate(batch):
            sample_preds = []
            for j in range(n):
                gen = tokenizer.decode(out[bi, j, inp_len:], skip_special_tokens=True).strip()
                sample_preds.append(extract_answer(gen, prompt_bench))
            voted = majority_answer(sample_preds)
            gold = normalize_gold(rec)
            try:
                ok = abs(float(voted) - float(gold)) < 0.01
            except (ValueError, TypeError):
                ok = voted.strip() == gold.strip()
            correct += int(ok); total += 1
            details.append({"id": rec.get("id", f"{batch_start+bi}"), "gold": gold,
                            "voted": voted, "samples": sample_preds, "correct": ok})

        pbar.update(len(batch))
        # Periodic save every ~50 problems
        if total - last_saved >= 50:
            last_saved = total
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"setup": setup, "bench": bench, "n": n, "temp": temp,
                           "total": total, "correct": correct,
                           "accuracy": round(correct/total*100, 2),
                           "partial": True, "details": details}, f, ensure_ascii=False, indent=2)
    pbar.close()

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
    ap.add_argument("--prompt-lang", default="auto", choices=["auto", "ko", "en"],
                    help="Override system-prompt language (default: matches bench). "
                         "Use 'en' on HRM8K to run the single-language EN*N control.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Number of problems per generate() call. Each batch produces "
                         "(batch_size * n) sequences in parallel on the GPU.")
    args = ap.parse_args()

    project = Path(__file__).resolve().parent.parent.parent
    benches = ["hrm8k", "gsm8k"] if args.bench == "all" else [args.bench]
    for b in benches:
        run_sc(args.setup, b, project, args.n, args.temp, args.limit,
               args.prompt_lang, args.batch_size)


if __name__ == "__main__":
    main()
