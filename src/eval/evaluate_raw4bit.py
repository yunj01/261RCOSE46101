"""
Evaluate with raw transformers + BitsAndBytes 4bit (NO unsloth at inference).

Combines:
- 4bit precision matching the adapter training condition (avoids 4bit→FP16 mismatch)
- Raw transformers + PEFT loader (avoids unsloth's batched-greedy accuracy degradation)

Loads the pre-quantized `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit` directly
so the base weights are bit-identical to the training base.

Output filename: setup_{s}_{bench}_raw4bit.json

Usage:
  python -m src.eval.evaluate_raw4bit --setup c --bench hrm8k --batch_size 8
"""
import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.eval.evaluate import (
    MAX_SEQ_LEN, MAX_NEW_TOKENS, SAMPLE_SEED,
    SETUP_ADAPTER, BENCH_FILE,
    SYSTEM_KO, SYSTEM_EN,
    load_bench, normalize_gold, extract_answer, make_prompt,
)


# Same pre-quantized base unsloth used for training the adapters.
PREQUANT_BASE = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
# Fallback: FP16 source + on-the-fly NF4 quantization matching unsloth's config.
FP16_BASE = "unsloth/Llama-3.2-3B-Instruct"


def load_model_4bit():
    """Try loading pre-quantized base first; fall back to on-the-fly NF4 quant."""
    # Tokenizer always from FP16 mirror (cleaner, same chat template)
    tokenizer = AutoTokenizer.from_pretrained(FP16_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        print(f"  Loading pre-quantized base: {PREQUANT_BASE}")
        model = AutoModelForCausalLM.from_pretrained(
            PREQUANT_BASE,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    except Exception as e:
        print(f"  [WARN] Pre-quantized load failed ({e}); falling back to on-the-fly NF4 quant")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            FP16_BASE,
            quantization_config=bnb,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    return model, tokenizer


def run_eval(setup: str, bench: str, project: Path, limit: int = 0, batch_size: int = 8):
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    bench_path = project / BENCH_FILE[bench]
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{limit}" if limit else ""
    out_path = results_dir / f"setup_{setup}_{bench}_raw4bit{suffix}.json"

    correct = 0
    total = 0
    details = []
    done_ids = set()

    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("total", 0) > 0 and not existing.get("partial", False):
                print(f"\n[SKIP] {setup}/{bench} (raw4bit) already done: {existing['accuracy']}%")
                return existing
            elif existing.get("partial") and existing.get("total", 0) > 0:
                details  = existing["details"]
                correct  = existing["correct"]
                total    = existing["total"]
                done_ids = {d["id"] for d in details}
                print(f"\n[RESUME] {setup}/{bench} raw4bit: {total} done")
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"\n{'='*60}")
    print(f"  Eval (raw 4bit): setup={setup}  bench={bench}")
    print(f"  Adapter: {adapter_path or 'BASE (no adapter)'}")
    print(f"  Bench: {bench_path}")
    print(f"{'='*60}\n")

    model, tokenizer = load_model_4bit()

    if adapter_path and adapter_path.exists():
        print(f"  Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    elif adapter_path:
        print(f"  [WARN] Adapter not found: {adapter_path}, using base only")
    model.eval()

    records = load_bench(bench_path)
    if limit and limit > 0 and limit < len(records):
        import random as _r
        _r.Random(SAMPLE_SEED).shuffle(records)
        records = records[:limit]

    if done_ids:
        records = [r for i, r in enumerate(records)
                   if r.get("id", str(i)) not in done_ids]
        print(f"  Resuming: {len(records)} problems remaining  |  batch_size={batch_size}")
    else:
        print(f"  Loaded {len(records)} problems  |  batch_size={batch_size}")

    last_saved = total

    for batch_start in tqdm(range(0, len(records), batch_size), desc=f"{setup}/{bench} r4"):
        batch_records = records[batch_start:batch_start + batch_size]
        batch_prompts = [make_prompt(rec, bench, tokenizer) for rec in batch_records]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to("cuda")
        inp_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, rec in enumerate(batch_records):
            gen_ids = outputs[j][inp_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = extract_answer(gen_text, bench)
            gold = normalize_gold(rec)
            try:
                ok = abs(float(pred) - float(gold)) < 0.01
            except (ValueError, TypeError):
                ok = pred.strip() == gold.strip()
            correct += int(ok)
            total += 1
            details.append({
                "id":        rec.get("id", f"{batch_start + j}"),
                "question":  rec.get("question", rec.get("problem", "")),
                "gold":      gold,
                "predicted": pred,
                "output":    gen_text,
                "correct":   ok,
            })

        if total - last_saved >= 50:
            partial = {
                "setup": setup, "bench": bench, "loader": "raw4bit",
                "total": total, "correct": correct,
                "accuracy": round(correct/total*100, 2),
                "partial": True, "details": details,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, indent=2)
            last_saved = total

    accuracy = correct / total if total else 0.0
    summary = {
        "setup": setup, "bench": bench, "loader": "raw4bit",
        "total": total, "correct": correct,
        "accuracy": round(accuracy * 100, 2),
        "details": details,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  [RESULT raw4bit] setup={setup} | bench={bench} | Acc={accuracy*100:.2f}% ({correct}/{total})")
    print(f"  Saved -> {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", default="a", choices=list(SETUP_ADAPTER.keys()))
    parser.add_argument("--bench", default="hrm8k", choices=list(BENCH_FILE.keys()))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent.parent
    run_eval(args.setup, args.bench, project, limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
