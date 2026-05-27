"""
Cascade XLSC with raw transformers + 4bit BnB (matched-precision).

Requires matching xlsc_{setup}_n{n}_t{temp}_{bench}_4bit.json from xlsc_4bit.py.

Usage:
  python -m src.eval.cascade_xlsc_4bit --setup e --bench hrm8k --n 3 --temp 0.7
"""
import json, argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval.evaluate import (
    MAX_SEQ_LEN, MAX_NEW_TOKENS,
    SETUP_ADAPTER, BENCH_FILE,
)
from src.eval.xlsc import make_prompt_xlsc, extract_answer_bilingual, vote, is_correct


PREQUANT_BASE = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
FP16_TOK      = "unsloth/Llama-3.2-3B-Instruct"


def run_cascade(setup, bench, project, n, temp, limit, batch_size=8):
    results_dir = project / "results"
    suffix = f"_limit{limit}" if limit else ""
    xlsc_path = results_dir / f"xlsc_{setup}_n{n}_t{temp}_{bench}{suffix}_4bit.json"
    out_path  = results_dir / f"cascade_xlsc_{setup}_n{n}_t{temp}_{bench}{suffix}_4bit.json"

    if not xlsc_path.exists():
        raise FileNotFoundError(
            f"XLSC 4bit input not found: {xlsc_path.name}\n"
            f"Run `python -m src.eval.xlsc_4bit --setup {setup} --bench {bench} --n {n} --temp {temp}` first."
        )

    print(f"\n{'='*60}\n  Cascade XLSC 4bit: setup={setup} bench={bench} N={n}\n{'='*60}")
    xlsc = json.loads(xlsc_path.read_text(encoding="utf-8"))
    if xlsc.get("partial"):
        raise ValueError(f"XLSC 4bit input is partial: {xlsc_path.name}")

    details = xlsc["details"]
    tied_idx = [i for i,d in enumerate(details) if d.get("is_tied", False)]
    print(f"  Total problems: {len(details)}")
    print(f"  Tied problems : {len(tied_idx)}  ({len(tied_idx)/len(details)*100:.2f}%)")

    if not tied_idx:
        print("  No ties — Cascade = XLSC.")
        out = dict(xlsc); out["method"] = "cascade_xlsc_4bit"; out["n_cascade_resamples"] = 0
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    # Load 4bit model
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
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
    model.eval()

    # Generate one extra EN sample on tied problems
    correct_before = sum(1 for d in details if d["correct"])
    print(f"  Batch size: {batch_size}")
    for batch_start in tqdm(range(0, len(tied_idx), batch_size), desc=f"Cascade4 {setup}/{bench}"):
        batch_tied = tied_idx[batch_start:batch_start + batch_size]
        batch_recs = [{"question": details[idx]["question"],
                       "answer": details[idx]["gold"]} for idx in batch_tied]
        prompts = [make_prompt_xlsc(rec, "en", tokenizer) for rec in batch_recs]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=MAX_SEQ_LEN).to("cuda")
        inp_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=temp, top_p=0.95,
                num_return_sequences=1, repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        for k, idx in enumerate(batch_tied):
            d = details[idx]
            gen = tokenizer.decode(out[k][inp_len:], skip_special_tokens=True).strip()
            extra_ans = extract_answer_bilingual(gen)
            en_samples = list(d.get("en_samples", [])) + [extra_ans]
            ko_samples = list(d.get("ko_samples", []))
            voted, counts, tied = vote(ko_samples + en_samples)
            ok = is_correct(voted, d["gold"])
            d["voted"] = voted; d["counts"] = counts; d["is_tied"] = tied
            d["correct"] = ok; d["cascade_extra_en"] = extra_ans

    correct_after = sum(1 for d in details if d["correct"])
    total = len(details)
    print(f"\n  XLSC correct  : {correct_before}/{total} = {correct_before/total*100:.2f}%")
    print(f"  Cascade correct: {correct_after}/{total} = {correct_after/total*100:.2f}%")

    out = {
        "setup": setup, "bench": bench, "n": n, "temp": temp,
        "method": "cascade_xlsc_4bit", "loader": "raw4bit",
        "total": total, "correct": correct_after,
        "accuracy": round(correct_after/total*100, 2),
        "n_cascade_resamples": len(tied_idx),
        "details": details,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved -> {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", default="e")
    p.add_argument("--bench", default="hrm8k")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()
    project = Path(__file__).resolve().parent.parent.parent
    run_cascade(args.setup, args.bench, project, args.n, args.temp, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
