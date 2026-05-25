"""
Cascade XLSC: tie-resolution layer on top of XLSC.

XLSC casts a 2N-way (default 6-way) vote and a non-trivial fraction of problems
land in a tie (e.g. 3-3 split between two answers). Cascade re-runs ONE
additional English sample on tied problems so the vote becomes (2N+1)-way and
the tie is necessarily broken. We re-use the English direction because (a) DALR
routes English CoT specifically to hard Korean-failing problems, and (b) in our
analysis the EN majority is slightly more often correct on tied cases.

Requires: matching `xlsc_{setup}_n{n}_t{temp}_{bench}{suffix}.json` from
`src/eval/xlsc.py`.

Usage:
  python -m src.eval.cascade_xlsc --setup e --bench hrm8k --n 3 --temp 0.7
"""

import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval.evaluate import (
    MODEL_NAME, MAX_SEQ_LEN, MAX_NEW_TOKENS,
    SETUP_ADAPTER, BENCH_FILE,
)
from src.eval.xlsc import (
    make_prompt_xlsc, extract_answer_bilingual, vote, is_correct,
)


def run_cascade(setup: str, bench: str, project: Path, n: int, temp: float,
                limit: int, batch_size: int = 16):
    results_dir = project / "results"
    suffix = f"_limit{limit}" if limit else ""
    xlsc_path = results_dir / f"xlsc_{setup}_n{n}_t{temp}_{bench}{suffix}.json"
    out_path  = results_dir / f"cascade_xlsc_{setup}_n{n}_t{temp}_{bench}{suffix}.json"

    if not xlsc_path.exists():
        raise FileNotFoundError(
            f"XLSC result not found: {xlsc_path.name}\n"
            f"Run `python -m src.eval.xlsc --setup {setup} --bench {bench} "
            f"--n {n} --temp {temp}` first."
        )

    print(f"\n{'='*60}\n  Cascade XLSC: setup={setup}  bench={bench}  N={n}  temp={temp}\n{'='*60}")
    print(f"  XLSC input : {xlsc_path.name}")
    with open(xlsc_path, encoding="utf-8") as f:
        xlsc = json.load(f)
    if xlsc.get("partial"):
        raise ValueError(f"XLSC result is partial; finish it first: {xlsc_path.name}")

    details = xlsc["details"]
    tied_idx = [i for i, d in enumerate(details) if d.get("is_tied", False)]
    print(f"  Total problems : {len(details)}")
    print(f"  Tied problems  : {len(tied_idx)}  ({len(tied_idx)/len(details)*100:.2f}%)")

    if not tied_idx:
        print("  No ties to resolve. Cascade output == XLSC output.")
        out = dict(xlsc); out["method"] = "cascade_xlsc"; out["n_cascade_resamples"] = 0
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  -> {out_path.name}")
        return out

    # ── Load model (raw transformers + PEFT, NOT unsloth) ──────────────
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if adapter_path and adapter_path.exists():
        print(f"  Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    # ── Generate one extra EN sample per tied problem (batched) ─────────
    correct_before = sum(1 for d in details if d["correct"])
    n_flipped_to_correct = 0
    n_flipped_to_wrong  = 0

    print(f"  Batch size: {batch_size}")
    for batch_start in tqdm(range(0, len(tied_idx), batch_size),
                            desc=f"Cascade {setup}/{bench}"):
        batch_tied = tied_idx[batch_start:batch_start + batch_size]
        batch_recs = [{"question": details[idx]["question"],
                       "answer":   details[idx]["gold"]} for idx in batch_tied]

        prompts = [make_prompt_xlsc(rec, "en", tokenizer) for rec in batch_recs]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to("cuda")
        inp_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                num_return_sequences=1,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        for k, idx in enumerate(batch_tied):
            d = details[idx]
            gen = tokenizer.decode(out[k][inp_len:], skip_special_tokens=True).strip()
            extra_ans = extract_answer_bilingual(gen)

            # Recompute vote with extra EN sample (now 2N+1 = odd → no tie)
            en_samples = list(d.get("en_samples", [])) + [extra_ans]
            ko_samples = list(d.get("ko_samples", []))
            all_samples = ko_samples + en_samples
            voted, counts, tied = vote(all_samples)

            before_ok = d["correct"]
            after_ok = is_correct(voted, d["gold"])
            if before_ok and not after_ok:
                n_flipped_to_wrong += 1
            elif (not before_ok) and after_ok:
                n_flipped_to_correct += 1

            d["en_samples"]       = en_samples
            d["cascade_en_extra"] = extra_ans
            d["vote_counts"]      = counts
            d["is_tied"]          = tied
            d["voted"]            = voted
            d["correct"]          = after_ok

    correct_after = sum(1 for d in details if d["correct"])
    total = len(details)
    acc_before = round(correct_before / total * 100, 2)
    acc_after  = round(correct_after  / total * 100, 2)

    out = {
        "setup": setup, "bench": bench, "n": n, "temp": temp,
        "method": "cascade_xlsc",
        "total": total, "correct": correct_after, "accuracy": acc_after,
        "accuracy_before_cascade": acc_before,
        "n_cascade_resamples": len(tied_idx),
        "n_flipped_to_correct": n_flipped_to_correct,
        "n_flipped_to_wrong":   n_flipped_to_wrong,
        "n_remaining_tied":     sum(1 for d in details if d["is_tied"]),
        "details": details,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n  [RESULT] Cascade {setup}/{bench}: {acc_after}% (was {acc_before}%)")
    print(f"           flips: +{n_flipped_to_correct} correct, -{n_flipped_to_wrong} correct")
    print(f"  -> {out_path.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=list(SETUP_ADAPTER.keys()))
    ap.add_argument("--bench", default="hrm8k", choices=["all", "hrm8k", "gsm8k"])
    ap.add_argument("--n",     type=int,   default=3,
                    help="must match the N used in xlsc.py (default 3)")
    ap.add_argument("--temp",  type=float, default=0.7)
    ap.add_argument("--limit", type=int,   default=0)
    ap.add_argument("--batch_size", type=int, default=16,
                    help="tied problems per batch (default 16)")
    args = ap.parse_args()

    project = Path(__file__).resolve().parent.parent.parent
    benches = ["hrm8k", "gsm8k"] if args.bench == "all" else [args.bench]
    for b in benches:
        run_cascade(args.setup, b, project, args.n, args.temp, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
