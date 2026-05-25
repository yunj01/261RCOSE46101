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

import json
import argparse
from pathlib import Path
from collections import Counter

import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval.evaluate import (
    MODEL_NAME, MAX_SEQ_LEN, MAX_NEW_TOKENS, SAMPLE_SEED,
    SETUP_ADAPTER, BENCH_FILE,
    SYSTEM_KO, SYSTEM_EN,
    load_bench, normalize_gold,
)


# ─── XLSC-specific helpers ──────────────────────────────────────────────────

def make_prompt_xlsc(record, system_lang: str, tokenizer):
    """Build chat prompt with a chosen system-prompt language.
    system_lang in {'ko', 'en'}.  The user question is kept in its original
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


def vote(preds: list) -> tuple:
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
        for p in nonempty:
            if counts[p] == top:
                return p, dict(counts), True
    for p in nonempty:
        if counts[p] == top:
            return p, dict(counts), False
    return nonempty[0], dict(counts), False


def is_correct(pred: str, gold: str) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-2
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


# ─── Batched generation helper ──────────────────────────────────────────────

def generate_batch(model, tokenizer, prompts: list, n: int, temp: float) -> list:
    """Generate n samples for each prompt in the batch.

    Args:
        prompts: list of B prompt strings
        n: num_return_sequences per prompt
    Returns:
        list of B lists, each with n answer strings.
        i.e. [[ans0_0, ans0_1, ans0_2], [ans1_0, ...], ...]
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to("cuda")

    inp_len = inputs["input_ids"].shape[1]  # uniform due to left-padding

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
    # out shape: [B * n, total_seq_len]
    # out[i*n + j] = j-th sample for prompt i
    B = len(prompts)
    results = []
    for i in range(B):
        samples = []
        for j in range(n):
            gen = tokenizer.decode(out[i * n + j][inp_len:], skip_special_tokens=True).strip()
            samples.append(extract_answer_bilingual(gen))
        results.append(samples)
    return results


# ─── XLSC runner ────────────────────────────────────────────────────────────

def run_xlsc(setup: str, bench: str, project: Path, n: int, temp: float,
             limit: int, batch_size: int = 8):
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    bench_path = project / BENCH_FILE[bench]
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{limit}" if limit else ""
    out_path = results_dir / f"xlsc_{setup}_n{n}_t{temp}_{bench}{suffix}.json"

    # ── Resume logic ─────────────────────────────────────────────────────
    correct = total = 0
    n_tied = 0
    details = []
    done_ids = set()

    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("total", 0) > 0 and not existing.get("partial", False):
                print(f"[SKIP] {out_path.name} done: {existing['accuracy']}%")
                return existing
            elif existing.get("partial") and existing.get("total", 0) > 0:
                details  = existing["details"]
                correct  = existing["correct"]
                total    = existing["total"]
                n_tied   = existing.get("n_tied", 0)
                done_ids = {d["id"] for d in details}
                print(f"[RESUME] {out_path.name}: {total} done, {len(done_ids)} ids skipped")
        except Exception:
            pass

    print(f"\n{'='*60}\n  XLSC: setup={setup}  bench={bench}  N={n}  temp={temp}  batch={batch_size}\n{'='*60}")

    # ── Load model (raw transformers + PEFT, NOT unsloth) ──────────────
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
    elif adapter_path:
        print(f"  [WARN] Adapter not found: {adapter_path}, using base model")
    model.eval()

    # ── Load benchmark ──────────────────────────────────────────────────
    records = load_bench(bench_path)
    if limit and 0 < limit < len(records):
        import random as _r
        _r.Random(SAMPLE_SEED).shuffle(records)
        records = records[:limit]

    # Filter already-done problems when resuming
    if done_ids:
        records = [r for i, r in enumerate(records)
                   if r.get("id", str(i)) not in done_ids]
        print(f"  Resuming: {len(records)} problems remaining")
    else:
        print(f"  Loaded {len(records)} problems  |  batch_size={batch_size}  n={n}")

    last_saved = total  # track save point independent of batch_size

    for batch_start in tqdm(range(0, len(records), batch_size),
                            desc=f"XLSC {setup}/{bench}"):
        batch_recs = records[batch_start:batch_start + batch_size]

        ko_prompts = [make_prompt_xlsc(rec, "ko", tokenizer) for rec in batch_recs]
        en_prompts = [make_prompt_xlsc(rec, "en", tokenizer) for rec in batch_recs]

        # Generate n samples per problem for each language
        # generate_batch returns [B][n] answer lists
        ko_results = generate_batch(model, tokenizer, ko_prompts, n, temp)
        en_results = generate_batch(model, tokenizer, en_prompts, n, temp)

        for j, rec in enumerate(batch_recs):
            gold = normalize_gold(rec)
            ko_samples = ko_results[j]
            en_samples = en_results[j]
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
                "id":          rec.get("id", f"{batch_start + j}"),
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

        # Save every 50 problems regardless of batch_size
        if total - last_saved >= 50:
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
            last_saved = total

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
    ap.add_argument("--batch_size", type=int, default=16,
                    help="problems per batch (default 16; VRAM = batch * n * seq_len)")
    args = ap.parse_args()

    project = Path(__file__).resolve().parent.parent.parent
    benches = ["hrm8k", "gsm8k"] if args.bench == "all" else [args.bench]
    for b in benches:
        run_xlsc(args.setup, b, project, args.n, args.temp, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
