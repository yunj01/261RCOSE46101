"""
Greedy evaluation for the 5 setups in the DALR pipeline.

Setups:
  a         = Base (no FT, Llama-3.2-3B-Instruct as-is)
  b         = English CoT FT
  c         = Korean CoT FT
  d         = Bilingual Mix FT (50/50)
  e         = DALR (Difficulty-Aware Language Routing)
  e_random  = DALR ablation (random EN bridges on easy problems)

Benchmarks:
  hrm8k = data/eval/hrm8k_ko.jsonl      (1,319 Korean math)
  gsm8k = data/eval/gsm8k_test_en.jsonl (1,319 English math)

Usage:
  python -m src.eval.evaluate --setup e --bench hrm8k
  python -m src.eval.evaluate --setup all --bench all   # full matrix
"""

import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL_NAME   = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
MAX_SEQ_LEN  = 1024
MAX_NEW_TOKENS = 512      # 팀원 코드와 동일
SAMPLE_SEED  = 42         # reproducible subsampling
EVAL_BATCH_SIZE = 8       # batch inference for speed (RTX 5070 Ti 16GB)

SYSTEM_KO = "당신은 수학 문제를 단계별로 풀어주는 친절한 선생님입니다."
SYSTEM_EN = "You are a helpful math tutor. Solve problems step by step."

# Override via env var for A/B testing the zero-shot baseline (Setup A).
# Example: $env:SYSTEM_KO_OVERRIDE = ""    # empty system prompt
#          $env:SYSTEM_KO_OVERRIDE = "EN"  # use SYSTEM_EN for hrm8k too
_OVERRIDE = os.environ.get("SYSTEM_KO_OVERRIDE", None)
if _OVERRIDE is not None:
    SYSTEM_KO = SYSTEM_EN if _OVERRIDE == "EN" else _OVERRIDE
    print(f"[SYSTEM_KO override] -> {SYSTEM_KO!r}")


SETUP_ADAPTER = {
    "a":        None,                       # base, no adapter
    "b":        "weights/setup_b",          # English-only CoT SFT
    "c":        "weights/setup_c",          # Korean-only CoT SFT
    "d":        "weights/setup_d",          # bilingual 50/50 mix SFT
    "e":        "weights/setup_e",          # DALR: difficulty-aware language routing
    "e_random": "weights/setup_e_random",   # DALR ablation: random EN bridges
}

BENCH_FILE = {
    "hrm8k": "data/eval/hrm8k_ko.jsonl",
    "gsm8k": "data/eval/gsm8k_test_en.jsonl",
}

def _patch_exaone_compat(model):
    """EXAONE-3.5 compatibility shim for unsloth for_inference.
    Traverses all submodules (works through PEFT/LoRA wrappers) to find
    ExaoneModel and patches get_input_embeddings → self.wte."""
    for _name, module in model.named_modules():
        cls = type(module)
        if "ExaoneModel" in cls.__name__ and not getattr(cls, "_exaone_compat_patched", False):
            def _get_input_embeddings(self):
                return getattr(self, "wte", None) or getattr(self, "embed_tokens", None)
            def _set_input_embeddings(self, value):
                attr = "wte" if hasattr(self, "wte") else "embed_tokens"
                setattr(self, attr, value)
            cls.get_input_embeddings = _get_input_embeddings
            cls.set_input_embeddings = _set_input_embeddings
            cls._exaone_compat_patched = True
            print(f"[PATCH] EXAONE compat applied to {cls.__name__}")
            break


def load_bench(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def make_prompt(record, bench_name, tokenizer):
    """Build prompt: system(CoT guide) + user(question), apply chat template"""
    q = record.get("question", record.get("problem", ""))
    system = SYSTEM_KO if bench_name == "hrm8k" else SYSTEM_EN
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": q},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# Numeric capture allows thousands separators (e.g., "70,000") and decimals.
_NUM = r"(-?[\d,]+(?:\.\d+)?)"
# Conservative marker patterns: only those rarely matching intermediate calculations.
_RE_BOXED  = re.compile(rf"\\boxed\{{\s*{_NUM}\s*\}}")
_RE_HASH4  = re.compile(rf"####\s*{_NUM}")
_RE_ANS_EN = re.compile(rf"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$?\s*{_NUM}", re.IGNORECASE)
_RE_ANS_KO = re.compile(rf"(?:답은|คำตอบ은)\s*\$?\s*{_NUM}")
# Final-position number with comma-aware US thousands format.
_RE_NUM_FALLBACK = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _clean(s: str) -> str:
    return s.replace(",", "").rstrip(".")


def extract_answer(text: str, bench: str) -> str:
    """Robust numeric-answer extractor.

    Strategy (apply in order, take LAST match per strategy = the model's final answer):
      1. LaTeX \\boxed{X}            (highest precision)
      2. GSM8K-canonical #### X
      3. Bench-priority "answer is X" (EN for gsm8k, KO for hrm8k)
      4. Cross-language "answer is X" (handles cross-lingual slip in base models)
      5. Last standalone number with thousands-separator support
    """
    for rx in (_RE_BOXED, _RE_HASH4):
        m = rx.findall(text)
        if m:
            return _clean(m[-1])

    primary, secondary = (_RE_ANS_KO, _RE_ANS_EN) if bench == "hrm8k" else (_RE_ANS_EN, _RE_ANS_KO)
    for rx in (primary, secondary):
        m = rx.findall(text)
        if m:
            return _clean(m[-1])

    nums = _RE_NUM_FALLBACK.findall(text)
    return _clean(nums[-1]) if nums else ""


def normalize_gold(record) -> str:
    ans = str(record.get("answer", "")).strip()
    return ans.replace(",", "").rstrip(".")


def run_eval(setup: str, bench: str, project: Path, limit: int = 0, batch_size: int = 16):
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    bench_path = project / BENCH_FILE[bench]
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{limit}" if limit else ""
    out_path = results_dir / f"setup_{setup}_{bench}{suffix}_exaone.json"

    # ---- Resume logic ----
    correct = 0
    total = 0
    details = []
    done_ids = set()

    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("total", 0) > 0 and not existing.get("partial", False):
                print(f"\n[SKIP] {setup}/{bench} already done: {existing['accuracy']}% ({existing['correct']}/{existing['total']})")
                print(f"  -> {out_path}")
                return existing
            elif existing.get("partial") and existing.get("total", 0) > 0:
                details  = existing["details"]
                correct  = existing["correct"]
                total    = existing["total"]
                done_ids = {d["id"] for d in details}
                print(f"\n[RESUME] {setup}/{bench}: {total} done, resuming from problem {total}")
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted, redo

    print(f"\n{'='*60}")
    print(f"  Eval: setup={setup}  bench={bench}")
    print(f"  Adapter: {adapter_path or 'BASE (no adapter)'}")
    print(f"  Bench: {bench_path}")
    print(f"{'='*60}\n")

    # ---- Load model (raw transformers + PEFT, NOT unsloth) ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-padding required for batched causal LM generation

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if adapter_path and adapter_path.exists():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    else:
        if adapter_path:
            print(f"  [WARN] Adapter not found: {adapter_path}, using base model")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

    _patch_exaone_compat(model)
    FastLanguageModel.for_inference(model)

    # ---- Load benchmark ----
    records = load_bench(bench_path)
    if limit and limit > 0 and limit < len(records):
        import random as _r
        _r.Random(SAMPLE_SEED).shuffle(records)
        records = records[:limit]

    # Filter already-done problems when resuming
    if done_ids:
        records = [r for i, r in enumerate(records)
                   if r.get("id", str(i)) not in done_ids]
        print(f"  Resuming: {len(records)} problems remaining  |  batch_size={batch_size}")
    else:
        print(f"  Loaded {len(records)} problems  |  batch_size={batch_size}")

    # ---- Batched inference (EVAL_BATCH_SIZE로 속도 향상) ----
    import torch
    correct = 0
    total = 0
    details = []

    # Left-padding for decoder-only batch inference
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts_all = [make_prompt(rec, bench, tokenizer) for rec in records]
    golds_all   = [normalize_gold(rec) for rec in records]

    for batch_start in tqdm(range(0, len(records), EVAL_BATCH_SIZE),
                            desc=f"{setup}/{bench}",
                            total=(len(records) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE):
        batch_end   = min(batch_start + EVAL_BATCH_SIZE, len(records))
        batch_recs  = records[batch_start:batch_end]
        batch_prompts = prompts_all[batch_start:batch_end]
        batch_golds   = golds_all[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt",
            padding=True, truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to("cuda")

        inp_lens = inputs["attention_mask"].sum(dim=1)  # actual token length per sample

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy (deterministic)
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, (rec, gold) in enumerate(zip(batch_recs, batch_golds)):
            inp_len  = inp_lens[j].item()
            gen_ids  = outputs[j][inp_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            pred = extract_answer(gen_text, bench)

            try:
                ok = abs(float(pred) - float(gold)) < 0.01
            except (ValueError, TypeError):
                ok = pred.strip() == gold.strip()

            correct += int(ok)
            total += 1
            details.append({
                "id":        rec.get("id", f"{batch_start+j}"),
                "question":  rec.get("question", rec.get("problem", "")),
                "gold":      gold,
                "predicted": pred,
                "output":    gen_text,
                "correct":   ok,
            })

        # Incremental save every 50 samples (재부팅 대비)
        if total % 50 == 0 or total == len(records):
            partial = {
                "setup": setup, "bench": bench,
                "total": total, "correct": correct,
                "accuracy": round(correct/total*100, 2),
                "partial": True, "details": details,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, indent=2)
            last_saved = total

    accuracy = correct / total if total else 0.0
    summary = {
        "setup":    setup,
        "bench":    bench,
        "total":    total,
        "correct":  correct,
        "accuracy": round(accuracy * 100, 2),
        "details":  details,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  [RESULT] setup={setup} | bench={bench} | Acc={accuracy*100:.2f}% ({correct}/{total})")
    print(f"  Saved -> {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", default="all",
                        choices=["all"] + list(SETUP_ADAPTER.keys()))
    parser.add_argument("--bench", default="all",
                        choices=["all"] + list(BENCH_FILE.keys()))
    parser.add_argument("--wandb_project", default="korean-cot-distill")
    parser.add_argument("--limit", type=int, default=0,
                        help="Subsample N problems per bench (0=all, default)")
    parser.add_argument("--tag", default="",
                        help="Suffix for results filename (e.g. 'quick')")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent.parent

    setups = list(SETUP_ADAPTER.keys()) if args.setup == "all" else [args.setup]
    benches = list(BENCH_FILE.keys())  if args.bench == "all"  else [args.bench]

    all_results = {}
    for setup in setups:
        for bench in benches:
            result = run_eval(setup, bench, project, limit=args.limit, batch_size=args.batch_size)
            all_results[f"{setup}_{bench}"] = {
                "accuracy": result["accuracy"],
                "correct":  result["correct"],
                "total":    result["total"],
            }

    # Summary matrix
    tag = f"_{args.tag}" if args.tag else ""
    summary_path = project / "results" / f"summary_matrix{tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)
    header = f"{'Setup':<8}" + "".join(f"{b:>12}" for b in benches)
    print(header)
    print("-" * len(header))
    for setup in setups:
        row = f"{setup:<8}"
        for bench in benches:
            key = f"{setup}_{bench}"
            acc = all_results.get(key, {}).get("accuracy", "N/A")
            row += f"{str(acc)+' %':>12}"
        print(row)
    print("="*60)
    print(f"[DONE] Results -> {project / 'results'}/")


if __name__ == "__main__":
    main()
