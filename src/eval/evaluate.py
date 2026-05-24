"""
Greedy evaluation for the 5 setups in the DALR pipeline.

Setups:
  a         = Base (no FT, Qwen2.5-3B-Instruct as-is)
  b         = English CoT FT
  c         = Korean CoT FT
  d         = Bilingual Mix FT (50/50)
  e         = DALR (Difficulty-Aware Language Routing)
  e_random  = DALR ablation (random EN bridges on easy problems)

Benchmarks:
  hrm8k = data/eval/hrm8k_ko.jsonl      (1,319 Korean math) ⭐ main
  gsm8k = data/eval/gsm8k_test_en.jsonl (1,319 English math)

Usage:
  python -m src.eval.evaluate --setup e --bench hrm8k
  python -m src.eval.evaluate --setup all --bench all   # full matrix
"""

from unsloth import FastLanguageModel

import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm


MODEL_NAME   = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
MAX_SEQ_LEN  = 1024
MAX_NEW_TOKENS = 512      # 팀원 코드와 동일
SAMPLE_SEED  = 42         # reproducible subsampling

# 팀원 코드 동일: system prompt로 CoT 유도 (zero-shot CoT prompting)
SYSTEM_KO = "당신은 수학 문제를 단계별로 풀어주는 친절한 선생님입니다."
SYSTEM_EN = "You are a helpful math tutor. Solve problems step by step."

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

def load_bench(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def make_prompt(record, bench_name, tokenizer):
    """팀원 코드 매칭: system(CoT 유도) + user(질문), chat template 적용"""
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


def extract_answer(text: str, bench: str) -> str:
    if bench == "gsm8k":
        m = re.search(r"answer is\s*(?:\$)?(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    else:
        m = re.search(r"답은\s*(?:\$)?(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def normalize_gold(record) -> str:
    ans = str(record.get("answer", "")).strip()
    # strip trailing dot and commas
    return ans.replace(",", "").rstrip(".")


def run_eval(setup: str, bench: str, project: Path, limit: int = 0):
    adapter_rel = SETUP_ADAPTER[setup]
    adapter_path = (project / adapter_rel) if adapter_rel else None
    bench_path = project / BENCH_FILE[bench]
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{limit}" if limit else ""
    out_path = results_dir / f"setup_{setup}_{bench}{suffix}_exaone.json"

    # ---- Resume: 완료된(partial=False) 결과만 skip ----
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("total", 0) > 0 and not existing.get("partial", False):
                print(f"\n[SKIP] {setup}/{bench} already done: {existing['accuracy']}% ({existing['correct']}/{existing['total']})")
                print(f"  -> {out_path}")
                return existing
            elif existing.get("partial"):
                print(f"\n[INFO] {setup}/{bench} has partial result ({existing['total']} done) - redoing from start")
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted, redo

    print(f"\n{'='*60}")
    print(f"  Eval: setup={setup}  bench={bench}")
    print(f"  Adapter: {adapter_path or 'BASE (no adapter)'}")
    print(f"  Bench: {bench_path}")
    print(f"{'='*60}\n")

    # ---- Load model ----
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

    FastLanguageModel.for_inference(model)

    # ---- Load benchmark ----
    records = load_bench(bench_path)
    if limit and limit > 0 and limit < len(records):
        import random as _r
        _r.Random(SAMPLE_SEED).shuffle(records)
        records = records[:limit]
        print(f"  Loaded {len(records)} problems (subsampled from full, seed={SAMPLE_SEED})")
    else:
        print(f"  Loaded {len(records)} problems")

    # ---- Single-sample inference (팀원 코드 매칭, batch padding 영향 제거) ----
    import torch
    correct = 0
    total = 0
    details = []

    for i, rec in enumerate(tqdm(records, desc=f"{setup}/{bench}")):
        prompt = make_prompt(rec, bench, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                # greedy (deterministic)
                temperature=1.0,
                repetition_penalty=1.1,         # 팀원 코드 매칭 (loop 방지)
                pad_token_id=tokenizer.eos_token_id,
            )

        inp_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][inp_len:]
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
                "id":        rec.get("id", f"{i}"),
                "question":  rec.get("question", rec.get("problem", "")),
                "gold":      gold,
                "predicted": pred,
                "output":    gen_text,
                "correct":   ok,
            })

        # Incremental save every 50 samples (재부팅 대비)
        if (i + 1) % 50 == 0:
            partial = {
                "setup": setup, "bench": bench,
                "total": total, "correct": correct,
                "accuracy": round(correct/total*100, 2),
                "partial": True, "details": details,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, indent=2)

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
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent.parent

    setups = list(SETUP_ADAPTER.keys()) if args.setup == "all" else [args.setup]
    benches = list(BENCH_FILE.keys())  if args.bench == "all"  else [args.bench]

    all_results = {}
    for setup in setups:
        for bench in benches:
            result = run_eval(setup, bench, project, limit=args.limit)
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
