"""
Phase 2: Teacher CoT generation via Gemini API.

Generates step-by-step reasoning for English & Korean GSM8K.
- Parallel API calls (5 workers)
- Validates answer correctness
- Resumes from existing partial output
- Falls back to slower mode if rate-limited

Usage:
  python -m src.data.generate_cot --lang en
  python -m src.data.generate_cot --lang ko
  python -m src.data.generate_cot --lang both  # default
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


GEMINI_MODEL = "gemini-3-flash-preview"  # 통제 변인: 단일 teacher, frontier quality
MAX_WORKERS = 8                           # preview 모델이라 conservative
MAX_RETRIES = 3
SLEEP_ON_RATE_LIMIT = 30


PROMPT_KO = """당신은 친절한 수학 선생님입니다. 다음 문제를 단계별로 풀어주세요.

**규칙:**
1. 각 단계를 새 줄에 명확히 적기
2. 마지막 줄은 반드시 "따라서 답은 X입니다." (X는 숫자만)
3. 한국어로만 답하기

문제:
{question}

풀이:"""

PROMPT_EN = """You are a helpful math tutor. Solve this problem step by step.

**Rules:**
1. Show each step on a new line
2. End with "Therefore, the answer is X." (X is a number)
3. Use English only

Problem:
{question}

Solution:"""


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set.")
    from google import genai
    return genai.Client(api_key=api_key)


def extract_answer(text: str, lang: str) -> str:
    """모델 출력에서 최종 답 추출"""
    if lang == "ko":
        m = re.search(r"답은\s*(?:\$)?(-?[\d,.]+)", text)
    else:
        m = re.search(r"answer is\s*(?:\$)?(-?[\d,.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    # fallback: 마지막 숫자
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def gen_one(client, problem: dict, lang: str) -> dict:
    """단일 문제 CoT 생성 + validation"""
    prompt_template = PROMPT_KO if lang == "ko" else PROMPT_EN
    prompt = prompt_template.format(question=problem["question"])

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            cot = (resp.text or "").strip()
            predicted = extract_answer(cot, lang)
            gold = str(problem["answer"]).strip().replace(",", "").rstrip(".")
            try:
                ok = abs(float(predicted) - float(gold)) < 0.01
            except (ValueError, TypeError):
                ok = predicted == gold

            return {
                "id": problem["id"],
                "question": problem["question"],
                "cot": cot,
                "predicted": predicted,
                "gold": gold,
                "valid": ok,
                "lang": lang,
                "teacher": GEMINI_MODEL,
            }
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(SLEEP_ON_RATE_LIMIT)
            elif attempt == MAX_RETRIES - 1:
                return {"id": problem["id"], "valid": False, "error": str(e), "lang": lang}
            else:
                time.sleep(2)

    return {"id": problem["id"], "valid": False, "error": "max retries", "lang": lang}


def load_existing(out_path: Path):
    """이미 생성된 결과 (resume support)"""
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("valid"):
                        done_ids.add(rec["id"])
                except json.JSONDecodeError:
                    pass
    return done_ids


def generate_for_lang(lang: str, project: Path):
    raw_dir = project / "data" / "raw"
    out_dir = project / "data" / "teacher_cot"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = raw_dir / f"gsm8k_train_{lang}.jsonl"
    out_path = out_dir / f"cot_{lang}.jsonl"

    problems = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    done_ids = load_existing(out_path)
    todo = [p for p in problems if p["id"] not in done_ids]

    print(f"\n=== Lang: {lang.upper()} ===")
    print(f"  Total: {len(problems):,}")
    print(f"  Done: {len(done_ids):,}")
    print(f"  TODO: {len(todo):,}")

    if not todo:
        print(f"  All done. Skipping.")
        return

    client = get_client()

    with open(out_path, "a", encoding="utf-8") as f_out, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(gen_one, client, p, lang): p for p in todo}
        valid_count = 0
        invalid_count = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{lang}"):
            result = fut.result()
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
            if result.get("valid"):
                valid_count += 1
            else:
                invalid_count += 1

    print(f"  [DONE] {lang}: {valid_count} valid, {invalid_count} invalid -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "ko", "both"], default="both")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent.parent
    langs = ["en", "ko"] if args.lang == "both" else [args.lang]

    for lang in langs:
        generate_for_lang(lang, project)

    print("\n" + "=" * 50)
    print("[DONE] Phase 2 complete - teacher CoT generated")
    print("=" * 50)


if __name__ == "__main__":
    main()
