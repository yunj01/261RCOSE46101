"""
Phase 1.4-1.8: 모든 source 데이터셋 다운로드 + 5000 parallel pair 추출.

Datasets:
  - openai/gsm8k (English, 7473 train + 1319 test)
  - kuotient/gsm8k-ko (Korean translation, parallel)
  - juletxara/mgsm (Multilingual GSM, ko subset)
  - HAERAE-HUB/KMMLU
  - skt/kobest_v1

Usage:
  python -m src.data.download
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


SEED = 42
NUM_PARALLEL_TRAIN = None  # None = use all matching pairs (7473 max)


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [OK] {len(records):,} -> {path}")


def normalize_gsm8k_answer(answer_text: str) -> str:
    """GSM8K answer 형식: 'reasoning ... #### 42' → '42' 만 추출"""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def download_gsm8k(out_dir: Path):
    """영어 GSM8K"""
    print("\n[1/5] Downloading openai/gsm8k...")
    ds = load_dataset("openai/gsm8k", "main")

    train = []
    for i, item in enumerate(ds["train"]):
        train.append({
            "id": f"gsm8k_en_{i:05d}",
            "question": item["question"],
            "raw_answer": item["answer"],  # 추론 포함
            "answer": normalize_gsm8k_answer(item["answer"]),
            "lang": "en",
            "source": "gsm8k",
        })
    save_jsonl(train, out_dir / "gsm8k_train_en_full.jsonl")

    test = []
    for i, item in enumerate(ds["test"]):
        test.append({
            "id": f"gsm8k_en_test_{i:05d}",
            "question": item["question"],
            "raw_answer": item["answer"],
            "answer": normalize_gsm8k_answer(item["answer"]),
            "lang": "en",
            "source": "gsm8k",
        })
    save_jsonl(test, out_dir.parent / "eval" / "gsm8k_test_en.jsonl")


def download_gsm8k_ko(out_dir: Path):
    """한국어 GSM8K (kuotient 번역본)"""
    print("\n[2/5] Downloading kuotient/gsm8k-ko...")
    try:
        ds = load_dataset("kuotient/gsm8k-ko")
    except Exception as e:
        print(f"  ! kuotient 실패: {e}, ChuGyouk/GSM8k-Ko 시도...")
        ds = load_dataset("ChuGyouk/GSM8k-Ko")

    train = []
    for i, item in enumerate(ds["train"]):
        train.append({
            "id": f"gsm8k_ko_{i:05d}",
            "question": item["question"],
            "raw_answer": item["answer"],
            "answer": normalize_gsm8k_answer(item["answer"]),
            "lang": "ko",
            "source": "gsm8k-ko",
        })
    save_jsonl(train, out_dir / "gsm8k_train_ko_full.jsonl")

    if "test" in ds:
        test = []
        for i, item in enumerate(ds["test"]):
            test.append({
                "id": f"gsm8k_ko_test_{i:05d}",
                "question": item["question"],
                "raw_answer": item["answer"],
                "answer": normalize_gsm8k_answer(item["answer"]),
                "lang": "ko",
                "source": "gsm8k-ko",
            })
        save_jsonl(test, out_dir.parent / "eval" / "gsm8k_test_ko.jsonl")


def download_mgsm_ko(out_dir: Path):
    """
    MGSM은 한국어 없음 (bn, de, en, es, fr, ja, ru, sw, te, th, zh만).
    대안: HAERAE-HUB/HRM8K 또는 GSM8K-Ko test set 활용.
    """
    print("\n[3/5] Korean math benchmark...")
    # 시도 1: HRM8K
    try:
        ds = load_dataset("HAERAE-HUB/HRM8K", "GSM8K")
        split_name = "test" if "test" in ds else list(ds.keys())[0]
        test = []
        for i, item in enumerate(ds[split_name]):
            test.append({
                "id": f"hrm8k_{i:04d}",
                "question": item.get("question", item.get("problem", "")),
                "answer": str(item.get("answer", "")).strip(),
                "lang": "ko",
                "source": "hrm8k",
            })
        save_jsonl(test, out_dir / "hrm8k_ko.jsonl")
        print(f"  [OK] HRM8K loaded as Korean math benchmark")
        return
    except Exception as e:
        print(f"  ! HRM8K 실패: {e}")

    # Fallback: GSM8K-Ko test set 이미 있음
    print("  [INFO] MGSM-ko 없음. GSM8K-Ko test set이 이미 eval/에 있음")


def download_kmmlu(out_dir: Path):
    """KMMLU 일부 카테고리 (수학, 과학)"""
    print("\n[4/5] Downloading HAERAE-HUB/KMMLU...")

    target_subjects = ["Math", "Computer-Science", "Biology", "Chemistry", "Economics"]
    all_test = []

    for subj in target_subjects:
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", subj)
            split_name = "test" if "test" in ds else list(ds.keys())[0]
            samples = list(ds[split_name])
            # 각 subject 100개씩
            random.seed(SEED)
            if len(samples) > 100:
                samples = random.sample(samples, 100)

            for i, item in enumerate(samples):
                try:
                    # KMMLU는 4지선다 (A,B,C,D)
                    choices = [item.get("A", ""), item.get("B", ""),
                               item.get("C", ""), item.get("D", "")]
                    gold = item.get("answer", "A")
                    # 다양한 형식 처리: int (0-3 or 1-4), str ("A".."D" or "1".."4")
                    if isinstance(gold, int):
                        idx = gold if gold < 4 else gold - 1
                        gold_letter = "ABCD"[max(0, min(3, idx))]
                    elif isinstance(gold, str):
                        g = gold.strip().upper()
                        if g in "ABCD":
                            gold_letter = g
                        elif g.isdigit():
                            idx = int(g) if int(g) < 4 else int(g) - 1
                            gold_letter = "ABCD"[max(0, min(3, idx))]
                        else:
                            gold_letter = "A"
                    else:
                        gold_letter = "A"

                    all_test.append({
                        "id": f"kmmlu_{subj}_{i:04d}",
                        "question": item.get("question", ""),
                        "choices": choices,
                        "answer": gold_letter,
                        "subject": subj,
                        "lang": "ko",
                        "source": "kmmlu",
                    })
                except Exception:
                    continue
            print(f"  [OK] KMMLU {subj}: {len(samples)} loaded")
        except Exception as e:
            print(f"  ! KMMLU {subj} 실패: {e}")

    save_jsonl(all_test, out_dir / "kmmlu_subset.jsonl")


def download_kobest(out_dir: Path):
    """KoBest 일부 task (BoolQ, COPA, HellaSwag)"""
    print("\n[5/5] Downloading skt/kobest_v1...")

    target_tasks = ["boolq", "copa", "hellaswag"]
    all_test = []

    for task in target_tasks:
        try:
            ds = load_dataset("skt/kobest_v1", task)
            split_name = "test" if "test" in ds else "validation"
            samples = list(ds[split_name])
            random.seed(SEED)
            if len(samples) > 200:
                samples = random.sample(samples, 200)

            for i, item in enumerate(samples):
                if task == "boolq":
                    rec = {
                        "question": item.get("question", ""),
                        "passage": item.get("paragraph", ""),
                        "answer": "예" if item.get("label", 0) == 1 else "아니오",
                    }
                elif task == "copa":
                    rec = {
                        "premise": item.get("premise", ""),
                        "choice_1": item.get("alternative_1", ""),
                        "choice_2": item.get("alternative_2", ""),
                        "question_type": item.get("question", ""),
                        "answer": str(item.get("label", 0) + 1),
                    }
                elif task == "hellaswag":
                    rec = {
                        "context": item.get("context", ""),
                        "endings": [item.get(f"ending_{j+1}", "") for j in range(4)],
                        "answer": str(item.get("label", 0)),
                    }
                rec.update({
                    "id": f"kobest_{task}_{i:04d}",
                    "task": task,
                    "lang": "ko",
                    "source": "kobest",
                })
                all_test.append(rec)
            print(f"  [OK] KoBest {task}: {len(samples)} loaded")
        except Exception as e:
            print(f"  ! KoBest {task} 실패: {e}")

    save_jsonl(all_test, out_dir / "kobest_subset.jsonl")


def make_parallel_pairs(raw_dir: Path):
    """영어/한국어 GSM8K parallel 5000 추출"""
    print(f"\n[Parallel] Aligning EN-KO pairs (target: {NUM_PARALLEL_TRAIN})...")

    en = []
    with open(raw_dir / "gsm8k_train_en_full.jsonl", encoding="utf-8") as f:
        for line in f:
            en.append(json.loads(line))

    ko = []
    with open(raw_dir / "gsm8k_train_ko_full.jsonl", encoding="utf-8") as f:
        for line in f:
            ko.append(json.loads(line))

    print(f"  EN total: {len(en):,}, KO total: {len(ko):,}")

    # 동일 인덱스로 paired (kuotient/gsm8k-ko가 openai/gsm8k의 번역본이므로)
    max_n = min(len(en), len(ko))
    n_target = NUM_PARALLEL_TRAIN if NUM_PARALLEL_TRAIN else max_n

    random.seed(SEED)
    indices = list(range(max_n))
    random.shuffle(indices)

    en_paired = []
    ko_paired = []
    for idx in indices:
        if en[idx]["answer"] == ko[idx]["answer"]:  # 답 일치하는 경우만
            en_paired.append(en[idx])
            ko_paired.append(ko[idx])
        if len(en_paired) >= n_target:
            break

    print(f"  Aligned pairs (answer match): {len(en_paired):,}")

    save_jsonl(en_paired, raw_dir / "gsm8k_train_en.jsonl")
    save_jsonl(ko_paired, raw_dir / "gsm8k_train_ko.jsonl")


def main():
    project = Path(__file__).resolve().parent.parent.parent
    raw_dir = project / "data" / "raw"
    eval_dir = project / "data" / "eval"
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    download_gsm8k(raw_dir)
    download_gsm8k_ko(raw_dir)
    download_mgsm_ko(eval_dir)
    download_kmmlu(eval_dir)
    download_kobest(eval_dir)
    make_parallel_pairs(raw_dir)

    print("\n" + "=" * 50)
    print("[DONE] Phase 1 complete - all data ready")
    print("=" * 50)


if __name__ == "__main__":
    main()
