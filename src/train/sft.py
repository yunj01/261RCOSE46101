"""
LoRA SFT for the 5 setups (DALR pipeline).

Setups:
  b         = English CoT FT                       (lr=2e-4, 3ep)
  c         = Korean CoT FT                        (lr=2e-4, 3ep)
  d         = Bilingual Mix FT (50/50)             (lr=2e-4, 3ep)
  e         = DALR: KO + EN bridge on hard cases   (lr=2e-4, 3ep)
  e_random  = DALR ablation: EN on random easy     (lr=2e-4, 3ep)

Usage:
  python -m src.train.sft --setup b
  python -m src.train.sft --setup e
"""

# Unsloth must be imported BEFORE transformers/peft/trl
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

import os
import json
import argparse
from pathlib import Path

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer


# ---- Constants ----
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
# EXAONE-3.5 attention: out_proj (≠ o_proj); MLP: c_fc_0/c_fc_1/c_proj (≠ gate/up/down)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj",
                "c_fc_0", "c_fc_1", "c_proj"]


def _patch_exaone_compat(model):
    """EXAONE-3.5 compatibility shim for PEFT.

    PEFT's tied-weight check calls model.get_input_embeddings(), which
    traverses down to ExaoneModel.  ExaoneModel does not override
    get_input_embeddings() in its custom code, so we add it here
    pointing at self.wte (the actual token-embedding table).
    """
    inner = getattr(model, "transformer", None) or getattr(model, "model", None)
    if inner is None:
        return
    cls = type(inner)
    if not getattr(cls, "_exaone_compat_patched", False):
        def _get_input_embeddings(self):
            return getattr(self, "wte", None) or getattr(self, "embed_tokens", None)
        def _set_input_embeddings(self, value):
            attr = "wte" if hasattr(self, "wte") else "embed_tokens"
            setattr(self, attr, value)
        cls.get_input_embeddings = _get_input_embeddings
        cls.set_input_embeddings = _set_input_embeddings
        cls._exaone_compat_patched = True

SETUP_CONFIG = {
    "b":        {"file": "setup_b_english_cot.jsonl",  "epochs": 3, "lr": 2e-4, "out": "setup_b"},
    "c":        {"file": "setup_c_korean_cot.jsonl",   "epochs": 3, "lr": 2e-4, "out": "setup_c"},
    "d":        {"file": "setup_d_bilingual_mix.jsonl","epochs": 3, "lr": 2e-4, "out": "setup_d"},
    "e":        {"file": "setup_e_dalr.jsonl",         "epochs": 3, "lr": 2e-4, "out": "setup_e"},
    "e_random": {"file": "setup_e_random.jsonl",       "epochs": 3, "lr": 2e-4, "out": "setup_e_random"},
}


def load_records(path: Path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_dataset(records, tokenizer):
    """Convert {question, cot} -> chat-templated text"""
    def map_fn(ex):
        messages = [
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["cot"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = Dataset.from_list(records)
    ds = ds.map(map_fn, remove_columns=ds.column_names, desc="chat templating")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", required=True, choices=list(SETUP_CONFIG.keys()))
    parser.add_argument("--resume_from", default=None,
                        help="Path to a LoRA adapter to continue training from (optional)")
    parser.add_argument("--wandb_project", default="korean-cot-distill")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SETUP_CONFIG[args.setup]
    project = Path(__file__).resolve().parent.parent.parent
    data_path = project / "data" / "train" / cfg["file"]
    out_dir = project / "weights" / cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = f"setup_{args.setup}"

    print(f"\n{'='*60}")
    print(f"  Setup: {args.setup}  ({cfg['out']})")
    print(f"  Data:  {data_path.name}")
    print(f"  LR:    {cfg['lr']}  Epochs: {cfg['epochs']}")
    print(f"  Out:   {out_dir}")
    if args.resume_from:
        print(f"  Resume from: {args.resume_from}")
    print(f"{'='*60}\n")

    # ---- Model ----
    if args.resume_from:
        # Stage 2: load Stage 1 adapter then continue training
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.resume_from,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            attn_implementation="sdpa",  # RTX 5060 Ti (Blackwell SM12): xformers 미지원
            trust_remote_code=True,
        )
        _patch_exaone_compat(model)
        # adapter is loaded but need to ensure trainable
        FastLanguageModel.for_training(model)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            attn_implementation="sdpa",  # RTX 5060 Ti (Blackwell SM12): xformers 미지원
            trust_remote_code=True,
        )
        _patch_exaone_compat(model)
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=LORA_TARGETS,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )

    # ---- Data ----
    records = load_records(data_path)
    print(f"[Data] loaded {len(records):,} records")
    ds = build_dataset(records, tokenizer)

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # ---- Train ----
    trainer.train()

    # ---- Save final adapter ----
    final_dir = out_dir
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[DONE] Saved adapter -> {final_dir}")


if __name__ == "__main__":
    main()
