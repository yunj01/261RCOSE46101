"""
Model Soup: average LoRA adapter weights across multiple setups.

Assumes all adapters share the same base model, LoRA rank, and target modules
(which is true for our B/C/D/E/F setups: Qwen2.5-3B, r=32, same targets).

Usage:
  python -m src.train.make_soup --adapters b c --out soup_bc
  python -m src.train.make_soup --adapters b c f --out soup_bcf
  python -m src.train.make_soup --adapters b c d e f --out soup_bcdef
  python -m src.train.make_soup --adapters b c --weights 0.5 0.5 --out soup_bc
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="adapter names (e.g. 'b c f'). Each must exist under weights/setup_<name>.")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="optional mixing weights (must sum to 1). Default: uniform.")
    ap.add_argument("--out", required=True,
                    help="output adapter dir name (will be at weights/<out>).")
    args = ap.parse_args()

    project = Path(__file__).resolve().parent.parent.parent
    weights_dir = project / "weights"

    adapter_dirs = [weights_dir / f"setup_{name}" for name in args.adapters]
    for d in adapter_dirs:
        if not (d / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Missing adapter_model.safetensors in {d}")

    n = len(adapter_dirs)
    if args.weights:
        if len(args.weights) != n:
            raise ValueError(f"--weights has {len(args.weights)} entries but {n} adapters given.")
        mix = args.weights
        s = sum(mix)
        if abs(s - 1.0) > 1e-3:
            print(f"[soup] Warning: weights sum to {s:.4f}, normalizing.")
            mix = [w / s for w in mix]
    else:
        mix = [1.0 / n] * n

    print(f"[soup] Adapters : {[d.name for d in adapter_dirs]}")
    print(f"[soup] Weights  : {[f'{w:.3f}' for w in mix]}")

    # ── Load and average ───────────────────────────────────────────────────
    state_dicts = [load_file(str(d / "adapter_model.safetensors")) for d in adapter_dirs]

    # Sanity: all must have the same keys
    keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != keys:
            extra = set(sd.keys()) - keys
            missing = keys - set(sd.keys())
            raise ValueError(
                f"Adapter {adapter_dirs[i].name} has different keys.\n"
                f"  extra={list(extra)[:5]}\n  missing={list(missing)[:5]}"
            )
    print(f"[soup] Tensor keys matched : {len(keys)}")

    avg = {}
    for k in keys:
        ref = state_dicts[0][k]
        if not torch.is_floating_point(ref):
            avg[k] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for w, sd in zip(mix, state_dicts):
            acc += w * sd[k].to(torch.float32)
        avg[k] = acc.to(ref.dtype)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = weights_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy non-weight files (config, tokenizer, ...) from the first adapter
    src = adapter_dirs[0]
    for fname in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json", "vocab.json",
                  "merges.txt", "chat_template.jinja"]:
        s = src / fname
        if s.exists():
            shutil.copy2(s, out_dir / fname)

    save_file(avg, str(out_dir / "adapter_model.safetensors"))

    # write a soup manifest for traceability
    manifest = {
        "soup_components": args.adapters,
        "mixing_weights": mix,
        "num_tensors": len(avg),
    }
    with open(out_dir / "soup_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[soup] Saved → {out_dir}")
    print(f"[soup] Manifest → {out_dir / 'soup_manifest.json'}")


if __name__ == "__main__":
    main()
