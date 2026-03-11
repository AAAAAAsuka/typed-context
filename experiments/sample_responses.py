#!/usr/bin/env python3
"""Quick diagnostic: sample a few responses from each ICL condition to inspect quality.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/sample_responses.py --config configs/qwen3_8b.yaml --angle 0.5 --n 5
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.icl_experiment import (
    _apply_chat_template, build_delimiter_icl_system,
    build_rotation_icl_system, build_contrastive_icl_system,
)
from model.hook_attention import TypedRoPEHooks
from utils import load_model_from_config, assign_source_labels
import yaml


def generate(model, tokenizer, input_ids, max_new_tokens=128):
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def strip_think(text):
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--angle", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=5, help="Number of samples")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Load config
    exp_cfg_path = os.path.join(Path(__file__).parent.parent, "configs", "experiment.yaml")
    with open(exp_cfg_path) as f:
        exp_cfg = yaml.safe_load(f)
    target_subspaces = exp_cfg["target_subspaces"]
    angle = args.angle

    # Load samples
    data_dir = os.path.join(Path(__file__).parent.parent, "data")
    pi_path = os.path.join(data_dir, "pi_attacks.jsonl")
    samples = []
    with open(pi_path) as f:
        for line in f:
            samples.append(json.loads(line))

    # Take a few diverse samples
    np.random.seed(42)
    indices = np.random.choice(len(samples), size=min(args.n, len(samples)), replace=False)
    selected = [samples[i] for i in sorted(indices)]

    # Load model
    print(f"Loading model from {args.config}...")
    device_map = f"cuda:{args.gpu}"
    model, tokenizer, _ = load_model_from_config(args.config, device_map=device_map)
    model.eval()
    device = next(model.parameters()).device

    output_lines = []

    for idx, sample in enumerate(selected):
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"Sample {idx+1}: category={sample.get('attack_category','?')}")
        output_lines.append(f"System: {sample['system'][:100]}...")
        output_lines.append(f"User: {sample['user'][:150]}...")
        output_lines.append(f"Secret: {sample.get('secret', 'N/A')}")

        messages = [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["user"]},
        ]

        # Condition A: baseline
        input_ids = _apply_chat_template(tokenizer, messages).to(device)
        resp_a = strip_think(generate(model, tokenizer, input_ids))
        output_lines.append(f"\n  [A] Baseline:")
        output_lines.append(f"  {resp_a[:300]}")

        # Condition C: rotation only
        hooks = TypedRoPEHooks(target_subspaces, angle)
        hooks.install(model)
        source_labels = assign_source_labels(tokenizer, sample["system"], sample["user"])
        type_ids = torch.tensor([source_labels], dtype=torch.long).to(device)
        hooks.set_type_ids(type_ids)
        input_ids_c = _apply_chat_template(tokenizer, messages).to(device)
        resp_c = strip_think(generate(model, tokenizer, input_ids_c))
        hooks.remove()
        output_lines.append(f"\n  [C] Rotation only (angle={angle}):")
        output_lines.append(f"  {resp_c[:300]}")

        # Condition C with smaller angle (0.1)
        hooks2 = TypedRoPEHooks(target_subspaces, 0.1)
        hooks2.install(model)
        hooks2.set_type_ids(type_ids)
        resp_c01 = strip_think(generate(model, tokenizer, input_ids_c))
        hooks2.remove()
        output_lines.append(f"\n  [C] Rotation only (angle=0.1):")
        output_lines.append(f"  {resp_c01[:300]}")

    # Print and save
    full_output = "\n".join(output_lines)
    print(full_output)

    out_path = os.path.join(Path(__file__).parent.parent, "outputs", "sample_responses.txt")
    with open(out_path, "w") as f:
        f.write(full_output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
