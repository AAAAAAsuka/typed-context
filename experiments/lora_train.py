#!/usr/bin/env python3
"""LoRA finetuning with typed RoPE — Phase 4.

Trains a LoRA adapter with typed RoPE hooks active during training,
teaching the model to use type signals for PI defense.

Usage:
    python experiments/lora_train.py                    # full training (needs GPU)
    python experiments/lora_train.py --max-samples 100  # quick test
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")


def prepare_training_data(tokenizer, data_path, max_samples=None):
    """Prepare training data with chat template formatting."""
    from utils import assign_source_labels

    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break

    processed = []
    for sample in samples:
        messages = [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["user"]},
            {"role": "assistant", "content": sample["response"]},
        ]

        # Tokenize full conversation with response
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, enable_thinking=False
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False
            )

        # Get source labels (system=0, user=1) for type rotation
        _, source_labels = assign_source_labels(
            tokenizer, sample["system"], sample["user"]
        )

        # Extend source labels for the response tokens (mark as system/trusted)
        full_labels = np.zeros(len(input_ids), dtype=np.int64)
        full_labels[:len(source_labels)] = source_labels
        # Response tokens: label as 0 (trusted/system)
        full_labels[len(source_labels):] = 0

        processed.append({
            "input_ids": input_ids,
            "source_labels": full_labels.tolist(),
            "label": sample["label"],
        })

    return processed


def train_lora(model, tokenizer, train_data, target_subspaces, rotation_angle,
               epochs=3, lr=2e-5, output_dir=None):
    """Train LoRA adapter with typed RoPE hooks active."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from model.hook_attention import install_type_rotation_hooks

    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    os.makedirs(output_dir, exist_ok=True)

    # Configure LoRA (dropout=0 for FP8/quantized models)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Prepare quantized model for training if needed
    try:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass  # Not needed for non-quantized models

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Install typed RoPE hooks
    hooks = install_type_rotation_hooks(model, target_subspaces, rotation_angle)

    # Training setup
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Simple training loop
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_losses = []
        for i, item in enumerate(train_data):
            input_ids = torch.tensor([item["input_ids"]], dtype=torch.long).to(device)
            type_ids = torch.tensor([item["source_labels"]], dtype=torch.long)
            hooks.set_type_ids(type_ids)

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if (i + 1) % 100 == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                print(f"  Epoch {epoch+1}/{epochs}, Step {i+1}/{len(train_data)}: "
                      f"loss={avg_loss:.4f}")

        avg_epoch_loss = np.mean(epoch_losses)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_epoch_loss:.4f}")

    # Remove hooks and save
    hooks.remove()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save loss history
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump({"loss_history": loss_history, "epochs": epochs, "lr": lr}, f)

    print(f"\nAdapter saved to: {output_dir}")
    return loss_history


def main():
    parser = argparse.ArgumentParser(description="LoRA training with typed RoPE")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "lora_adapter"))
    parser.add_argument("--config-dir", default=CONFIG_DIR)
    parser.add_argument("--config", default=None,
                        help="Model config YAML (e.g. configs/qwen3_8b.yaml)")
    parser.add_argument("--precision", default="bf16",
                        help="Override model precision (bf16 recommended for LoRA training)")
    parser.add_argument("--model", default=None,
                        help="Override model name (e.g. Qwen/Qwen3-8B for non-FP8)")
    args = parser.parse_args()

    # Load config
    ts_path = os.path.join(args.config_dir, "target_subspaces.json")
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            target_subspaces = json.load(f)["target_subspaces"]
    else:
        target_subspaces = [59, 60, 61, 62, 63]

    exp_path = os.path.join(args.config_dir, "experiment.yaml")
    if os.path.exists(exp_path):
        with open(exp_path) as f:
            rotation_angle = yaml.safe_load(f).get("max_safe_angle", math.pi / 4)
    else:
        rotation_angle = math.pi / 4

    from utils import load_model, load_model_from_config
    if args.model:
        model, tokenizer = load_model(args.model, precision=args.precision)
    elif args.config:
        model, tokenizer, _ = load_model_from_config(
            args.config, precision=args.precision)
    else:
        model, tokenizer = load_model(precision=args.precision)

    data_path = os.path.join(DATA_DIR, "training_data.jsonl")
    if not os.path.exists(data_path):
        print("Training data not found. Run data/build_training_data.py first.")
        sys.exit(1)

    print("Preparing training data...")
    train_data = prepare_training_data(tokenizer, data_path, args.max_samples)
    print(f"Prepared {len(train_data)} training samples")

    loss_history = train_lora(
        model, tokenizer, train_data, target_subspaces, rotation_angle,
        epochs=args.epochs, lr=args.lr, output_dir=args.output_dir
    )

    # Verification
    print("\n=== Verification ===")
    log_path = os.path.join(args.output_dir, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        losses = log["loss_history"]
        if len(losses) >= 2 and losses[-1] < losses[0]:
            print(f"  PASS: loss decreased ({losses[0]:.4f} → {losses[-1]:.4f})")
        else:
            print(f"  CHECK: loss history = {losses}")
    else:
        print(f"  MISSING: training_log.json")

    adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in adapter_files:
        path = os.path.join(args.output_dir, fname)
        if os.path.exists(path):
            print(f"  OK: {fname}")
        else:
            print(f"  MISSING: {fname}")

    print("\nDone!")


if __name__ == "__main__":
    main()
