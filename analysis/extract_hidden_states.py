#!/usr/bin/env python3
"""Extract hidden states from all layers for probing experiments.

Loads the model in fp16, runs forward passes on each dataset sample with
output_hidden_states=True, and saves per-layer hidden states + source labels
to .npz files in outputs/hidden_states/.

Usage:
    python analysis/extract_hidden_states.py                    # all datasets
    python analysis/extract_hidden_states.py --datasets normal  # single dataset
    python analysis/extract_hidden_states.py --max-samples 10   # limit for testing
    python analysis/extract_hidden_states.py --layers 0 8 16 24 31  # specific layers
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, load_model_from_config, assign_source_labels

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "hidden_states")

DATASET_FILES = {
    "normal": os.path.join(DATA_DIR, "normal.jsonl"),
    "pi_success": os.path.join(DATA_DIR, "pi_success.jsonl"),
    "pi_fail": os.path.join(DATA_DIR, "pi_fail.jsonl"),
    "swapped": os.path.join(DATA_DIR, "swapped.jsonl"),
}


def load_jsonl(path, max_samples=None):
    """Load samples from a JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def extract_hidden_states(model, tokenizer, samples, dataset_name,
                          batch_size=4, target_layers=None):
    """Extract hidden states and source labels from model for all samples.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        samples: list of dicts with 'system' and 'user' fields
        dataset_name: name for logging
        batch_size: number of samples to process before saving intermediate
        target_layers: list of layer indices to save (None = all layers)

    Returns:
        hidden_states: dict[layer_idx] -> np.array (total_tokens, hidden_dim)
        source_labels: np.array (total_tokens,)
    """
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    if target_layers is None:
        target_layers = list(range(num_layers))

    all_hidden = {l: [] for l in target_layers}
    all_labels = []
    device = next(model.parameters()).device

    for i, sample in enumerate(samples):
        system_content = sample["system"]
        user_content = sample["user"]
        is_swapped = sample.get("swap_order", False)

        if is_swapped:
            # For swapped dataset: user content goes first (as "system"),
            # system content goes second (as "user")
            input_ids, source_labels = assign_source_labels(
                tokenizer, user_content, system_content
            )
            # Flip labels: in swapped, position 0 is actually user content,
            # position 1 is actually system content
            source_labels = 1 - source_labels
        else:
            input_ids, source_labels = assign_source_labels(
                tokenizer, system_content, user_content
            )

        inputs = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)

        # outputs.hidden_states: tuple of (num_layers+1) tensors,
        # each shape (1, seq_len, hidden_dim)
        for layer_idx in target_layers:
            hs = outputs.hidden_states[layer_idx]
            all_hidden[layer_idx].append(hs[0].cpu().float().numpy())

        all_labels.append(source_labels)

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  [{dataset_name}] Processed {i+1}/{len(samples)} samples")

        # Free GPU memory
        del outputs, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Concatenate all samples
    for l in target_layers:
        all_hidden[l] = np.concatenate(all_hidden[l], axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_hidden, all_labels


def save_hidden_states(hidden_states, source_labels, dataset_name, output_dir):
    """Save hidden states and labels to .npz file.

    Saves one file per dataset containing:
        - hidden_layer_{i}: array of shape (num_tokens, hidden_dim) for each layer
        - source_labels: array of shape (num_tokens,)
    """
    os.makedirs(output_dir, exist_ok=True)
    save_dict = {"source_labels": source_labels}
    for layer_idx, hs in hidden_states.items():
        save_dict[f"hidden_layer_{layer_idx}"] = hs

    out_path = os.path.join(output_dir, f"{dataset_name}.npz")
    np.savez_compressed(out_path, **save_dict)

    # Report shapes
    print(f"  Saved {out_path}")
    print(f"    source_labels shape: {source_labels.shape}")
    print(f"    Layers saved: {sorted(hidden_states.keys())}")
    if hidden_states:
        first_layer = next(iter(hidden_states.values()))
        print(f"    Per-layer shape: {first_layer.shape}")
    print(f"    Label distribution: 0(sys)={np.sum(source_labels==0)}, "
          f"1(user)={np.sum(source_labels==1)}")

    return out_path


def verify_outputs(output_dir, dataset_names):
    """Verify that all expected .npz files exist with correct shapes."""
    print("\n=== Verification ===")
    all_ok = True
    for name in dataset_names:
        path = os.path.join(output_dir, f"{name}.npz")
        if not os.path.exists(path):
            print(f"  FAIL: {path} not found")
            all_ok = False
            continue

        data = np.load(path)
        keys = list(data.keys())
        layer_keys = [k for k in keys if k.startswith("hidden_layer_")]
        has_labels = "source_labels" in keys

        if not has_labels:
            print(f"  FAIL: {name} missing source_labels")
            all_ok = False
            continue

        labels = data["source_labels"]
        print(f"  {name}.npz: {len(layer_keys)} layers, "
              f"{labels.shape[0]} tokens, "
              f"labels 0={np.sum(labels==0)} / 1={np.sum(labels==1)}")

        # Check shapes consistency
        for lk in layer_keys:
            hs = data[lk]
            if hs.shape[0] != labels.shape[0]:
                print(f"    FAIL: {lk} shape {hs.shape} != labels {labels.shape}")
                all_ok = False

        data.close()

    if all_ok:
        print("  All checks passed!")
    else:
        print("  Some checks FAILED!")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states for probing")
    parser.add_argument("--datasets", nargs="+",
                        default=["normal", "pi_success", "pi_fail", "swapped"],
                        choices=["normal", "pi_success", "pi_fail", "swapped"],
                        help="Which datasets to process")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (for testing)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for GPU memory management")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Specific layers to extract (default: all)")
    parser.add_argument("--precision", default="fp16",
                        choices=["fp16", "int8", "int4"],
                        help="Model loading precision")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for .npz files")
    parser.add_argument("--config", default=None,
                        help="Model config YAML (e.g. configs/qwen3_8b.yaml). "
                             "Overrides --precision with auto-detected value.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing outputs")
    args = parser.parse_args()

    if args.verify_only:
        verify_outputs(args.output_dir, args.datasets)
        return

    # Load model
    print("Loading model...")
    if args.config:
        model, tokenizer, _ = load_model_from_config(args.config, precision=args.precision
                                                      if args.precision != "fp16" else None)
    else:
        model, tokenizer = load_model(precision=args.precision)
    device = next(model.parameters()).device
    print(f"  Model loaded on {device}, dtype={next(model.parameters()).dtype}")
    print(f"  Num layers: {model.config.num_hidden_layers}")

    # Process each dataset
    for dataset_name in args.datasets:
        filepath = DATASET_FILES[dataset_name]
        if not os.path.exists(filepath):
            print(f"Skipping {dataset_name}: {filepath} not found")
            continue

        print(f"\nProcessing {dataset_name}...")
        samples = load_jsonl(filepath, max_samples=args.max_samples)
        print(f"  Loaded {len(samples)} samples from {filepath}")

        hidden_states, source_labels = extract_hidden_states(
            model, tokenizer, samples, dataset_name,
            batch_size=args.batch_size, target_layers=args.layers
        )

        save_hidden_states(hidden_states, source_labels, dataset_name,
                           args.output_dir)

        # Free memory
        del hidden_states, source_labels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Verify
    verify_outputs(args.output_dir, args.datasets)
    print("\nDone!")


if __name__ == "__main__":
    main()
