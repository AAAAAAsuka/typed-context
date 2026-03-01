#!/usr/bin/env python3
"""Generate synthetic hidden-state .npz files for validation/testing.

Since running the real extraction script (extract_hidden_states.py) requires
a GPU and the full Llama-3.1-8B-Instruct model, this script produces
synthetic data that matches the exact same format.  It reads the first few
samples from each dataset JSONL, estimates token counts from the text
lengths, creates random hidden states and deterministic source labels, and
saves compressed .npz files identical in structure to what the real
extraction script produces.

Usage:
    python analysis/generate_synthetic_hidden_states.py
    python analysis/generate_synthetic_hidden_states.py --max-samples 5
    python analysis/generate_synthetic_hidden_states.py --layers 0 16 31
    python analysis/generate_synthetic_hidden_states.py --seed 42
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "hidden_states")

DATASET_FILES = {
    "normal":     os.path.join(DATA_DIR, "normal.jsonl"),
    "pi_success": os.path.join(DATA_DIR, "pi_success.jsonl"),
    "pi_fail":    os.path.join(DATA_DIR, "pi_fail.jsonl"),
    "swapped":    os.path.join(DATA_DIR, "swapped.jsonl"),
}

# Llama-3.1-8B-Instruct model parameters
HIDDEN_DIM = 4096
NUM_MODEL_LAYERS = 33  # layer indices 0..32 (embedding + 32 transformer layers)

# Approximate number of chat-template overhead tokens for Llama-3 Instruct.
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n ... <|eot_id|>
# <|start_header_id|>user<|end_header_id|>\n\n ... <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>\n\n
TEMPLATE_OVERHEAD_SYSTEM = 6   # BOS + system header tokens + trailing eot
TEMPLATE_OVERHEAD_USER = 6     # user header tokens + trailing eot
TEMPLATE_OVERHEAD_GENERATION = 4  # assistant header + generation prompt tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def estimate_token_count(text):
    """Rough estimate: ~4 characters per token for English text (Llama tokenizer)."""
    return max(1, len(text) // 4)


def simulate_source_labels(system_text, user_text, is_swapped=False):
    """Create source labels mimicking assign_source_labels from utils.py.

    For a normal (non-swapped) sample:
        - Tokens from the system turn (system header + system content + eot)
          are labeled 0.
        - Tokens from the user turn onward (user header + user content + eot
          + assistant header) are labeled 1.

    For a swapped sample:
        - The user content is placed in the system position and vice-versa,
          then labels are flipped (0->1, 1->0) so that the labels still
          reflect the *semantic* origin of the text.

    Returns:
        source_labels: np.ndarray of shape (total_tokens,), dtype int64
        total_tokens: int
    """
    sys_content_tokens = estimate_token_count(system_text)
    user_content_tokens = estimate_token_count(user_text)

    sys_portion = TEMPLATE_OVERHEAD_SYSTEM + sys_content_tokens
    user_portion = (TEMPLATE_OVERHEAD_USER + user_content_tokens
                    + TEMPLATE_OVERHEAD_GENERATION)
    total_tokens = sys_portion + user_portion

    if not is_swapped:
        labels = np.array(
            [0] * sys_portion + [1] * user_portion, dtype=np.int64
        )
    else:
        # In swapped: user text sits in system position, system text in user
        # position.  The real code first assigns labels based on position
        # (0 for first, 1 for second), then flips them.
        # After flipping: first portion (originally user text) gets label 1,
        # second portion (originally system text) gets label 0.
        swapped_sys_tokens = estimate_token_count(user_text)
        swapped_user_tokens = estimate_token_count(system_text)
        sys_portion_sw = TEMPLATE_OVERHEAD_SYSTEM + swapped_sys_tokens
        user_portion_sw = (TEMPLATE_OVERHEAD_USER + swapped_user_tokens
                           + TEMPLATE_OVERHEAD_GENERATION)
        total_tokens = sys_portion_sw + user_portion_sw
        # Before flip: [0]*sys_portion_sw + [1]*user_portion_sw
        # After  flip: [1]*sys_portion_sw + [0]*user_portion_sw
        labels = np.array(
            [1] * sys_portion_sw + [0] * user_portion_sw, dtype=np.int64
        )

    return labels, total_tokens


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def generate_synthetic(datasets, max_samples, target_layers, seed, output_dir):
    """Generate synthetic .npz files for each dataset."""
    rng = np.random.RandomState(seed)
    os.makedirs(output_dir, exist_ok=True)

    for ds_name in datasets:
        filepath = DATASET_FILES.get(ds_name)
        if filepath is None or not os.path.exists(filepath):
            print(f"Skipping {ds_name}: file not found ({filepath})")
            continue

        samples = load_jsonl(filepath, max_samples=max_samples)
        print(f"\nDataset: {ds_name}  ({len(samples)} samples from {filepath})")

        # Collect per-sample labels and token counts
        all_labels = []
        sample_token_counts = []
        for sample in samples:
            system_text = sample["system"]
            user_text = sample["user"]
            is_swapped = sample.get("swap_order", False)

            labels, n_tokens = simulate_source_labels(
                system_text, user_text, is_swapped=is_swapped
            )
            all_labels.append(labels)
            sample_token_counts.append(n_tokens)

        # Concatenate labels across samples
        source_labels = np.concatenate(all_labels, axis=0)
        total_tokens = source_labels.shape[0]

        print(f"  Total tokens: {total_tokens}")
        print(f"  Label distribution: 0(sys)={np.sum(source_labels==0)}, "
              f"1(user)={np.sum(source_labels==1)}")

        # Generate random hidden states for each target layer
        hidden_states = {}
        for layer_idx in target_layers:
            # Use float32 to match real extraction (hs[0].cpu().float().numpy())
            hidden_states[layer_idx] = rng.randn(
                total_tokens, HIDDEN_DIM
            ).astype(np.float32)
            print(f"  Generated hidden_layer_{layer_idx}: "
                  f"shape ({total_tokens}, {HIDDEN_DIM})")

        # Save to .npz  (matches save_hidden_states format in extract_hidden_states.py)
        save_dict = {"source_labels": source_labels}
        for layer_idx, hs in hidden_states.items():
            save_dict[f"hidden_layer_{layer_idx}"] = hs

        out_path = os.path.join(output_dir, f"{ds_name}.npz")
        np.savez_compressed(out_path, **save_dict)
        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Saved: {out_path}  ({file_size_mb:.2f} MB)")

        # Free memory
        del hidden_states, save_dict

    # ------------------------------------------------------------------
    # Verify outputs (same logic as extract_hidden_states.verify_outputs)
    # ------------------------------------------------------------------
    print("\n=== Verification ===")
    all_ok = True
    for ds_name in datasets:
        path = os.path.join(output_dir, f"{ds_name}.npz")
        if not os.path.exists(path):
            print(f"  FAIL: {path} not found")
            all_ok = False
            continue

        data = np.load(path)
        keys = list(data.keys())
        layer_keys = sorted([k for k in keys if k.startswith("hidden_layer_")])
        has_labels = "source_labels" in keys

        if not has_labels:
            print(f"  FAIL: {ds_name} missing source_labels")
            all_ok = False
            data.close()
            continue

        labels = data["source_labels"]
        print(f"  {ds_name}.npz: {len(layer_keys)} layers, "
              f"{labels.shape[0]} tokens, "
              f"labels 0={np.sum(labels==0)} / 1={np.sum(labels==1)}")

        # Check shape consistency across layers
        for lk in layer_keys:
            hs = data[lk]
            if hs.shape[0] != labels.shape[0]:
                print(f"    FAIL: {lk} shape {hs.shape} vs labels {labels.shape}")
                all_ok = False
            if hs.shape[1] != HIDDEN_DIM:
                print(f"    FAIL: {lk} hidden_dim {hs.shape[1]} != {HIDDEN_DIM}")
                all_ok = False

        data.close()

    if all_ok:
        print("  All checks passed!")
    else:
        print("  Some checks FAILED!")

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic hidden-state .npz files for testing"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["normal", "pi_success", "pi_fail", "swapped"],
        choices=["normal", "pi_success", "pi_fail", "swapped"],
        help="Which datasets to generate synthetic data for"
    )
    parser.add_argument(
        "--max-samples", type=int, default=3,
        help="Number of samples to read from each JSONL (default: 3)"
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[0, 8, 16, 24, 31],
        help="Layer indices to generate hidden states for (default: 0 8 16 24 31)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Output directory for .npz files"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=HIDDEN_DIM,
        help="Hidden dimension size (default: 4096)"
    )
    args = parser.parse_args()

    # Allow overriding hidden dim globally
    global HIDDEN_DIM
    HIDDEN_DIM = args.hidden_dim

    print("=== Synthetic Hidden State Generator ===")
    print(f"  Datasets:    {args.datasets}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Layers:      {args.layers}")
    print(f"  Hidden dim:  {HIDDEN_DIM}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output dir:  {args.output_dir}")

    ok = generate_synthetic(
        datasets=args.datasets,
        max_samples=args.max_samples,
        target_layers=args.layers,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    if ok:
        print("\nDone! Synthetic hidden states are ready for downstream analysis.")
    else:
        print("\nDone with errors -- check verification output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
