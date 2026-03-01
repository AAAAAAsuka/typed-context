#!/usr/bin/env python3
"""Attention pattern analysis under type rotation — Phase 3.

Computes attention weights with and without type rotation on representative
samples, measures the attention gap (same-type vs cross-type attention),
and generates attention heatmaps.

Usage:
    python experiments/attention_analysis.py                # full analysis (needs GPU)
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")


def compute_attention_with_rotation(model, tokenizer, samples, target_subspaces,
                                     rotation_angle, use_rotation=True):
    """Compute attention weights for samples with/without type rotation.

    Returns:
        attention_data: list of dicts per sample with:
            - 'attn_weights': np.array (num_layers, num_heads, seq_len, seq_len)
            - 'type_ids': np.array (seq_len,)
            - 'tokens': list of token strings
    """
    import torch
    from model.hook_attention import install_type_rotation_hooks
    from utils import assign_source_labels

    device = next(model.parameters()).device
    attention_data = []

    hooks = None
    if use_rotation:
        hooks = install_type_rotation_hooks(model, target_subspaces, rotation_angle)

    for sample in samples:
        input_ids, source_labels = assign_source_labels(
            tokenizer, sample["system"], sample["user"]
        )
        inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
        type_ids = torch.tensor([source_labels], dtype=torch.long)

        if hooks:
            hooks.set_type_ids(type_ids)

        with torch.no_grad():
            outputs = model(inputs, output_attentions=True)

        # outputs.attentions: tuple of (num_layers) tensors,
        # each shape (1, num_heads, seq_len, seq_len)
        attn_weights = np.stack([
            a[0].cpu().float().numpy() for a in outputs.attentions
        ])  # (num_layers, num_heads, seq_len, seq_len)

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        attention_data.append({
            "attn_weights": attn_weights,
            "type_ids": source_labels,
            "tokens": tokens,
        })

        del outputs, inputs

    if hooks:
        hooks.remove()

    return attention_data


def compute_attention_gap(attn_data):
    """Compute attention gap: mean(same-type) - mean(cross-type) per head/layer.

    Returns:
        gap: np.array (num_layers, num_heads) — attention gap per head
    """
    all_gaps = []
    for data in attn_data:
        attn = data["attn_weights"]  # (L, H, S, S)
        type_ids = data["type_ids"]   # (S,)
        num_layers, num_heads, seq_len, _ = attn.shape

        # Create same-type mask
        type_row = type_ids[np.newaxis, :]  # (1, S)
        type_col = type_ids[:, np.newaxis]  # (S, 1)
        same_mask = (type_row == type_col).astype(np.float32)  # (S, S)
        cross_mask = 1.0 - same_mask

        gap = np.zeros((num_layers, num_heads))
        for l in range(num_layers):
            for h in range(num_heads):
                a = attn[l, h]  # (S, S)
                same_attn = (a * same_mask).sum() / max(same_mask.sum(), 1)
                cross_attn = (a * cross_mask).sum() / max(cross_mask.sum(), 1)
                gap[l, h] = same_attn - cross_attn

        all_gaps.append(gap)

    return np.mean(all_gaps, axis=0)  # (L, H)


def generate_figure7(attn_data_no_rot, attn_data_with_rot, gap_no_rot, gap_with_rot,
                     output_dir=OUTPUT_DIR):
    """Generate 2x2 attention heatmap grid.

    Layout:
        [Without rotation, most affected head] [With rotation, most affected head]
        [Without rotation, least affected head] [With rotation, least affected head]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find most and least affected heads (largest/smallest gap change)
    gap_diff = gap_with_rot - gap_no_rot  # (L, H)
    most_affected = np.unravel_index(np.argmax(np.abs(gap_diff)), gap_diff.shape)
    least_affected = np.unravel_index(np.argmin(np.abs(gap_diff)), gap_diff.shape)

    most_layer, most_head = most_affected
    least_layer, least_head = least_affected

    # Use first sample for visualization
    sample_idx = 0
    attn_no = attn_data_no_rot[sample_idx]["attn_weights"]
    attn_with = attn_data_with_rot[sample_idx]["attn_weights"]
    type_ids = attn_data_no_rot[sample_idx]["type_ids"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    configs = [
        (axes[0, 0], attn_no[most_layer, most_head],
         f"Without Rotation — L{most_layer}/H{most_head} (Most Affected)"),
        (axes[0, 1], attn_with[most_layer, most_head],
         f"With Rotation — L{most_layer}/H{most_head} (Most Affected)"),
        (axes[1, 0], attn_no[least_layer, least_head],
         f"Without Rotation — L{least_layer}/H{least_head} (Least Affected)"),
        (axes[1, 1], attn_with[least_layer, least_head],
         f"With Rotation — L{least_layer}/H{least_head} (Least Affected)"),
    ]

    for ax, attn_map, title in configs:
        # Truncate for readability
        max_show = min(40, attn_map.shape[0])
        a = attn_map[:max_show, :max_show]

        im = ax.imshow(a, cmap="Blues", aspect="auto", vmin=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

        # Color axis labels by type
        type_colors = {0: "blue", 1: "red"}
        for idx in range(max_show):
            if idx < len(type_ids):
                color = type_colors.get(type_ids[idx], "black")
                ax.get_xticklabels() and None  # just set tick colors below

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Attention Patterns: Effect of Type Rotation", fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig7_attention_heatmaps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 7: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Attention pattern analysis")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--config-dir", default=CONFIG_DIR)
    parser.add_argument("--config", default=None,
                        help="Model config YAML (e.g. configs/qwen3_8b.yaml)")
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
    from analysis.extract_hidden_states import load_jsonl
    if args.config:
        model, tokenizer, _ = load_model_from_config(
            args.config, attn_implementation="eager")
    else:
        model, tokenizer = load_model(attn_implementation="eager")
    samples = load_jsonl(os.path.join(DATA_DIR, "normal.jsonl"), args.num_samples)

    print("Computing attention without rotation...")
    attn_no_rot = compute_attention_with_rotation(
        model, tokenizer, samples, target_subspaces, rotation_angle,
        use_rotation=False
    )
    print("Computing attention with rotation...")
    attn_with_rot = compute_attention_with_rotation(
        model, tokenizer, samples, target_subspaces, rotation_angle,
        use_rotation=True
    )

    # Compute attention gaps
    gap_no_rot = compute_attention_gap(attn_no_rot)
    gap_with_rot = compute_attention_gap(attn_with_rot)

    print(f"\nAttention gap without rotation: mean={gap_no_rot.mean():.6f}")
    print(f"Attention gap with rotation: mean={gap_with_rot.mean():.6f}")

    # Find heads with largest positive gap
    gap_diff = gap_with_rot - gap_no_rot
    top_heads = np.argsort(gap_diff.ravel())[-5:]
    print("\nTop 5 type-sensitive heads (largest gap increase):")
    for idx in reversed(top_heads):
        l, h = np.unravel_index(idx, gap_diff.shape)
        print(f"  Layer {l}, Head {h}: gap_diff={gap_diff[l,h]:.6f}")

    # Check if gap is positive for at least some heads
    positive_count = (gap_with_rot > 0).sum()
    total_heads = gap_with_rot.size
    print(f"\nHeads with positive gap: {positive_count}/{total_heads}")

    # Generate Figure 7
    generate_figure7(attn_no_rot, attn_with_rot, gap_no_rot, gap_with_rot,
                     args.output_dir)

    # Save results
    results = {
        "gap_no_rotation_mean": float(gap_no_rot.mean()),
        "gap_with_rotation_mean": float(gap_with_rot.mean()),
        "positive_gap_heads": int(positive_count),
        "total_heads": int(total_heads),
    }
    results_path = os.path.join(args.output_dir, "attention_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Verification
    print("\n=== Verification ===")
    fig_path = os.path.join(args.output_dir, "fig7_attention_heatmaps.png")
    status = "OK" if os.path.exists(fig_path) else "MISSING"
    print(f"  {status}: fig7_attention_heatmaps.png")
    if positive_count > 0:
        print(f"  PASS: attention gap is positive for {positive_count} heads")
    else:
        print(f"  CHECK: no heads with positive attention gap")

    print("\nDone!")


if __name__ == "__main__":
    main()
