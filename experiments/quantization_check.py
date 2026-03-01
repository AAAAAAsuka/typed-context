#!/usr/bin/env python3
"""Quantization robustness check — Phase 3.5.

Tests whether typed RoPE signal survives quantization (fp16, int8, int4)
by measuring attention gap and type category capacity at each precision.

Usage:
    python experiments/quantization_check.py                # full check (needs GPU)
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
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

PRECISIONS = ["fp16", "int8", "int4"]
ROTATION_ANGLES = [0.1, 0.2, math.pi / 8, math.pi / 4, math.pi / 2]


def measure_attention_gap_at_precision(model, tokenizer, samples, target_subspaces,
                                        rotation_angle):
    """Measure same-type vs cross-type attention gap at given precision."""
    import torch
    from model.hook_attention import install_type_rotation_hooks
    from utils import assign_source_labels

    device = next(model.parameters()).device
    hooks = install_type_rotation_hooks(model, target_subspaces, rotation_angle)

    gaps = []
    for sample in samples:
        input_ids, source_labels = assign_source_labels(
            tokenizer, sample["system"], sample["user"]
        )
        type_ids = torch.tensor([source_labels], dtype=torch.long)
        hooks.set_type_ids(type_ids)

        inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(inputs, output_attentions=True)

        # Compute gap for this sample
        for attn in outputs.attentions:
            a = attn[0].cpu().float().numpy()  # (heads, seq, seq)
            type_row = source_labels[np.newaxis, :]
            type_col = source_labels[:, np.newaxis]
            same = (type_row == type_col).astype(np.float32)
            cross = 1.0 - same
            for h in range(a.shape[0]):
                same_attn = (a[h] * same).sum() / max(same.sum(), 1)
                cross_attn = (a[h] * cross).sum() / max(cross.sum(), 1)
                gaps.append(same_attn - cross_attn)

        del outputs, inputs

    hooks.remove()
    return float(np.mean(gaps)), float(np.std(gaps))


def test_type_capacity(model, tokenizer, sample, target_subspaces, rotation_angle, max_k=8):
    """Find max k where adjacent types are distinguishable."""
    import torch
    from model.hook_attention import install_type_rotation_hooks

    device = next(model.parameters()).device
    hooks = install_type_rotation_hooks(model, target_subspaces, rotation_angle / max_k)

    input_ids = tokenizer(sample["system"] + " " + sample["user"],
                          return_tensors="pt").input_ids[0].tolist()
    seq_len = len(input_ids)

    # For each k, check if attention patterns are distinguishable
    max_distinguishable = 1
    for k in range(2, max_k + 1):
        type_ids_a = torch.zeros(1, seq_len, dtype=torch.long)
        type_ids_b = torch.ones(1, seq_len, dtype=torch.long)

        hooks.set_type_ids(type_ids_a)
        inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            out_a = model(inputs, output_attentions=True)
        attn_a = out_a.attentions[-1][0].cpu()

        hooks.set_type_ids(type_ids_b)
        with torch.no_grad():
            out_b = model(inputs, output_attentions=True)
        attn_b = out_b.attentions[-1][0].cpu()

        # Check if attention patterns are sufficiently different
        diff = (attn_a - attn_b).abs().mean().item()
        if diff > 1e-4:
            max_distinguishable = k
        else:
            break

        del out_a, out_b

    hooks.remove()
    return max_distinguishable


def generate_figure9(results, output_dir=OUTPUT_DIR):
    """Generate quantization survival heatmap."""
    os.makedirs(output_dir, exist_ok=True)

    precisions = list(results.keys())
    angles = sorted(set(
        float(k) for p in precisions for k in results[p].keys()
    ))

    # Build heatmap matrix
    gap_matrix = np.zeros((len(precisions), len(angles)))
    for i, p in enumerate(precisions):
        for j, a in enumerate(angles):
            key = f"{a:.4f}"
            if key in results[p]:
                gap_matrix[i, j] = results[p][key]["mean_gap"]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(gap_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:.3f}" for a in angles], rotation=45)
    ax.set_yticks(range(len(precisions)))
    ax.set_yticklabels(precisions)
    ax.set_xlabel("Rotation Angle")
    ax.set_ylabel("Precision")
    ax.set_title("Type Rotation Signal Survival Across Quantization", fontsize=14)

    # Add text annotations
    for i in range(len(precisions)):
        for j in range(len(angles)):
            val = gap_matrix[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Attention Gap")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig9_quant_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 9: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantization robustness check")
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

    from utils import load_model, load_model_from_config
    from analysis.extract_hidden_states import load_jsonl
    samples = load_jsonl(os.path.join(PROJECT_ROOT, "data", "normal.jsonl"),
                         args.num_samples)

    # Detect base precision for FP8 models
    precisions_to_test = list(PRECISIONS)
    if args.config:
        from utils import load_config
        cfg = load_config(args.config)
        if "fp8" in cfg.get("model_name", "").lower():
            precisions_to_test = ["fp8", "int8", "int4"]

    results = {}
    for precision in precisions_to_test:
        print(f"\nLoading model at {precision}...")
        try:
            if args.config:
                model, tokenizer, _ = load_model_from_config(
                    args.config, precision=precision,
                    attn_implementation="eager")
            else:
                model, tokenizer = load_model(precision=precision,
                                              attn_implementation="eager")
        except Exception as e:
            print(f"  Failed to load at {precision}: {e}")
            print(f"  Skipping this precision level.")
            continue
        results[precision] = {}

        for angle in ROTATION_ANGLES:
            mean_gap, std_gap = measure_attention_gap_at_precision(
                model, tokenizer, samples, target_subspaces, angle
            )
            capacity = test_type_capacity(
                model, tokenizer, samples[0], target_subspaces, angle
            )
            results[precision][f"{angle:.4f}"] = {
                "angle": angle, "mean_gap": mean_gap, "std_gap": std_gap,
                "type_capacity": capacity,
            }
            print(f"  angle={angle:.4f}: gap={mean_gap:.6f}, capacity={capacity}")

        del model, tokenizer

    # Generate Figure 9
    generate_figure9(results, args.output_dir)

    # Save results
    results_path = os.path.join(args.output_dir, "quantization_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Report
    print("\n=== Quantization Robustness Report ===")
    for precision in results.keys():
            gaps = [v["mean_gap"] for v in results[precision].values()]
            caps = [v["type_capacity"] for v in results[precision].values()]
            signal_preserved = any(g > 0.001 for g in gaps)
            print(f"  {precision}: mean gap={np.mean(gaps):.6f}, "
                  f"max capacity={max(caps)}, "
                  f"signal {'PRESERVED' if signal_preserved else 'LOST'}")

    # Verification
    print("\n=== Verification ===")
    fig_path = os.path.join(args.output_dir, "fig9_quant_heatmap.png")
    status = "OK" if os.path.exists(fig_path) else "MISSING"
    print(f"  {status}: fig9_quant_heatmap.png")
    print("\nDone!")


if __name__ == "__main__":
    main()
