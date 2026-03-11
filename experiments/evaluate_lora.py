#!/usr/bin/env python3
"""LoRA evaluation with ablation — Phase 4.

Evaluates 4 conditions:
  1. Baseline (no LoRA, no rotation)
  2. LoRA without rotation
  3. LoRA with rotation
  4. LoRA with rotation removed at test time (ablation)

Generates Figure 10 (ASR bar chart) and Figure 11 (radar chart).

Usage:
    python experiments/evaluate_lora.py                 # full eval (needs GPU)
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


def generate_figure10(results, output_dir=OUTPUT_DIR):
    """Generate ASR bar chart with ablation."""
    os.makedirs(output_dir, exist_ok=True)

    conditions = list(results.keys())
    strict_asr = [results[c]["strict_ASR"] * 100 for c in conditions]
    soft_asr = [results[c]["soft_ASR"] * 100 for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, strict_asr, width, label="Strict ASR",
                    color="#F44336", alpha=0.8)
    bars2 = ax.bar(x + width / 2, soft_asr, width, label="Soft ASR",
                    color="#FF9800", alpha=0.8, hatch="//")

    label_map = {
        "baseline": "Baseline\n(no LoRA)",
        "lora_no_rotation": "LoRA\n(no rotation)",
        "lora_with_rotation": "LoRA\n+ Rotation",
        "lora_rotation_removed": "LoRA\n(rotation removed)",
    }

    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(c, c) for c in conditions], fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("LoRA + Typed RoPE: Ablation Study", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate ablation arrow
    if "lora_with_rotation" in results and "lora_rotation_removed" in results:
        idx_with = conditions.index("lora_with_rotation")
        idx_removed = conditions.index("lora_rotation_removed")
        y_with = strict_asr[idx_with]
        y_removed = strict_asr[idx_removed]
        ax.annotate("",
                     xy=(idx_removed, y_removed + 1),
                     xytext=(idx_with, y_with + 1),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.text((idx_with + idx_removed) / 2, max(y_with, y_removed) + 3,
                f"+{y_removed - y_with:.1f}%\n(rotation needed!)",
                ha="center", fontsize=9, color="red")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig10_lora_ablation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 10: {out_path}")


def generate_figure11(results, output_dir=OUTPUT_DIR):
    """Generate per-attack-type breakdown radar chart."""
    os.makedirs(output_dir, exist_ok=True)

    categories = ["extraction", "override", "role_play", "smuggling"]
    num_cats = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = {"baseline": "#F44336", "lora_no_rotation": "#FF9800",
              "lora_with_rotation": "#4CAF50", "lora_rotation_removed": "#9C27B0"}
    labels = {"baseline": "Baseline", "lora_no_rotation": "LoRA (no rot)",
              "lora_with_rotation": "LoRA + Rotation",
              "lora_rotation_removed": "LoRA (rot removed)"}

    for cond_name, cond_data in results.items():
        per_cat = cond_data.get("per_category", {})
        values = [per_cat.get(c, 0) * 100 for c in categories]
        values += values[:1]

        ax.plot(angles, values, marker="o", linewidth=2,
                color=colors.get(cond_name, "#999"),
                label=labels.get(cond_name, cond_name))
        ax.fill(angles, values, alpha=0.1, color=colors.get(cond_name, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=10)
    ax.set_ylim(0, 60)
    ax.set_title("Per-Attack-Type ASR Breakdown", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig11_attack_breakdown.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 11: {out_path}")


def evaluate_condition(model, tokenizer, samples, hooks=None):
    """Run evaluation on benchmark samples, return judgments and per-category ASR."""
    import torch
    from utils import assign_source_labels

    device = next(model.parameters()).device
    judgments = []
    per_category = {}

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(samples)}]", flush=True)
        if hooks is not None:
            input_ids, source_labels = assign_source_labels(
                tokenizer, sample["system"], sample["user"]
            )
            type_ids = torch.tensor([source_labels], dtype=torch.long)
            hooks.set_type_ids(type_ids)
            inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
        else:
            messages = [
                {"role": "system", "content": sample["system"]},
                {"role": "user", "content": sample["user"]},
            ]
            try:
                inputs = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt",
                    enable_thinking=False
                ).to(device)
            except TypeError:
                inputs = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt",
                ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs, max_new_tokens=128,
                do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(output_ids[0][inputs.shape[-1]:],
                                     skip_special_tokens=True)

        from experiments.icl_experiment import keyword_judge
        j = keyword_judge(response, sample)
        judgments.append(j)

        cat = sample.get("attack_category", "extraction")
        per_category.setdefault(cat, []).append(j)

    from experiments.icl_experiment import compute_metrics
    metrics = compute_metrics(judgments)

    # Per-category strict ASR
    cat_asr = {}
    for cat, cat_judgments in per_category.items():
        cat_asr[cat] = compute_metrics(cat_judgments)["strict_ASR"]
    metrics["per_category"] = cat_asr

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LoRA evaluation with ablation")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--config-dir", default=CONFIG_DIR)
    parser.add_argument("--config", default=None,
                        help="Model config YAML (e.g. configs/qwen3_8b.yaml)")
    parser.add_argument("--adapter-dir", default=os.path.join(OUTPUT_DIR, "lora_adapter"))
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from utils import load_model, load_model_from_config
    from model.hook_attention import install_type_rotation_hooks
    from analysis.extract_hidden_states import load_jsonl

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

    # Load benchmark data
    bench_path = os.path.join(DATA_DIR, "pi_benchmark.jsonl")
    if not os.path.exists(bench_path):
        bench_path = os.path.join(DATA_DIR, "pi_attacks.jsonl")
    samples = load_jsonl(bench_path, args.num_samples)
    print(f"Loaded {len(samples)} benchmark samples")

    results = {}

    # Condition 1: Baseline (no LoRA, no rotation)
    print("\n=== Condition: Baseline ===")
    print("Loading base model...")
    if args.config:
        model, tokenizer, _ = load_model_from_config(args.config)
    else:
        model, tokenizer = load_model()
    results["baseline"] = evaluate_condition(model, tokenizer, samples)
    print(f"  Strict ASR: {results['baseline']['strict_ASR']:.4f}")

    # Condition 2: LoRA without rotation
    print("\n=== Condition: LoRA (no rotation) ===")
    if os.path.exists(args.adapter_dir):
        lora_model = PeftModel.from_pretrained(model, args.adapter_dir)
        lora_model.eval()
        results["lora_no_rotation"] = evaluate_condition(
            lora_model, tokenizer, samples)
        print(f"  Strict ASR: {results['lora_no_rotation']['strict_ASR']:.4f}")
    else:
        print(f"  Adapter not found at {args.adapter_dir}, skipping.")

    # Condition 3: LoRA with rotation
    print("\n=== Condition: LoRA + Rotation ===")
    if os.path.exists(args.adapter_dir):
        hooks = install_type_rotation_hooks(
            lora_model, target_subspaces, rotation_angle)
        results["lora_with_rotation"] = evaluate_condition(
            lora_model, tokenizer, samples, hooks=hooks)
        hooks.remove()
        print(f"  Strict ASR: {results['lora_with_rotation']['strict_ASR']:.4f}")

        # Condition 4: LoRA with rotation removed (ablation)
        print("\n=== Condition: LoRA (rotation removed) ===")
        results["lora_rotation_removed"] = evaluate_condition(
            lora_model, tokenizer, samples)
        print(f"  Strict ASR: {results['lora_rotation_removed']['strict_ASR']:.4f}")

        del lora_model
    else:
        print(f"  Adapter not found, skipping conditions 3 & 4.")

    del model

    # Generate figures
    generate_figure10(results, args.output_dir)
    generate_figure11(results, args.output_dir)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "lora_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Verification
    print("\n=== Verification ===")
    for fname in ["fig10_lora_ablation.png", "fig11_attack_breakdown.png"]:
        path = os.path.join(args.output_dir, fname)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  {status}: {fname}")

    # Check ablation: rotation removed should have higher ASR
    if "lora_with_rotation" in results and "lora_rotation_removed" in results:
        asr_with = results["lora_with_rotation"]["strict_ASR"]
        asr_removed = results["lora_rotation_removed"]["strict_ASR"]
        if asr_removed > asr_with:
            print(f"  PASS: Rotation removal increases ASR ({asr_with:.4f} → {asr_removed:.4f})")
            print(f"  → Model learned to use type signal!")
        else:
            print(f"  CHECK: ASR didn't increase with rotation removal")

    print("\nDone!")


if __name__ == "__main__":
    main()
