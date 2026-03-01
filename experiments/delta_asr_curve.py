#!/usr/bin/env python3
"""Experiment B1: delta-ASR curve — map certified attention gap to empirical ASR.

Hypothesis: certified attention gap delta has a strong monotonic decreasing
relationship with empirical ASR. There exists a "phase transition" delta
threshold beyond which ASR drops below 5%.

Design:
  - delta = mean attention gap (same-type minus cross-type) in target subspaces
  - Systematically vary delta via:
    |S| in {1, 2, 4, 6, 8, 12, 16} x theta in {0, pi/8, pi/4, 3pi/8, pi/2}
    = 35 configurations
  - For each config: compute theoretical delta, evaluate strict ASR
  - Generate scatter plot with decay curve fit
  - Break down by 4 attack categories
  - Compare with-LoRA vs without-LoRA (pure ICL)

Usage:
    python experiments/delta_asr_curve.py --adapter-dir outputs/lora_adapter
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
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

# Sweep parameters
NUM_SUBSPACES_LIST = [1, 2, 4, 6, 8, 12, 16]
THETA_LIST = [0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2]
THETA_LABELS = ["0", "π/8", "π/4", "3π/8", "π/2"]

# Attack categories
ATTACK_CATEGORIES = ["extraction", "override", "role_play", "smuggling"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_jsonl(path, max_samples=None):
    """Load samples from a JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
    return samples


def compute_theoretical_delta(theta, num_subspaces, total_subspaces=64):
    """Compute theoretical attention gap delta.

    In target subspaces, cross-type attention is modulated by cos(theta_A - theta_B).
    For system (type 0, angle=0) and user (type 1, angle=theta):
        same-type dot product factor: cos(0) = 1.0
        cross-type dot product factor: cos(theta)

    Delta = (same_type - cross_type) averaged over target subspaces.
    Since non-target subspaces contribute equally to both, the effective delta
    is weighted by the fraction of subspaces that are targets.

    delta = (num_subspaces / total_subspaces) * (1 - cos(theta))
    """
    if theta == 0 or num_subspaces == 0:
        return 0.0
    return (num_subspaces / total_subspaces) * (1.0 - math.cos(theta))


def exponential_decay(x, a, b, c):
    """Exponential decay: ASR = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def logistic_decay(x, L, k, x0, c):
    """Logistic decay: ASR = L / (1 + exp(k*(x - x0))) + c"""
    return L / (1.0 + np.exp(k * (x - x0))) + c


# ---------------------------------------------------------------------------
# Real model evaluation
# ---------------------------------------------------------------------------

def evaluate_config(model, tokenizer, pi_samples, target_subspaces, theta):
    """Evaluate strict ASR for a given (target_subspaces, theta) config."""
    from utils import assign_source_labels
    from experiments.icl_experiment import keyword_judge, compute_metrics
    from experiments.trust_hierarchy_sweep import CustomAngleHooks

    type_angle_map = {0: 0.0, 1: theta}
    hooks = CustomAngleHooks(target_subspaces, type_angle_map)
    hooks.install(model)

    import torch
    device = next(model.parameters()).device
    all_judgments = []
    per_cat_judgments = {cat: [] for cat in ATTACK_CATEGORIES}

    for i, sample in enumerate(pi_samples):
        if (i + 1) % 50 == 0:
            print(f"      [{i+1}/{len(pi_samples)}]", flush=True)

        input_ids, source_labels = assign_source_labels(
            tokenizer, sample["system"], sample["user"]
        )
        type_ids = torch.tensor([source_labels], dtype=torch.long)
        hooks.set_type_ids(type_ids)

        inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                inputs, max_new_tokens=128, temperature=0.7, top_p=0.9,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            output_ids[0][len(input_ids):], skip_special_tokens=True
        )

        j = keyword_judge(response, sample)
        all_judgments.append(j)
        cat = sample.get("attack_category", "extraction")
        if cat in per_cat_judgments:
            per_cat_judgments[cat].append(j)

    hooks.remove()

    overall = compute_metrics(all_judgments)
    per_cat = {cat: compute_metrics(jl) for cat, jl in per_cat_judgments.items()}

    return overall, per_cat


def run_real_sweep(model, tokenizer, pi_samples, base_subspaces):
    """Run full (|S|, theta) sweep on real model."""
    results = []

    total_configs = len(NUM_SUBSPACES_LIST) * len(THETA_LIST)
    config_idx = 0

    for num_sub in NUM_SUBSPACES_LIST:
        # Use first num_sub subspaces from the base list (pad if needed)
        if num_sub <= len(base_subspaces):
            target_subs = base_subspaces[:num_sub]
        else:
            # Extend with adjacent subspaces
            target_subs = list(base_subspaces)
            for extra in range(len(base_subspaces) - 1, -1, -1):
                if len(target_subs) >= num_sub:
                    break
                candidate = base_subspaces[0] - (len(base_subspaces) - extra)
                if candidate >= 0 and candidate not in target_subs:
                    target_subs.insert(0, candidate)
            # If still not enough, add more from lower indices
            idx = base_subspaces[0] - len(target_subs)
            while len(target_subs) < num_sub and idx >= 0:
                if idx not in target_subs:
                    target_subs.insert(0, idx)
                idx -= 1

        for theta_idx, theta in enumerate(THETA_LIST):
            config_idx += 1
            delta = compute_theoretical_delta(theta, len(target_subs))

            print(f"\n  Config {config_idx}/{total_configs}: "
                  f"|S|={len(target_subs)}, θ={THETA_LABELS[theta_idx]}, "
                  f"δ={delta:.4f}")

            if theta == 0:
                # No rotation = baseline, skip heavy evaluation
                # Use a fixed baseline ASR
                results.append({
                    "num_subspaces": len(target_subs),
                    "theta": theta,
                    "theta_label": THETA_LABELS[theta_idx],
                    "theoretical_delta": round(delta, 6),
                    "strict_ASR": None,  # filled below
                    "per_category_ASR": {},
                    "mode": "lora",
                    "_is_baseline": True,
                })
                continue

            overall, per_cat = evaluate_config(
                model, tokenizer, pi_samples, target_subs, theta
            )

            cat_asrs = {cat: per_cat[cat]["strict_ASR"] for cat in ATTACK_CATEGORIES}

            results.append({
                "num_subspaces": len(target_subs),
                "theta": theta,
                "theta_label": THETA_LABELS[theta_idx],
                "theoretical_delta": round(delta, 6),
                "strict_ASR": round(overall["strict_ASR"], 4),
                "per_category_ASR": {k: round(v, 4) for k, v in cat_asrs.items()},
                "mode": "lora",
            })

    # Fill baseline entries (theta=0) with average of baselines
    # or run baseline once
    baseline_entries = [r for r in results if r.get("_is_baseline")]
    non_baseline = [r for r in results if not r.get("_is_baseline")]

    if baseline_entries and non_baseline:
        # Evaluate baseline once (no rotation)
        print("\n  Evaluating baseline (no rotation)...")
        from experiments.icl_experiment import keyword_judge, compute_metrics
        from utils import assign_source_labels
        import torch
        device = next(model.parameters()).device
        judgments = []
        for i, sample in enumerate(pi_samples):
            if (i + 1) % 50 == 0:
                print(f"      [{i+1}/{len(pi_samples)}]", flush=True)
            messages = [
                {"role": "system", "content": sample["system"]},
                {"role": "user", "content": sample["user"]},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    inputs, max_new_tokens=128, temperature=0.7, top_p=0.9,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                output_ids[0][len(input_ids):], skip_special_tokens=True
            )
            judgments.append(keyword_judge(response, sample))

        baseline_asr = compute_metrics(judgments)["strict_ASR"]
        for entry in baseline_entries:
            entry["strict_ASR"] = round(baseline_asr, 4)
            entry.pop("_is_baseline", None)

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def fit_decay_curve(deltas, asrs):
    """Fit exponential decay curve to delta-ASR data.

    Returns fitted parameters and R^2.
    """
    deltas = np.array(deltas)
    asrs = np.array(asrs)

    # Remove zero-delta points for fitting
    mask = deltas > 0
    if mask.sum() < 3:
        return None, None

    x = deltas[mask]
    y = asrs[mask]

    try:
        popt, _ = curve_fit(
            exponential_decay, x, y,
            p0=[0.4, 10.0, 0.02],
            bounds=([0, 0, 0], [1.0, 100.0, 0.5]),
            maxfev=5000,
        )
        y_pred = exponential_decay(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt, r2
    except (RuntimeError, ValueError):
        return None, None


def find_phase_transition(popt, threshold=0.05):
    """Find delta where fitted ASR drops below threshold."""
    if popt is None:
        return None
    a, b, c = popt
    if c >= threshold:
        return None  # never drops below threshold
    if a + c <= threshold:
        return 0.0  # always below threshold
    # a * exp(-b * delta) + c = threshold
    # exp(-b * delta) = (threshold - c) / a
    delta_critical = -math.log((threshold - c) / a) / b
    return delta_critical


def generate_figure(results_lora, results_icl, output_dir):
    """Generate delta-ASR scatter plot with fitted curve.

    Layout: 2x2 grid
      Top-left: Overall delta-ASR scatter with fit (LoRA + ICL)
      Top-right: Per-category delta-ASR breakdown (LoRA)
      Bottom-left: LoRA vs ICL comparison
      Bottom-right: Phase transition annotation
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ---- Colors and markers ----
    cat_colors = {
        "extraction": "#E53935",
        "override": "#1E88E5",
        "role_play": "#43A047",
        "smuggling": "#FB8C00",
    }
    # Marker by |S| for visual grouping
    sub_markers = {1: "o", 2: "s", 4: "^", 6: "D", 8: "v", 12: "p", 16: "h"}

    # ==== Panel 1: Overall delta-ASR (LoRA) ====
    ax1 = axes[0, 0]
    deltas_lora = [r["theoretical_delta"] for r in results_lora]
    asrs_lora = [r["strict_ASR"] for r in results_lora]

    for r in results_lora:
        marker = sub_markers.get(r["num_subspaces"], "o")
        ax1.scatter(r["theoretical_delta"], r["strict_ASR"],
                    marker=marker, s=60, color="#E53935", alpha=0.7,
                    edgecolors="black", linewidths=0.5, zorder=5)

    # Fit decay curve
    popt_lora, r2_lora = fit_decay_curve(deltas_lora, asrs_lora)
    if popt_lora is not None:
        x_fit = np.linspace(0, max(deltas_lora) * 1.1, 200)
        y_fit = exponential_decay(x_fit, *popt_lora)
        ax1.plot(x_fit, y_fit, "-", color="#C62828", linewidth=2.5,
                 label=f"Exp. fit (R²={r2_lora:.3f})", zorder=3)

        # Phase transition
        delta_crit = find_phase_transition(popt_lora)
        if delta_crit is not None and delta_crit < max(deltas_lora) * 1.2:
            ax1.axvline(x=delta_crit, color="gray", linestyle="--",
                        alpha=0.7, linewidth=1.5)
            ax1.annotate(
                f"Phase transition\nδ = {delta_crit:.3f}",
                xy=(delta_crit, 0.05),
                xytext=(delta_crit + 0.02, 0.15),
                fontsize=9, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
            )

    ax1.axhline(y=0.05, color="#888", linestyle=":", alpha=0.5, linewidth=1)
    ax1.text(max(deltas_lora) * 0.95, 0.06, "ASR = 5%", fontsize=8,
             color="#888", ha="right")
    ax1.set_xlabel("Theoretical Attention Gap δ", fontsize=12)
    ax1.set_ylabel("Strict ASR", fontsize=12)
    ax1.set_title("B1: δ-ASR Curve (LoRA)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.02, max(asrs_lora) * 1.15 if asrs_lora else 0.5)
    ax1.grid(True, alpha=0.3)

    # ==== Panel 2: Per-category breakdown ====
    ax2 = axes[0, 1]
    for cat in ATTACK_CATEGORIES:
        cat_deltas = [r["theoretical_delta"] for r in results_lora
                      if r["theoretical_delta"] > 0]
        cat_asrs = [r["per_category_ASR"].get(cat, 0) for r in results_lora
                    if r["theoretical_delta"] > 0]
        ax2.scatter(cat_deltas, cat_asrs, s=40, color=cat_colors[cat],
                    alpha=0.6, label=cat, edgecolors="black", linewidths=0.3)

        # Fit per-category
        popt_cat, _ = fit_decay_curve(cat_deltas, cat_asrs)
        if popt_cat is not None:
            x_fit = np.linspace(0, max(cat_deltas) * 1.1, 100)
            y_fit = exponential_decay(x_fit, *popt_cat)
            ax2.plot(x_fit, y_fit, "-", color=cat_colors[cat],
                     linewidth=1.5, alpha=0.8)

    ax2.axhline(y=0.05, color="#888", linestyle=":", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Theoretical Attention Gap δ", fontsize=12)
    ax2.set_ylabel("Strict ASR", fontsize=12)
    ax2.set_title("Per-Attack-Category δ-ASR", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ==== Panel 3: LoRA vs ICL comparison ====
    ax3 = axes[1, 0]
    deltas_icl = [r["theoretical_delta"] for r in results_icl]
    asrs_icl = [r["strict_ASR"] for r in results_icl]

    ax3.scatter(deltas_lora, asrs_lora, s=50, color="#E53935", alpha=0.6,
                label="LoRA + Rotation", edgecolors="black", linewidths=0.3)
    ax3.scatter(deltas_icl, asrs_icl, s=50, color="#1E88E5", alpha=0.6,
                marker="^", label="ICL Only", edgecolors="black", linewidths=0.3)

    # Fit both
    if popt_lora is not None:
        x_fit = np.linspace(0, max(max(deltas_lora), max(deltas_icl)) * 1.1, 200)
        y_lora = exponential_decay(x_fit, *popt_lora)
        ax3.plot(x_fit, y_lora, "-", color="#C62828", linewidth=2, alpha=0.8)

    popt_icl, r2_icl = fit_decay_curve(deltas_icl, asrs_icl)
    if popt_icl is not None:
        x_fit = np.linspace(0, max(deltas_icl) * 1.1, 200)
        y_icl = exponential_decay(x_fit, *popt_icl)
        ax3.plot(x_fit, y_icl, "--", color="#1565C0", linewidth=2, alpha=0.8)

    ax3.axhline(y=0.05, color="#888", linestyle=":", alpha=0.5, linewidth=1)
    ax3.set_xlabel("Theoretical Attention Gap δ", fontsize=12)
    ax3.set_ylabel("Strict ASR", fontsize=12)
    ax3.set_title("LoRA vs ICL-Only δ-ASR", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ==== Panel 4: Phase transition and |S| legend ====
    ax4 = axes[1, 1]

    # Color by theta to show angle contribution
    theta_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(THETA_LIST)))
    for r in results_lora:
        theta_idx = THETA_LIST.index(r["theta"])
        marker = sub_markers.get(r["num_subspaces"], "o")
        ax4.scatter(r["theoretical_delta"], r["strict_ASR"],
                    marker=marker, s=80, color=theta_colors[theta_idx],
                    alpha=0.8, edgecolors="black", linewidths=0.5, zorder=5)

    # Add custom legends
    # Marker legend for |S|
    legend_markers = []
    for ns, mk in sorted(sub_markers.items()):
        legend_markers.append(
            ax4.scatter([], [], marker=mk, s=60, color="gray",
                        edgecolors="black", linewidths=0.5, label=f"|S|={ns}")
        )
    leg1 = ax4.legend(handles=legend_markers, loc="upper right", fontsize=8,
                      title="|S| (subspaces)", title_fontsize=9)
    ax4.add_artist(leg1)

    # Color legend for theta
    from matplotlib.patches import Patch
    theta_patches = [Patch(color=theta_colors[i], label=THETA_LABELS[i])
                     for i in range(len(THETA_LIST))]
    ax4.legend(handles=theta_patches, loc="center right", fontsize=8,
               title="θ (angle)", title_fontsize=9)

    if popt_lora is not None:
        x_fit = np.linspace(0, max(deltas_lora) * 1.1, 200)
        y_fit = exponential_decay(x_fit, *popt_lora)
        ax4.plot(x_fit, y_fit, "-", color="#C62828", linewidth=2, alpha=0.6)

    ax4.axhline(y=0.05, color="#888", linestyle=":", alpha=0.5, linewidth=1)
    ax4.set_xlabel("Theoretical Attention Gap δ", fontsize=12)
    ax4.set_ylabel("Strict ASR", fontsize=12)
    ax4.set_title("Config Space (|S| × θ)", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("B1: Certified Attention Gap → Empirical Attack Success Rate",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig_delta_asr.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp B1: delta-ASR curve — certified gap vs empirical ASR"
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--config-dir", default=CONFIG_DIR)
    parser.add_argument("--adapter-dir",
                        default=os.path.join(OUTPUT_DIR, "lora_adapter"))
    parser.add_argument("--max-pi-samples", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load target subspaces (base set from configs)
    ts_path = os.path.join(args.config_dir, "target_subspaces.json")
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            base_subspaces = json.load(f)["target_subspaces"]
    else:
        base_subspaces = [59, 60, 61, 62, 63]

    print(f"Base target subspaces: {base_subspaces}")
    print(f"|S| sweep: {NUM_SUBSPACES_LIST}")
    print(f"θ sweep: {THETA_LABELS}")
    print(f"Total configs: {len(NUM_SUBSPACES_LIST) * len(THETA_LIST)}")

    import torch
    from peft import PeftModel
    from utils import load_model

    # Load base model on GPU 0
    model, tokenizer = load_model(device_map={"": 0})

    # Load PI benchmark
    pi_path = os.path.join(DATA_DIR, "pi_attacks.jsonl")
    if not os.path.exists(pi_path):
        pi_path = os.path.join(DATA_DIR, "pi_benchmark.jsonl")
    pi_samples = load_jsonl(pi_path, args.max_pi_samples)
    print(f"Loaded {len(pi_samples)} PI samples")

    # --- LoRA mode ---
    if os.path.exists(args.adapter_dir):
        print(f"\nLoading LoRA adapter from {args.adapter_dir}")
        lora_model = PeftModel.from_pretrained(model, args.adapter_dir)
        lora_model.eval()
    else:
        print(f"Warning: LoRA adapter not found at {args.adapter_dir}, "
              "using base model for LoRA mode")
        lora_model = model

    print("\n=== LoRA + Rotation sweep ===")
    results_lora = run_real_sweep(
        lora_model, tokenizer, pi_samples, base_subspaces
    )
    for r in results_lora:
        r["mode"] = "lora"

    # --- ICL mode (base model, no LoRA) ---
    if os.path.exists(args.adapter_dir):
        # Unload LoRA to get base model back
        del lora_model
        torch.cuda.empty_cache()
        model, tokenizer = load_model(device_map={"": 0})

    print("\n=== ICL-only sweep ===")
    results_icl = run_real_sweep(
        model, tokenizer, pi_samples, base_subspaces
    )
    for r in results_icl:
        r["mode"] = "icl"

    del model
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("LoRA + Rotation Results")
    print("=" * 100)
    print(f"{'|S|':>5} | {'θ':>6} | {'δ':>8} | {'ASR':>8} | "
          f"{'extract':>8} | {'override':>8} | {'role_play':>8} | {'smuggling':>8}")
    print("-" * 100)
    for r in sorted(results_lora, key=lambda x: x["theoretical_delta"]):
        cats = r["per_category_ASR"]
        print(f"{r['num_subspaces']:>5} | {r['theta_label']:>6} | "
              f"{r['theoretical_delta']:>8.4f} | {r['strict_ASR']:>8.4f} | "
              f"{cats.get('extraction', 0):>8.4f} | {cats.get('override', 0):>8.4f} | "
              f"{cats.get('role_play', 0):>8.4f} | {cats.get('smuggling', 0):>8.4f}")

    print("\n" + "=" * 100)
    print("ICL-Only Results")
    print("=" * 100)
    print(f"{'|S|':>5} | {'θ':>6} | {'δ':>8} | {'ASR':>8}")
    print("-" * 100)
    for r in sorted(results_icl, key=lambda x: x["theoretical_delta"]):
        print(f"{r['num_subspaces']:>5} | {r['theta_label']:>6} | "
              f"{r['theoretical_delta']:>8.4f} | {r['strict_ASR']:>8.4f}")

    # -----------------------------------------------------------------------
    # Generate figure
    # -----------------------------------------------------------------------
    generate_figure(results_lora, results_icl, args.output_dir)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    all_results = {
        "lora": results_lora,
        "icl": results_icl,
        "config": {
            "num_subspaces_list": NUM_SUBSPACES_LIST,
            "theta_list": THETA_LIST,
            "theta_labels": THETA_LABELS,
            "base_subspaces": base_subspaces,
        },
    }

    # Add fitted curve parameters
    deltas_lora = [r["theoretical_delta"] for r in results_lora]
    asrs_lora = [r["strict_ASR"] for r in results_lora]
    popt_lora, r2_lora = fit_decay_curve(deltas_lora, asrs_lora)
    if popt_lora is not None:
        all_results["fit_lora"] = {
            "params": {"a": popt_lora[0], "b": popt_lora[1], "c": popt_lora[2]},
            "r_squared": r2_lora,
            "phase_transition_delta": find_phase_transition(popt_lora),
        }

    deltas_icl = [r["theoretical_delta"] for r in results_icl]
    asrs_icl = [r["strict_ASR"] for r in results_icl]
    popt_icl, r2_icl = fit_decay_curve(deltas_icl, asrs_icl)
    if popt_icl is not None:
        all_results["fit_icl"] = {
            "params": {"a": popt_icl[0], "b": popt_icl[1], "c": popt_icl[2]},
            "r_squared": r2_icl,
            "phase_transition_delta": find_phase_transition(popt_icl),
        }

    results_path = os.path.join(args.output_dir, "delta_asr_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # 1. Figure saved
    fig_ok = os.path.exists(os.path.join(args.output_dir, "fig_delta_asr.png"))
    print(f"  Figure saved:              {'PASS' if fig_ok else 'FAIL'}")

    # 2. Results saved
    res_ok = os.path.exists(results_path)
    print(f"  Results saved:             {'PASS' if res_ok else 'FAIL'}")

    # 3. Strong monotonic decreasing relationship
    # Sort by delta, check Spearman rank correlation
    from scipy.stats import spearmanr
    sorted_lora = sorted(results_lora, key=lambda x: x["theoretical_delta"])
    deltas_sorted = [r["theoretical_delta"] for r in sorted_lora]
    asrs_sorted = [r["strict_ASR"] for r in sorted_lora]

    # Filter non-zero delta for correlation
    nonzero_mask = [d > 0 for d in deltas_sorted]
    deltas_nz = [d for d, m in zip(deltas_sorted, nonzero_mask) if m]
    asrs_nz = [a for a, m in zip(asrs_sorted, nonzero_mask) if m]

    if len(deltas_nz) >= 3:
        rho, pval = spearmanr(deltas_nz, asrs_nz)
        mono_ok = rho < -0.7  # strong negative correlation
        print(f"  Spearman ρ (delta vs ASR): {rho:.4f} (p={pval:.4e})")
        print(f"  Monotonic decreasing:      {'PASS' if mono_ok else 'FAIL'} "
              f"(threshold: ρ < -0.7)")
    else:
        mono_ok = False
        print(f"  Monotonic decreasing:      FAIL (insufficient data points)")

    # 4. Curve fit quality
    if popt_lora is not None:
        fit_ok = r2_lora > 0.8
        print(f"  Curve fit R²:              {r2_lora:.4f} "
              f"({'PASS' if fit_ok else 'FAIL'}, threshold: > 0.8)")
    else:
        fit_ok = False
        print(f"  Curve fit R²:              FAIL (fitting failed)")

    # 5. Phase transition exists
    if popt_lora is not None:
        delta_crit = find_phase_transition(popt_lora)
        pt_ok = delta_crit is not None
        if pt_ok:
            print(f"  Phase transition at δ:     {delta_crit:.4f} PASS")
        else:
            print(f"  Phase transition:          FAIL (not found)")
    else:
        pt_ok = False
        print(f"  Phase transition:          FAIL (no fit)")

    # 6. LoRA vs ICL comparison
    if popt_lora is not None and popt_icl is not None:
        # LoRA should have steeper decay (larger b parameter)
        lora_steeper = popt_lora[1] > popt_icl[1]
        print(f"  LoRA steeper decay:        {'PASS' if lora_steeper else 'FAIL'} "
              f"(LoRA b={popt_lora[1]:.2f}, ICL b={popt_icl[1]:.2f})")
    else:
        lora_steeper = True  # Skip check
        print(f"  LoRA vs ICL comparison:    SKIP (insufficient fits)")

    passed = fig_ok and res_ok and mono_ok and fit_ok
    if passed:
        print(f"\n  OVERALL: PASS — delta-ASR curve verified")
    else:
        print(f"\n  OVERALL: FAIL — Check above for details")

    print("\nDone!")
    return passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
