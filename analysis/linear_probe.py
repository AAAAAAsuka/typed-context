#!/usr/bin/env python3
"""Train linear probes on hidden states and generate visualizations.

Loads .npz hidden state files, trains LogisticRegression per layer for each
dataset, and generates:
  - Figure 1: layer-wise probe accuracy curves with 95% CI
  - Figure 2: t-SNE at early/mid/late layers
  - Figure 3: top-20 probe weight dimensions bar chart

Usage:
    python analysis/linear_probe.py                       # all analysis
    python analysis/linear_probe.py --probe-only          # just accuracy curves
    python analysis/linear_probe.py --tsne-only           # just t-SNE
    python analysis/linear_probe.py --weights-only        # just probe weights
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIDDEN_DIR = os.path.join(PROJECT_ROOT, "outputs", "hidden_states")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def load_hidden_states(dataset_name, hidden_dir=HIDDEN_DIR):
    """Load hidden states and source labels from .npz file.

    Returns:
        hidden_states: dict[layer_idx] -> np.array (num_tokens, hidden_dim)
        source_labels: np.array (num_tokens,)
    """
    path = os.path.join(hidden_dir, f"{dataset_name}.npz")
    data = np.load(path)

    source_labels = data["source_labels"]
    hidden_states = {}
    for key in data.keys():
        if key.startswith("hidden_layer_"):
            layer_idx = int(key.split("_")[-1])
            hidden_states[layer_idx] = data[key]

    return hidden_states, source_labels


def probe_accuracy(hidden_dict, labels, n_folds=5, seed=42):
    """Train logistic regression probe at each layer, return accuracy + CI.

    Uses 5-fold cross-validation to get mean accuracy and 95% CI.

    Returns:
        results: dict[layer_idx] -> {
            'mean': float, 'std': float,
            'ci_low': float, 'ci_high': float,
            'fold_scores': list[float]
        }
    """
    results = {}
    for layer_idx in sorted(hidden_dict.keys()):
        X = hidden_dict[layer_idx]
        y = labels

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring="accuracy")

        mean_acc = scores.mean()
        std_acc = scores.std()
        ci_half = 1.96 * std_acc / np.sqrt(n_folds)

        results[layer_idx] = {
            "mean": float(mean_acc),
            "std": float(std_acc),
            "ci_low": float(mean_acc - ci_half),
            "ci_high": float(mean_acc + ci_half),
            "fold_scores": scores.tolist(),
        }

        print(f"  Layer {layer_idx:2d}: acc={mean_acc:.4f} ± {std_acc:.4f} "
              f"(95% CI: [{mean_acc - ci_half:.4f}, {mean_acc + ci_half:.4f}])")

    return results


def get_probe_weights(hidden_dict, labels, seed=42):
    """Train logistic regression and return model weights per layer.

    Returns:
        weights: dict[layer_idx] -> np.array (hidden_dim,) — absolute coef values
    """
    weights = {}
    for layer_idx in sorted(hidden_dict.keys()):
        X = hidden_dict[layer_idx]
        y = labels

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X_scaled, y)

        # coef_ has shape (1, hidden_dim) for binary classification
        weights[layer_idx] = np.abs(clf.coef_[0])

    return weights


# ---------------------------------------------------------------------------
# Figure 1: Layer-wise probe accuracy curves
# ---------------------------------------------------------------------------
def generate_figure1(results_by_dataset, output_dir=OUTPUT_DIR):
    """Generate layer-wise probe accuracy curves with 95% CI.

    Args:
        results_by_dataset: dict[dataset_name] -> dict[layer_idx] -> {mean, ci_low, ci_high}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"normal": "#2196F3", "pi_success": "#F44336", "pi_fail": "#4CAF50"}
    labels = {"normal": "Normal", "pi_success": "PI Success", "pi_fail": "PI Fail"}

    for ds_name in ["normal", "pi_success", "pi_fail"]:
        if ds_name not in results_by_dataset:
            continue
        results = results_by_dataset[ds_name]
        layers = sorted(results.keys())
        means = [results[l]["mean"] for l in layers]
        ci_lows = [results[l]["ci_low"] for l in layers]
        ci_highs = [results[l]["ci_high"] for l in layers]

        color = colors.get(ds_name, "#999999")
        label = labels.get(ds_name, ds_name)

        ax.plot(layers, means, marker="o", markersize=3, color=color,
                label=label, linewidth=2)
        ax.fill_between(layers, ci_lows, ci_highs, alpha=0.2, color=color)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Linear Probe Accuracy", fontsize=12)
    ax.set_title("Layer-wise Source Type Probe Accuracy (5-fold CV)", fontsize=14)
    ax.set_ylim(0.45, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "fig1_probe_accuracy.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure 1: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: t-SNE visualizations at early/mid/late layers
# ---------------------------------------------------------------------------
def generate_figure2(hidden_by_dataset, labels_by_dataset, layers_to_plot=None,
                     max_tokens_per_dataset=500, output_dir=OUTPUT_DIR, seed=42):
    """Generate t-SNE plots at early, mid, and late layers.

    Colors by source type (system=blue, user=red).
    Shapes by condition (normal=circle, pi_success=triangle, pi_fail=square).
    """
    if layers_to_plot is None:
        # Determine early/mid/late from available layers
        all_layers = set()
        for hs_dict in hidden_by_dataset.values():
            all_layers.update(hs_dict.keys())
        all_layers = sorted(all_layers)
        if len(all_layers) >= 3:
            layers_to_plot = [all_layers[1], all_layers[len(all_layers) // 2],
                              all_layers[-2]]
        else:
            layers_to_plot = all_layers

    layer_names = {layers_to_plot[0]: "early", layers_to_plot[1]: "mid",
                   layers_to_plot[2]: "late"} if len(layers_to_plot) >= 3 else {}

    markers = {"normal": "o", "pi_success": "^", "pi_fail": "s"}
    source_colors = {0: "#2196F3", 1: "#F44336"}  # system=blue, user=red
    source_labels = {0: "System", 1: "User"}

    for layer_idx in layers_to_plot:
        fig, ax = plt.subplots(figsize=(8, 8))

        # Collect tokens from all datasets for this layer
        all_X = []
        all_source = []
        all_dataset = []

        for ds_name in ["normal", "pi_success", "pi_fail"]:
            if ds_name not in hidden_by_dataset:
                continue
            hs_dict = hidden_by_dataset[ds_name]
            if layer_idx not in hs_dict:
                continue

            X = hs_dict[layer_idx]
            labels = labels_by_dataset[ds_name]

            # Subsample for speed
            n = min(max_tokens_per_dataset, X.shape[0])
            rng = np.random.RandomState(seed)
            idx = rng.choice(X.shape[0], n, replace=False)

            all_X.append(X[idx])
            all_source.append(labels[idx])
            all_dataset.extend([ds_name] * n)

        if not all_X:
            continue

        X_combined = np.concatenate(all_X, axis=0)
        source_combined = np.concatenate(all_source, axis=0)
        dataset_combined = np.array(all_dataset)

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=seed, perplexity=30,
                     max_iter=1000)
        X_2d = tsne.fit_transform(X_combined)

        # Plot by dataset and source type
        for ds_name in ["normal", "pi_success", "pi_fail"]:
            for src_val in [0, 1]:
                mask = (dataset_combined == ds_name) & (source_combined == src_val)
                if not mask.any():
                    continue
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=source_colors[src_val],
                    marker=markers.get(ds_name, "o"),
                    s=15, alpha=0.5,
                    label=f"{ds_name} ({source_labels[src_val]})"
                )

        layer_label = layer_names.get(layer_idx, f"layer_{layer_idx}")
        ax.set_title(f"t-SNE of Hidden States — Layer {layer_idx} ({layer_label})",
                     fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=8, markerscale=1.5, loc="best")
        ax.grid(True, alpha=0.2)

        out_path = os.path.join(output_dir, f"fig2_tsne_{layer_label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved Figure 2 ({layer_label}): {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Top-20 probe weight dimensions
# ---------------------------------------------------------------------------
def generate_figure3(weights_by_dataset, num_top=20, output_dir=OUTPUT_DIR):
    """Generate top-20 probe weight dimensions bar chart per layer.

    Shows which hidden dimensions are most important for source classification.
    """
    # Use normal dataset weights as primary
    ds_name = "normal"
    if ds_name not in weights_by_dataset:
        ds_name = next(iter(weights_by_dataset.keys()))

    weights = weights_by_dataset[ds_name]
    layers = sorted(weights.keys())

    # Select representative layers (early, mid, late)
    if len(layers) >= 3:
        selected = [layers[0], layers[len(layers) // 2], layers[-1]]
    else:
        selected = layers

    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, selected):
        w = weights[layer_idx]
        top_indices = np.argsort(w)[-num_top:][::-1]
        top_values = w[top_indices]

        ax.bar(range(num_top), top_values, color="#FF9800", alpha=0.8)
        ax.set_xticks(range(num_top))
        ax.set_xticklabels([str(i) for i in top_indices], rotation=45, fontsize=8)
        ax.set_xlabel("Hidden Dimension Index")
        ax.set_ylabel("|Probe Weight|")
        ax.set_title(f"Top-{num_top} Probe Weight Dimensions — Layer {layer_idx}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Source Type Probe: Most Important Dimensions", fontsize=14, y=1.01)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig3_probe_weights.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure 3: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Linear probing of hidden states")
    parser.add_argument("--hidden-dir", default=HIDDEN_DIR,
                        help="Directory with .npz hidden state files")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for figures")
    parser.add_argument("--probe-only", action="store_true",
                        help="Only run probing (Figure 1)")
    parser.add_argument("--tsne-only", action="store_true",
                        help="Only run t-SNE (Figure 2)")
    parser.add_argument("--weights-only", action="store_true",
                        help="Only run probe weights (Figure 3)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all = not (args.probe_only or args.tsne_only or args.weights_only)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all available datasets
    datasets_to_load = ["normal", "pi_success", "pi_fail", "swapped"]
    hidden_by_dataset = {}
    labels_by_dataset = {}

    for ds_name in datasets_to_load:
        try:
            hs, labels = load_hidden_states(ds_name, args.hidden_dir)
            hidden_by_dataset[ds_name] = hs
            labels_by_dataset[ds_name] = labels
            print(f"Loaded {ds_name}: {labels.shape[0]} tokens, "
                  f"{len(hs)} layers")
        except FileNotFoundError:
            print(f"Warning: {ds_name}.npz not found, skipping")

    if not hidden_by_dataset:
        print("ERROR: No hidden state files found. Run extract_hidden_states.py first.")
        sys.exit(1)

    # ---- Probe accuracy ----
    if run_all or args.probe_only:
        print("\n=== Probe Accuracy (5-fold CV) ===")
        results_by_dataset = {}
        for ds_name in ["normal", "pi_success", "pi_fail", "swapped"]:
            if ds_name not in hidden_by_dataset:
                continue
            print(f"\nDataset: {ds_name}")
            results = probe_accuracy(
                hidden_by_dataset[ds_name], labels_by_dataset[ds_name],
                n_folds=args.n_folds, seed=args.seed
            )
            results_by_dataset[ds_name] = results

        # Figure 1: accuracy curves (normal, pi_success, pi_fail only)
        print("\nGenerating Figure 1...")
        generate_figure1(results_by_dataset, args.output_dir)

        # Report swapped separately
        if "swapped" in results_by_dataset:
            print("\n--- Confound Control: Swapped Dataset ---")
            swapped = results_by_dataset["swapped"]
            for layer_idx in sorted(swapped.keys()):
                r = swapped[layer_idx]
                print(f"  Layer {layer_idx:2d}: acc={r['mean']:.4f} ± {r['std']:.4f}")

        # Save all results as JSON
        results_path = os.path.join(args.output_dir, "probe_results.json")
        with open(results_path, "w") as f:
            json.dump(results_by_dataset, f, indent=2)
        print(f"\nSaved probe results: {results_path}")

    # ---- t-SNE ----
    if run_all or args.tsne_only:
        print("\n=== t-SNE Visualization ===")
        generate_figure2(
            hidden_by_dataset, labels_by_dataset,
            output_dir=args.output_dir, seed=args.seed
        )

    # ---- Probe weights ----
    if run_all or args.weights_only:
        print("\n=== Probe Weight Analysis ===")
        weights_by_dataset = {}
        for ds_name in ["normal", "pi_success", "pi_fail"]:
            if ds_name not in hidden_by_dataset:
                continue
            print(f"Computing probe weights for {ds_name}...")
            weights = get_probe_weights(
                hidden_by_dataset[ds_name], labels_by_dataset[ds_name],
                seed=args.seed
            )
            weights_by_dataset[ds_name] = weights

        print("Generating Figure 3...")
        generate_figure3(weights_by_dataset, output_dir=args.output_dir)

    # ---- Verification ----
    print("\n=== Verification ===")
    expected_files = ["fig1_probe_accuracy.png"]
    if run_all or args.tsne_only:
        expected_files.extend(["fig2_tsne_early.png", "fig2_tsne_mid.png",
                               "fig2_tsne_late.png"])
    if run_all or args.weights_only:
        expected_files.append("fig3_probe_weights.png")

    all_ok = True
    for fname in expected_files:
        path = os.path.join(args.output_dir, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  OK: {fname} ({size_kb:.1f} KB)")
        else:
            print(f"  MISSING: {fname}")
            all_ok = False

    # Sanity check: probe accuracy on normal data > 0.7 at early layers
    if (run_all or args.probe_only) and "normal" in results_by_dataset:
        early_layers = sorted(results_by_dataset["normal"].keys())[:3]
        for l in early_layers:
            acc = results_by_dataset["normal"][l]["mean"]
            status = "PASS" if acc > 0.7 else "CHECK"
            print(f"  Sanity: normal layer {l} acc={acc:.4f} [{status}]")

    if all_ok:
        print("\nAll figures generated successfully!")
    else:
        print("\nSome files missing - check output above.")

    print("\nDone!")


if __name__ == "__main__":
    main()
