#!/usr/bin/env python3
"""RoPE frequency spectrum analysis — Phase 2.

Computes the RoPE frequency spectrum for Llama-3.1-8B-Instruct, categorizes
each 2D subspace by wavelength, and identifies candidate subspaces for
repurposing (type embedding).

Usage:
    python analysis/rope_analysis.py                    # full analysis + Figure 4
    python analysis/rope_analysis.py --from-config      # use config yaml (no model load)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "llama8b.yaml")


def load_rope_params_from_config(config_path=CONFIG_PATH):
    """Load RoPE parameters from config YAML (no model needed)."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {
        "head_dim": cfg["head_dim"],           # 128
        "rope_theta": cfg["rope_theta"],       # 500000.0
        "max_position_embeddings": cfg["max_position_embeddings"],  # 131072
        "num_hidden_layers": cfg["num_hidden_layers"],  # 32
    }


def load_rope_params_from_model(model):
    """Extract RoPE parameters from loaded model."""
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    return {
        "head_dim": head_dim,
        "rope_theta": model.config.rope_theta,
        "max_position_embeddings": model.config.max_position_embeddings,
        "num_hidden_layers": model.config.num_hidden_layers,
    }


def analyze_rope_frequencies(head_dim, rope_theta, max_position_embeddings):
    """Compute RoPE frequency spectrum and categorize subspaces.

    For Llama-3.1-8B with head_dim=128:
      - 64 subspaces (pairs of dimensions)
      - freq_i = 1 / (theta^(2i/d)) for i in [0, d/2)
      - wavelength_i = 2*pi / freq_i (in token positions)

    Categories:
      - HIGH_FREQ: wavelength < 4 tokens (noise, oscillates too rapidly)
      - USEFUL: 4 <= wavelength <= max_ctx (meaningful position encoding)
      - LOW_FREQ: wavelength > max_ctx (near-constant, little positional info)

    Returns:
        results: list of dicts with per-subspace info
        summary: dict with category counts and candidate subspaces
    """
    num_subspaces = head_dim // 2

    # Compute frequencies: freq_i = 1 / (theta^(2i/d))
    indices = np.arange(0, head_dim, 2, dtype=np.float64)
    freqs = 1.0 / (rope_theta ** (indices / head_dim))
    wavelengths = 2 * np.pi / freqs  # in token positions
    rotations_in_ctx = max_position_embeddings * freqs / (2 * np.pi)

    results = []
    high_freq = []
    useful = []
    low_freq = []

    print(f"\nRoPE Frequency Spectrum (head_dim={head_dim}, theta={rope_theta}, "
          f"max_pos={max_position_embeddings})")
    print(f"{'Subspace':>8} {'Freq':>12} {'Wavelength':>14} {'Rotations':>12} {'Category':>12}")
    print("-" * 70)

    for i in range(num_subspaces):
        f = freqs[i]
        w = wavelengths[i]
        r = rotations_in_ctx[i]

        if w < 4:
            category = "HIGH_FREQ"
            high_freq.append(i)
        elif w > max_position_embeddings:
            category = "LOW_FREQ"
            low_freq.append(i)
        else:
            category = "USEFUL"
            useful.append(i)

        results.append({
            "subspace": i,
            "dim_pair": [int(2 * i), int(2 * i + 1)],
            "frequency": float(f),
            "wavelength": float(w),
            "rotations_in_context": float(r),
            "category": category,
        })

        print(f"{i:>8d} {f:>12.8f} {w:>14.1f} {r:>12.1f} {category:>12}")

    summary = {
        "num_subspaces": num_subspaces,
        "high_freq_count": len(high_freq),
        "useful_count": len(useful),
        "low_freq_count": len(low_freq),
        "high_freq_indices": high_freq,
        "useful_indices": useful,
        "low_freq_indices": low_freq,
        "candidate_subspaces": low_freq + high_freq[:5],  # low-freq are best candidates
    }

    print(f"\n=== Summary ===")
    print(f"  HIGH_FREQ (wavelength < 4): {len(high_freq)} subspaces — {high_freq}")
    print(f"  USEFUL (4 <= wavelength <= {max_position_embeddings}): {len(useful)} subspaces")
    print(f"  LOW_FREQ (wavelength > {max_position_embeddings}): {len(low_freq)} subspaces — {low_freq}")
    print(f"\n  Candidate subspaces for repurposing: {summary['candidate_subspaces']}")

    return results, summary


# ---------------------------------------------------------------------------
# Figure 4: Frequency spectrum bar + line chart
# ---------------------------------------------------------------------------
def generate_figure4(results, summary, max_position_embeddings, output_dir=OUTPUT_DIR):
    """Generate Figure 4: RoPE frequency spectrum with candidate highlights.

    Combined bar + line chart:
    - X-axis: subspace index
    - Left Y-axis (bars): frequency value
    - Right Y-axis (line): wavelength (log scale)
    - Red bands: candidate subspaces
    - Dashed lines: wavelength = 4 and wavelength = max_ctx
    """
    os.makedirs(output_dir, exist_ok=True)

    subspaces = [r["subspace"] for r in results]
    freqs = [r["frequency"] for r in results]
    wavelengths = [r["wavelength"] for r in results]
    categories = [r["category"] for r in results]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Color bars by category
    bar_colors = []
    for cat in categories:
        if cat == "HIGH_FREQ":
            bar_colors.append("#F44336")  # red
        elif cat == "LOW_FREQ":
            bar_colors.append("#FF9800")  # orange
        else:
            bar_colors.append("#4CAF50")  # green

    # Bar chart: frequency on left y-axis
    ax1.bar(subspaces, freqs, color=bar_colors, alpha=0.7, width=0.8)
    ax1.set_xlabel("Subspace Index (2D pair)", fontsize=12)
    ax1.set_ylabel("Frequency (rad/token)", fontsize=12, color="#333333")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="#333333")

    # Line chart: wavelength on right y-axis (log scale)
    ax2 = ax1.twinx()
    ax2.plot(subspaces, wavelengths, color="#2196F3", linewidth=2,
             marker=".", markersize=3, label="Wavelength")
    ax2.set_ylabel("Wavelength (tokens, log scale)", fontsize=12, color="#2196F3")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="#2196F3")

    # Reference lines
    ax2.axhline(y=4, color="red", linestyle="--", alpha=0.6,
                label="Wavelength = 4 (high-freq boundary)")
    ax2.axhline(y=max_position_embeddings, color="orange", linestyle="--", alpha=0.6,
                label=f"Wavelength = {max_position_embeddings} (low-freq boundary)")

    # Highlight candidate subspaces
    candidates = summary.get("candidate_subspaces", [])
    for idx in candidates:
        ax1.axvspan(idx - 0.4, idx + 0.4, alpha=0.15, color="red",
                    zorder=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", alpha=0.7, label="USEFUL"),
        Patch(facecolor="#F44336", alpha=0.7, label="HIGH_FREQ"),
        Patch(facecolor="#FF9800", alpha=0.7, label="LOW_FREQ"),
        Patch(facecolor="red", alpha=0.15, label="Candidate for repurposing"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax2.legend(loc="center right", fontsize=9)

    # Use model name from config if available in summary, else generic title
    model_label = summary.get("model_name", "Model")
    ax1.set_title(f"RoPE Frequency Spectrum — {model_label}", fontsize=14)
    ax1.set_xlim(-1, len(subspaces))

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig4_rope_spectrum.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 4: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="RoPE frequency spectrum analysis")
    parser.add_argument("--from-config", action="store_true",
                        help="Load params from config YAML (no model needed)")
    parser.add_argument("--config", default=CONFIG_PATH,
                        help="Path to model config YAML")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for figures")
    args = parser.parse_args()

    # Load parameters
    if args.from_config:
        print("Loading RoPE parameters from config...")
        params = load_rope_params_from_config(args.config)
    else:
        print("Loading model to extract RoPE parameters...")
        try:
            from utils import load_model
            model, _ = load_model()
            params = load_rope_params_from_model(model)
            del model
        except Exception as e:
            print(f"Model loading failed ({e}), falling back to config...")
            params = load_rope_params_from_config(args.config)

    print(f"Parameters: head_dim={params['head_dim']}, "
          f"theta={params['rope_theta']}, "
          f"max_pos={params['max_position_embeddings']}")

    # Analyze frequencies
    results, summary = analyze_rope_frequencies(
        head_dim=params["head_dim"],
        rope_theta=params["rope_theta"],
        max_position_embeddings=params["max_position_embeddings"],
    )

    # Attach model name to summary for figure title
    if args.from_config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        summary["model_name"] = cfg.get("model_name", "Model")

    # Generate Figure 4
    generate_figure4(results, summary, params["max_position_embeddings"],
                     args.output_dir)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "rope_spectrum.json")
    with open(results_path, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print(f"Saved spectrum data: {results_path}")

    # Verification
    print("\n=== Verification ===")
    fig_path = os.path.join(args.output_dir, "fig4_rope_spectrum.png")
    if os.path.exists(fig_path):
        size_kb = os.path.getsize(fig_path) / 1024
        print(f"  OK: fig4_rope_spectrum.png ({size_kb:.1f} KB)")
    else:
        print(f"  MISSING: fig4_rope_spectrum.png")

    if summary["candidate_subspaces"]:
        print(f"  OK: {len(summary['candidate_subspaces'])} candidate subspaces identified")
        print(f"  Candidates: {summary['candidate_subspaces']}")
    else:
        print("  WARNING: No candidate subspaces found")

    print("\nDone!")


if __name__ == "__main__":
    main()
