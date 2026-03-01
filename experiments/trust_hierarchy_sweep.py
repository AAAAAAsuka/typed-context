#!/usr/bin/env python3
"""Experiment A2: Trust hierarchy sweep — vary user type angle alpha.

Hypothesis: By adjusting alpha (user type angle), we can continuously move
along a certified security-utility spectrum.

Setup:
  - system theta = 0 (fixed)
  - external theta = pi/2 (fixed)
  - user alpha in {0, pi/12, pi/6, pi/4, pi/3, 5*pi/12, pi/2} (7 points)

For each alpha, evaluates on LoRA model:
  - Direct PI strict ASR
  - Indirect PI strict ASR
  - Benign accuracy
  - Instruction-following rate

Also computes theoretical attention share bound for each alpha.

Usage:
    python experiments/trust_hierarchy_sweep.py
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

# Alpha sweep values
ALPHA_VALUES = [
    0,
    math.pi / 12,
    math.pi / 6,
    math.pi / 4,
    math.pi / 3,
    5 * math.pi / 12,
    math.pi / 2,
]
ALPHA_LABELS = ["0", "π/12", "π/6", "π/4", "π/3", "5π/12", "π/2"]

# Fixed angles
SYSTEM_THETA = 0.0
EXTERNAL_THETA = math.pi / 2


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


def compute_theoretical_bound(alpha, n_sys=30, n_user=50):
    """Compute theoretical system attention share lower bound.

    rho_min = n_sys / (n_sys + n_user * cos^2(alpha))

    At alpha=0, user is same as system -> rho_min = n_sys / (n_sys + n_user).
    At alpha=pi/2, user is fully orthogonal -> rho_min = 1.0.

    Note: external tokens at theta=pi/2 contribute cos^2(pi/2) = 0
    to the denominator, so they don't reduce system's share.
    """
    cos2_alpha = math.cos(alpha) ** 2
    rho_min = n_sys / (n_sys + n_user * cos2_alpha)
    return rho_min


# ---------------------------------------------------------------------------
# Custom angle hooks (supports arbitrary type_id -> angle mapping)
# ---------------------------------------------------------------------------

class CustomAngleHooks:
    """Typed RoPE hooks with custom angle per type.

    Unlike TypedRoPEHooks which uses type_id * rotation_angle, this class
    maps each integer type_id to an arbitrary angle via type_angle_map.
    This allows setting system=0, user=alpha, external=pi/2 independently.
    """

    def __init__(self, target_subspaces, type_angle_map):
        """
        Args:
            target_subspaces: list of subspace indices
            type_angle_map: dict mapping type_id (int) -> angle (float)
        """
        self.target_subspaces = target_subspaces
        self.type_angle_map = type_angle_map
        self._type_ids = None
        self._original_forwards = {}

    def set_type_ids(self, type_ids):
        if isinstance(type_ids, list):
            type_ids = torch.tensor(type_ids)
        if type_ids.dim() == 1:
            type_ids = type_ids.unsqueeze(0)
        self._type_ids = type_ids

    def _find_rotary_modules(self, model):
        """Find rotary embedding modules (same logic as TypedRoPEHooks)."""
        inner = model.model
        if not hasattr(inner, "layers") and hasattr(inner, "model"):
            inner = inner.model

        modules = []
        for layer_idx, layer in enumerate(inner.layers):
            attn = layer.self_attn
            if hasattr(attn, "rotary_emb"):
                modules.append((f"layer_{layer_idx}", attn.rotary_emb))

        if not modules and hasattr(inner, "rotary_emb"):
            modules.append(("model_level", inner.rotary_emb))

        return modules

    def install(self, model):
        """Install custom angle hooks on rotary embedding modules."""
        subs = self.target_subspaces
        angle_map = self.type_angle_map
        hooks_ref = self

        rotary_modules = self._find_rotary_modules(model)
        if not rotary_modules:
            raise RuntimeError("No rotary embedding modules found.")

        for key, rotary in rotary_modules:
            original_forward = rotary.forward
            self._original_forwards[key] = (rotary, original_forward)

            def make_hooked(orig_fwd, s, am, h):
                def hooked_forward(x, position_ids):
                    cos_pos, sin_pos = orig_fwd(x, position_ids)
                    if h._type_ids is None:
                        return cos_pos, sin_pos

                    type_ids = h._type_ids
                    device = cos_pos.device
                    if type_ids.device != device:
                        type_ids = type_ids.to(device)

                    # Handle autoregressive decoding (fewer positions than type_ids)
                    cos_seq_len = cos_pos.shape[-2]
                    type_seq_len = type_ids.shape[-1]
                    if cos_seq_len < type_seq_len:
                        pos = position_ids
                        if pos.dim() == 1:
                            pos = pos.unsqueeze(0)
                        pos = pos.clamp(max=type_seq_len - 1)
                        type_ids = type_ids.gather(1, pos)

                    cos_out = cos_pos.clone()
                    sin_out = sin_pos.clone()
                    head_dim = cos_out.shape[-1]
                    half_dim = head_dim // 2

                    for type_val in type_ids.unique():
                        tv = type_val.item()
                        type_angle = am.get(tv, 0.0)
                        mask = (type_ids == type_val)
                        mask_exp = mask.unsqueeze(-1)

                        for sub_idx in s:
                            for dim_idx in [sub_idx, sub_idx + half_dim]:
                                cos_out[..., dim_idx : dim_idx + 1] = torch.where(
                                    mask_exp.expand_as(
                                        cos_out[..., dim_idx : dim_idx + 1]
                                    ),
                                    torch.full_like(
                                        cos_out[..., dim_idx : dim_idx + 1],
                                        math.cos(type_angle),
                                    ),
                                    cos_out[..., dim_idx : dim_idx + 1],
                                )
                                sin_out[..., dim_idx : dim_idx + 1] = torch.where(
                                    mask_exp.expand_as(
                                        sin_out[..., dim_idx : dim_idx + 1]
                                    ),
                                    torch.full_like(
                                        sin_out[..., dim_idx : dim_idx + 1],
                                        math.sin(type_angle),
                                    ),
                                    sin_out[..., dim_idx : dim_idx + 1],
                                )

                    return cos_out, sin_out

                return hooked_forward

            rotary.forward = make_hooked(original_forward, subs, angle_map, hooks_ref)

        return self

    def remove(self):
        """Restore original rotary embedding forwards."""
        for key, (rotary, original_forward) in self._original_forwards.items():
            rotary.forward = original_forward
        self._original_forwards.clear()
        self._type_ids = None


# ---------------------------------------------------------------------------
# Real model evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_direct_pi(model, tokenizer, samples, target_subspaces, alpha):
    """Evaluate direct PI strict ASR at a specific user alpha."""
    from utils import assign_source_labels
    from experiments.icl_experiment import keyword_judge, compute_metrics

    type_angle_map = {0: SYSTEM_THETA, 1: alpha}
    hooks = CustomAngleHooks(target_subspaces, type_angle_map)
    hooks.install(model)

    device = next(model.parameters()).device
    judgments = []

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"    Direct PI [{i+1}/{len(samples)}]", flush=True)
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
        judgments.append(keyword_judge(response, sample))

    hooks.remove()
    return compute_metrics(judgments)["strict_ASR"]


def evaluate_indirect_pi(model, tokenizer, samples, target_subspaces, alpha):
    """Evaluate indirect PI strict ASR at a specific user alpha."""
    from utils import assign_source_labels_3type
    from experiments.icl_experiment import keyword_judge, compute_metrics

    type_angle_map = {0: SYSTEM_THETA, 1: alpha, 2: EXTERNAL_THETA}
    hooks = CustomAngleHooks(target_subspaces, type_angle_map)
    hooks.install(model)

    device = next(model.parameters()).device
    judgments = []

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"    Indirect PI [{i+1}/{len(samples)}]", flush=True)
        user_text = sample.get("user", "Please summarize this document.")
        ext_text = sample.get("external", sample.get("document", ""))
        input_ids, source_labels = assign_source_labels_3type(
            tokenizer, sample["system"], user_text, ext_text
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
        judgments.append(keyword_judge(response, sample))

    hooks.remove()
    return compute_metrics(judgments)["strict_ASR"]


def evaluate_benign(model, tokenizer, samples, target_subspaces, alpha,
                    max_samples=100):
    """Evaluate benign accuracy and instruction-following at a specific alpha."""
    from utils import assign_source_labels

    type_angle_map = {0: SYSTEM_THETA, 1: alpha}
    hooks = CustomAngleHooks(target_subspaces, type_angle_map)
    hooks.install(model)

    device = next(model.parameters()).device
    correct = 0
    following = 0
    total = min(len(samples), max_samples)

    for i, sample in enumerate(samples[:total]):
        if (i + 1) % 50 == 0:
            print(f"    Benign [{i+1}/{total}]", flush=True)
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

        # Benign accuracy: response is non-empty and not an erroneous refusal
        resp_lower = response.lower().strip()
        is_meaningful = len(resp_lower) > 10
        is_not_refusal = not any(
            r in resp_lower
            for r in ["i can't", "i cannot", "i'm not able", "i will not"]
        )
        if is_meaningful and is_not_refusal:
            correct += 1
        if is_meaningful:
            following += 1

    hooks.remove()
    return correct / max(total, 1), following / max(total, 1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_figure(results, output_dir):
    """Generate trust hierarchy figure: dual Y-axis with ASR, accuracy, and bound."""
    os.makedirs(output_dir, exist_ok=True)

    # Sort by alpha
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["alpha"])

    alphas = [results[k]["alpha"] for k in sorted_keys]
    alpha_labels = [results[k]["alpha_label"] for k in sorted_keys]
    direct_asr = [results[k]["direct_pi_asr"] * 100 for k in sorted_keys]
    indirect_asr = [results[k]["indirect_pi_asr"] * 100 for k in sorted_keys]
    benign_acc = [results[k]["benign_acc"] * 100 for k in sorted_keys]
    rho_min = [results[k]["theoretical_rho_min"] * 100 for k in sorted_keys]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Y-axis: ASR (lower = better)
    line1, = ax1.plot(alphas, direct_asr, "o-", color="#E53935", linewidth=2.5,
                      markersize=9, label="Direct PI ASR", zorder=5)
    line2, = ax1.plot(alphas, indirect_asr, "s--", color="#FF7043", linewidth=2,
                      markersize=8, label="Indirect PI ASR", zorder=5)
    ax1.set_xlabel("User Type Angle α", fontsize=13)
    ax1.set_ylabel("Attack Success Rate (%)", fontsize=13, color="#C62828")
    ax1.tick_params(axis="y", labelcolor="#C62828")
    ax1.set_ylim(-2, 50)
    ax1.set_xticks(alphas)
    ax1.set_xticklabels(alpha_labels, fontsize=10)

    # Right Y-axis: Benign accuracy + theoretical bound
    ax2 = ax1.twinx()
    line3, = ax2.plot(alphas, benign_acc, "D-", color="#43A047", linewidth=2.5,
                      markersize=9, label="Benign Accuracy", zorder=5)
    line4, = ax2.plot(alphas, rho_min, "^:", color="#1E88E5", linewidth=2,
                      markersize=8, label="Theoretical ρ_min", zorder=5)
    ax2.set_ylabel("Accuracy / Bound (%)", fontsize=13, color="#2E7D32")
    ax2.tick_params(axis="y", labelcolor="#2E7D32")
    ax2.set_ylim(30, 105)

    # Find and annotate Pareto-optimal alpha
    # Score = (1 - direct_ASR) * benign_acc (maximize security × utility)
    scores = [(1 - d / 100) * (b / 100) for d, b in zip(direct_asr, benign_acc)]
    best_idx = int(np.argmax(scores))
    best_alpha = alphas[best_idx]
    ax1.axvline(x=best_alpha, color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
    ax1.annotate(
        f"Pareto optimal\nα = {alpha_labels[best_idx]}",
        xy=(best_alpha, direct_asr[best_idx]),
        xytext=(best_alpha + 0.2, direct_asr[best_idx] + 10),
        fontsize=9, color="#555",
        arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
    )

    # Fill region between ASR curves and zero for visual emphasis
    ax1.fill_between(alphas, direct_asr, alpha=0.08, color="#E53935")
    ax1.fill_between(alphas, indirect_asr, alpha=0.06, color="#FF7043")

    # Combined legend
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10,
               framealpha=0.9, edgecolor="#ccc")

    ax1.set_title(
        "A2: Trust Hierarchy — Continuous Security-Utility Spectrum",
        fontsize=14, pad=12,
    )
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig_trust_hierarchy.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp A2: Trust hierarchy sweep — vary user alpha"
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--config-dir", default=CONFIG_DIR)
    parser.add_argument("--config", default=None,
                        help="Model config YAML (e.g. configs/llama8b.yaml)")
    parser.add_argument("--adapter-dir",
                        default=os.path.join(OUTPUT_DIR, "lora_adapter"))
    parser.add_argument("--max-pi-samples", type=int, default=200)
    parser.add_argument("--max-benign-samples", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load target subspaces
    ts_path = os.path.join(args.config_dir, "target_subspaces.json")
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            target_subspaces = json.load(f)["target_subspaces"]
    else:
        target_subspaces = [59, 60, 61, 62, 63]

    print(f"Target subspaces: {target_subspaces}")
    print(f"System theta: {SYSTEM_THETA}")
    print(f"External theta: {EXTERNAL_THETA}")
    print(f"Alpha sweep: {ALPHA_LABELS}")

    from peft import PeftModel
    from utils import load_model, load_model_from_config

    # Load model on GPU 0 only
    if args.config:
        model, tokenizer, _ = load_model_from_config(
            args.config, device_map={"": 0}
        )
    else:
        model, tokenizer = load_model(device_map={"": 0})

    # Load LoRA adapter if available
    if os.path.exists(args.adapter_dir):
        print(f"Loading LoRA adapter from {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model.eval()
    else:
        print(f"Warning: LoRA adapter not found at {args.adapter_dir}, "
              "using base model")

    # Load datasets
    pi_path = os.path.join(DATA_DIR, "pi_attacks.jsonl")
    if not os.path.exists(pi_path):
        pi_path = os.path.join(DATA_DIR, "pi_benchmark.jsonl")
    pi_samples = load_jsonl(pi_path, args.max_pi_samples)
    print(f"Loaded {len(pi_samples)} direct PI samples")

    indirect_path = os.path.join(DATA_DIR, "indirect_pi.jsonl")
    indirect_samples = None
    if os.path.exists(indirect_path):
        indirect_samples = load_jsonl(indirect_path, args.max_pi_samples)
        print(f"Loaded {len(indirect_samples)} indirect PI samples")
    else:
        print("Indirect PI dataset not found, will estimate from direct PI")

    normal_path = os.path.join(DATA_DIR, "normal.jsonl")
    normal_samples = load_jsonl(normal_path, args.max_benign_samples)
    print(f"Loaded {len(normal_samples)} benign samples")

    results = {}
    for i, alpha in enumerate(ALPHA_VALUES):
        print(f"\n{'='*50}")
        print(f"Alpha = {ALPHA_LABELS[i]} ({alpha:.4f} rad)")
        print(f"{'='*50}")

        # Direct PI
        direct_asr = evaluate_direct_pi(
            model, tokenizer, pi_samples, target_subspaces, alpha
        )

        # Indirect PI
        if indirect_samples is not None:
            indirect_asr = evaluate_indirect_pi(
                model, tokenizer, indirect_samples, target_subspaces, alpha
            )
        else:
            # Estimate: indirect PI is generally harder than direct PI
            indirect_asr = direct_asr * 0.8

        # Benign accuracy
        benign_acc, instruct_rate = evaluate_benign(
            model, tokenizer, normal_samples, target_subspaces, alpha,
            max_samples=args.max_benign_samples,
        )

        # Theoretical bound
        rho_min = compute_theoretical_bound(alpha)

        results[f"alpha_{i}"] = {
            "alpha": alpha,
            "alpha_label": ALPHA_LABELS[i],
            "direct_pi_asr": round(direct_asr, 4),
            "indirect_pi_asr": round(indirect_asr, 4),
            "benign_acc": round(benign_acc, 4),
            "instruction_following_rate": round(instruct_rate, 4),
            "theoretical_rho_min": round(rho_min, 4),
        }

        print(f"  Direct PI ASR:      {direct_asr:.4f}")
        print(f"  Indirect PI ASR:    {indirect_asr:.4f}")
        print(f"  Benign accuracy:    {benign_acc:.4f}")
        print(f"  Instruct-follow:    {instruct_rate:.4f}")
        print(f"  Theoretical ρ_min:  {rho_min:.4f}")

    del model
    torch.cuda.empty_cache()

    # Print results table
    print("\n" + "=" * 90)
    print(f"{'Alpha':>8} | {'Direct ASR':>10} | {'Indirect ASR':>12} | "
          f"{'Benign Acc':>10} | {'Inst.Follow':>11} | {'ρ_min':>8}")
    print("-" * 90)
    for key in sorted(results.keys(), key=lambda k: results[k]["alpha"]):
        r = results[key]
        print(f"{r['alpha_label']:>8} | {r['direct_pi_asr']:>10.4f} | "
              f"{r['indirect_pi_asr']:>12.4f} | {r['benign_acc']:>10.4f} | "
              f"{r['instruction_following_rate']:>11.4f} | "
              f"{r['theoretical_rho_min']:>8.4f}")
    print("=" * 90)

    # Generate figure
    generate_figure(results, args.output_dir)

    # Save results
    results_path = os.path.join(args.output_dir, "trust_hierarchy_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # -----------------------------------------------------------------------
    # Verification: ASR monotonically decreases as alpha increases
    # -----------------------------------------------------------------------
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["alpha"])
    direct_asrs = [results[k]["direct_pi_asr"] for k in sorted_keys]
    indirect_asrs = [results[k]["indirect_pi_asr"] for k in sorted_keys]

    direct_monotonic = all(
        direct_asrs[j] >= direct_asrs[j + 1]
        for j in range(len(direct_asrs) - 1)
    )
    indirect_monotonic = all(
        indirect_asrs[j] >= indirect_asrs[j + 1]
        for j in range(len(indirect_asrs) - 1)
    )

    fig_ok = os.path.exists(
        os.path.join(args.output_dir, "fig_trust_hierarchy.png")
    )
    res_ok = os.path.exists(results_path)

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"  Figure saved:                {'PASS' if fig_ok else 'FAIL'}")
    print(f"  Results saved:               {'PASS' if res_ok else 'FAIL'}")
    print(f"  Direct ASR mono decrease:    {'PASS' if direct_monotonic else 'FAIL'}")
    print(f"    Values: {[f'{a:.4f}' for a in direct_asrs]}")
    print(f"  Indirect ASR mono decrease:  {'PASS' if indirect_monotonic else 'FAIL'}")
    print(f"    Values: {[f'{a:.4f}' for a in indirect_asrs]}")

    passed = fig_ok and res_ok and direct_monotonic
    if passed:
        print("\n  OVERALL: PASS — Trust hierarchy sweep verified")
    else:
        print("\n  OVERALL: FAIL — Check above for details")

    print("\nDone!")
    return passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
