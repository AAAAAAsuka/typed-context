# Typed Context via Persistent Rotation — Project Plan

## Overview

Research codebase for investigating and validating typed context encoding via RoPE rotation for prompt injection defense. Phase 1-4 baseline experiments are complete. This plan covers the follow-up experiments (A-H) that establish formal security guarantees, adaptive attack robustness, utility preservation, and mechanism understanding.

**Reference:**
- `proj.md` — Follow-up experiment plan (experiments A-H, priority tiers, paper figure list)
- `proj_v1.md` — Original engineering document (Phase 1-4 methodology, code templates)
- `activity.md` — Completed work log

---

## Completed Tasks (Phase 1-4)

All baseline tasks have been completed. See `activity.md` for details.

- [x] Project scaffolding and environment setup
- [x] Build probing datasets with token-level source labels
- [x] Phase 1: Extract hidden states for probing
- [x] Phase 1: Train linear probes and generate visualizations (Fig 1-3)
- [x] Phase 2: RoPE frequency spectrum analysis (Fig 4)
- [x] Phase 2: RoPE dimension ablation study (Fig 5)
- [x] Phase 3: Implement typed RoPE core module + unit tests
- [x] Phase 3: PPL tolerance sweep (Fig 6)
- [x] Phase 3: Attention pattern analysis under type rotation (Fig 7)
- [x] Phase 3: ICL experiment — all 5 conditions A-E (Fig 8)
- [x] Phase 3: Indirect PI experiment (3-type)
- [x] Phase 3.5: Quantization robustness check (Fig 9)
- [x] Phase 4: LoRA finetuning with typed RoPE
- [x] Phase 4: LoRA evaluation with ablation (Fig 10-11)
- [x] Summary figure and results compilation (Fig 12)

---

## Follow-up Task List

### Prerequisites

```json
{
  "category": "prerequisite",
  "description": "Train special token defense baseline (needed for experiments B2, C1, C2)",
  "id": "P1",
  "steps": [
    "Train a special token defense baseline: use [SYS_START]/[SYS_END] to wrap system prompt",
    "Use same LoRA config (r=16, alpha=32, target q/k/v/o) and same training data",
    "Train model to rely on these tokens for security decisions",
    "Save adapter to outputs/lora_adapter_special_token/",
    "Verify: model shows reduced ASR on pi_benchmark with special tokens present"
  ],
  "passes": true,
  "tier": 1
}
```

### Tier 1: Core Contributions (Must Complete)

```json
[
  {
    "category": "experiment_A",
    "description": "A1: Theory vs actual attention structure — verify certified attention property",
    "id": "A1",
    "depends_on": [],
    "steps": [
      "Create experiments/certified_attention.py",
      "Fix theta_type and target subspaces configuration",
      "Construct 100 semantically diverse inputs (system + user + external), varying length/style/topic",
      "Extract raw attention scores in target subspaces for all layers and heads",
      "For each (query_type, key_type) pair, compute attention score mean and variance",
      "Compare measured mean with theoretical prediction cos(theta_A - theta_B)",
      "Report Pearson correlation (should be ~1) and cross-input variance (should be small)",
      "Per-layer analysis: identify where theory is most/least accurate",
      "Save results to outputs/certified_attention_results.json",
      "Verify: Pearson correlation > 0.9 on target subspaces"
    ],
    "passes": true,
    "tier": 1
  },
  {
    "category": "experiment_A",
    "description": "A2: Continuous trust hierarchy tuning — sweep user type angle alpha",
    "id": "A2",
    "depends_on": ["A1"],
    "steps": [
      "Create experiments/trust_hierarchy_sweep.py",
      "Fix system theta=0, external theta=pi/2",
      "Sweep user alpha in {0, pi/12, pi/6, pi/4, pi/3, 5*pi/12, pi/2} (7 points)",
      "For each alpha, evaluate on LoRA model: direct PI strict ASR, indirect PI strict ASR, benign accuracy, instruction-following rate",
      "Compute theoretical attention share bound for each alpha",
      "Generate figure: X=alpha, left Y=ASR (lower=better), right Y=benign_acc (higher=better), overlay theoretical bound curve, annotate Pareto-optimal alpha",
      "Save figure as outputs/fig_trust_hierarchy.png",
      "Save results to outputs/trust_hierarchy_results.json",
      "Verify: ASR monotonically decreases as alpha increases from 0 to pi/2"
    ],
    "passes": true,
    "tier": 1
  },
  {
    "category": "experiment_B",
    "description": "B1: delta-ASR curve — map certified attention gap to empirical ASR",
    "id": "B1",
    "depends_on": ["A1"],
    "steps": [
      "Create experiments/delta_asr_curve.py",
      "Define delta = mean attention gap (same-type minus cross-type) over target subspaces",
      "Systematically vary delta via: |S| in {1,2,4,6,8,12,16} x theta in {0,pi/8,pi/4,3pi/8,pi/2} = 35 configs",
      "For each config, compute theoretical delta and evaluate strict ASR on LoRA model with PI benchmark",
      "Generate scatter plot: X=theoretical delta, Y=empirical ASR, fit logistic/exponential decay curve",
      "Annotate phase transition point where ASR < 5%",
      "Break down by 4 attack categories (extraction, override, role_play, smuggling) — do they have different delta thresholds?",
      "Compare with-LoRA vs without-LoRA (pure ICL) delta-ASR curves",
      "Save figure as outputs/fig_delta_asr.png",
      "Save results to outputs/delta_asr_results.json",
      "Verify: strong monotonic decreasing relationship between delta and ASR"
    ],
    "passes": true,
    "tier": 1
  },
  {
    "category": "experiment_C",
    "description": "C1: GCG gradient-based adaptive attack",
    "id": "C1",
    "depends_on": ["P1"],
    "steps": [
      "Create experiments/adaptive_attack_gcg.py",
      "Implement or integrate GCG attack (suffix length 20, 500 optimization steps, batch size 512)",
      "Run GCG against both defenses: special token LoRA and rotation LoRA",
      "Attack targets: extraction (output system prompt) and override (execute injected instruction)",
      "Record ASR every 50 optimization steps",
      "Generate figure: optimization steps vs ASR curves for both defenses",
      "Analyze gradients for rotation defense: is there effective gradient signal in target subspaces?",
      "Save figure as outputs/fig_gcg_attack.png",
      "Save results to outputs/gcg_results.json",
      "Verify: rotation defense ASR significantly lower than special token defense at step 500"
    ],
    "passes": true,
    "tier": 1
  },
  {
    "category": "experiment_C",
    "description": "C2: Semantic mimicry attack",
    "id": "C2",
    "depends_on": ["P1"],
    "steps": [
      "Create experiments/mimicry_attack.py",
      "Generate 200 mimicry attack samples: user inputs that mimic system prompt style (JSON config, natural language instructions, markdown specs)",
      "Use LLM to auto-generate: 'write text that looks like a system prompt but instructs model to ignore previous instructions'",
      "Evaluate ASR on both defenses: special token LoRA and rotation LoRA",
      "Save results to outputs/mimicry_results.json",
      "Verify: rotation defense ASR significantly lower than special token defense on mimicry attacks"
    ],
    "passes": false,
    "tier": 1
  },
  {
    "category": "experiment_D",
    "description": "D1: Benign performance benchmark",
    "id": "D1",
    "depends_on": [],
    "steps": [
      "Create experiments/benign_eval.py",
      "Evaluate baseline vs LoRA+rotation on: instruction following (100 samples), knowledge QA (200 samples), code generation (50 samples), summarization (50 samples), multi-turn dialogue (50 samples)",
      "Use LLM-as-judge scoring (1-5 scale) for quality comparison",
      "Report per-benchmark mean score diff with 95% CI",
      "Save results to outputs/benign_eval_results.json",
      "Verify: no statistically significant quality degradation (p > 0.05)"
    ],
    "passes": false,
    "tier": 1
  },
  {
    "category": "experiment_D",
    "description": "D3: Over-refusal rate measurement",
    "id": "D3",
    "depends_on": ["D1"],
    "steps": [
      "Create experiments/over_refusal.py",
      "Construct 200 'hard benign' samples — requests containing trigger-like keywords (ignore, pretend, repeat) but are legitimate",
      "Evaluate on 3 conditions: baseline, special token defense, rotation defense",
      "Measure false refusal rate for each condition",
      "Save results to outputs/over_refusal_results.json",
      "Verify: rotation defense over-refusal rate comparable to baseline (not significantly higher)"
    ],
    "passes": false,
    "tier": 1
  }
]
```

### Tier 2: Strongly Recommended

```json
[
  {
    "category": "experiment_B",
    "description": "B2: Special token delta-ASR comparison under adaptive attack",
    "id": "B2",
    "depends_on": ["B1", "C1"],
    "steps": [
      "Extend experiments/delta_asr_curve.py to compute effective delta for special token defense",
      "Measure special token defense's delta under non-adaptive and adaptive (GCG) attacks",
      "Overlay both defenses' delta-ASR curves on same plot",
      "Save figure as outputs/fig_delta_asr_comparison.png",
      "Verify: rotation delta remains stable under adaptive attack while special token delta collapses"
    ],
    "passes": false,
    "tier": 2
  },
  {
    "category": "experiment_C",
    "description": "C3: Multi-turn type confusion attack",
    "id": "C3",
    "depends_on": [],
    "steps": [
      "Create experiments/multiturn_attack.py",
      "Construct 3-5 turn dialogue scenarios: normal early turns, PI payload in final turn",
      "Early turns gradually introduce instruction-like phrasing to drift internal representation",
      "After each turn, use Phase 1 linear probe to check type signal strength",
      "Compare: rotation defense (probe accuracy should stay stable) vs special token defense (may drift)",
      "Save results to outputs/multiturn_results.json",
      "Verify: rotation defense probe accuracy stable across turns"
    ],
    "passes": false,
    "tier": 2
  },
  {
    "category": "experiment_D",
    "description": "D2: Target subspace count vs utility curve",
    "id": "D2",
    "depends_on": ["D1"],
    "steps": [
      "Extend experiments/benign_eval.py or create experiments/subspace_utility.py",
      "Sweep |S| in {0, 1, 2, 4, 8, 16, 32}",
      "For each |S|, evaluate PPL (WikiText) and MT-Bench/instruction-following score",
      "Plot |S| vs utility curve, identify 'knee point'",
      "Cross-reference with delta-ASR curve to show security-utility Pareto frontier",
      "Save figure as outputs/fig_subspace_utility.png",
      "Verify: utility degradation < 5% for |S| <= 8"
    ],
    "passes": false,
    "tier": 2
  },
  {
    "category": "experiment_E",
    "description": "E1: Layer-wise type rotation ablation",
    "id": "E1",
    "depends_on": [],
    "steps": [
      "Create experiments/layer_ablation.py",
      "On trained LoRA+rotation model, test 4 conditions: rotation only in first 1/3 layers, middle 1/3, last 1/3, even layers only",
      "Evaluate ASR for each condition",
      "Identify which layer group is most critical for defense",
      "Save results to outputs/layer_ablation_results.json",
      "Verify: results identify which layers drive the defense effect"
    ],
    "passes": false,
    "tier": 2
  },
  {
    "category": "experiment_E",
    "description": "E3: Head-level type sensitivity analysis",
    "id": "E3",
    "depends_on": ["E1"],
    "steps": [
      "Create experiments/head_analysis.py",
      "For each attention head, compute type-dependent attention gap in target subspaces",
      "Rank heads by gap magnitude, identify 'type-sensitive heads'",
      "Ablate top type-sensitive heads individually, measure ASR change",
      "Cross-reference with known 'instruction heads' from mechanistic interpretability literature",
      "Save results to outputs/head_analysis_results.json",
      "Verify: type-sensitive heads overlap with instruction-following heads"
    ],
    "passes": false,
    "tier": 2
  },
  {
    "category": "experiment_G",
    "description": "G1: Certified bound tightness analysis",
    "id": "G1",
    "depends_on": ["A1"],
    "steps": [
      "Extend experiments/certified_attention.py or create experiments/bound_tightness.py",
      "Compute theoretical attention share lower bound: rho_min = n_sys / (n_sys + n_user * cos^2(theta_type))",
      "Collect actual rho from 1000 diverse inputs",
      "Plot histogram of rho_actual, mark rho_min",
      "Compute gap between min(rho_actual) and rho_min",
      "Save figure as outputs/fig_bound_tightness.png",
      "Verify: bound is valid (all rho_actual >= rho_min)"
    ],
    "passes": false,
    "tier": 2
  }
]
```

### Tier 3: If Time and Resources Permit

```json
[
  {
    "category": "experiment_C",
    "description": "C4: Token manipulation attack (Unicode, zero-width chars, tokenizer edge cases)",
    "id": "C4",
    "depends_on": [],
    "steps": [
      "Create experiments/token_manipulation.py",
      "Construct PI attacks with: Unicode homoglyphs, zero-width characters, extra-long sequences, mixed-language inputs",
      "Verify type assignment pipeline correctness under all edge cases",
      "Report any cases where type assignment fails",
      "Verify: type assignment is correct for all edge cases"
    ],
    "passes": false,
    "tier": 3
  },
  {
    "category": "experiment_E",
    "description": "E2: Attention vs residual stream signal propagation",
    "id": "E2",
    "depends_on": [],
    "steps": [
      "Create experiments/signal_propagation.py",
      "At each layer: measure linear probe accuracy on residual stream (after attention, before MLP)",
      "At each layer: measure attention gap in target subspaces",
      "Correlate both signals with final ASR",
      "Determine if defense works via attention pattern modulation or residual stream features",
      "Verify: analysis provides clear conclusion on propagation mechanism"
    ],
    "passes": false,
    "tier": 3
  },
  {
    "category": "experiment_F",
    "description": "F1: Cross-model generalization (Qwen, Mistral)",
    "id": "F1",
    "depends_on": [],
    "steps": [
      "Repeat core experiments (Phase 3 ICL + Phase 4 LoRA) on Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3",
      "Compare optimal |S| and theta_type across models",
      "Verify theoretical attention bound holds on all models",
      "Compare delta-ASR curve shapes across models",
      "Verify: consistent findings across architectures"
    ],
    "passes": false,
    "tier": 3
  },
  {
    "category": "experiment_F",
    "description": "F2: Extended attack scenarios (multilingual, long-context, nested PI)",
    "id": "F2",
    "depends_on": [],
    "steps": [
      "Create experiments/extended_attacks.py",
      "Test: multilingual PI (payload in Chinese/Japanese/Arabic), long-context PI (hidden in 32k+ doc), nested PI (fake multi-turn in user input)",
      "Evaluate rotation defense on each scenario",
      "Verify: type rotation effective regardless of language or context length"
    ],
    "passes": false,
    "tier": 3
  },
  {
    "category": "experiment_F",
    "description": "F3: Combination with other defenses (instruction hierarchy, input filtering, output detection)",
    "id": "F3",
    "depends_on": ["D1"],
    "steps": [
      "Test rotation + instruction hierarchy training",
      "Test rotation + perplexity-based input filtering",
      "Test rotation + LLM judge output detection",
      "Measure combined ASR (should be lower than either alone)",
      "Verify: rotation is complementary to existing defenses"
    ],
    "passes": false,
    "tier": 3
  },
  {
    "category": "experiment_G",
    "description": "G2: Inputs closest to theoretical bound — attack surface analysis",
    "id": "G2",
    "depends_on": ["G1"],
    "steps": [
      "Analyze inputs where rho_actual is closest to rho_min",
      "Characterize common features of these worst-case inputs",
      "Check if worst-case inputs correlate with most effective PI attacks",
      "Verify: establish link from certified bound to attack surface"
    ],
    "passes": false,
    "tier": 3
  },
  {
    "category": "experiment_H",
    "description": "H1: Modality rotation for visual prompt injection (LLaVA)",
    "id": "H1",
    "depends_on": [],
    "steps": [
      "On LLaVA or similar VLM, assign different type rotations to image vs text tokens",
      "Check attention pattern shift without affecting multimodal understanding",
      "Construct visual prompt injection (text instructions embedded in images)",
      "Test if modality rotation defends against visual PI",
      "Verify: preliminary evidence that typed rotation extends to multimodal"
    ],
    "passes": false,
    "tier": 3
  },
  {
    "category": "experiment_H",
    "description": "H2: Retrieval confidence rotation for RAG hallucination control",
    "id": "H2",
    "depends_on": [],
    "steps": [
      "Simulate RAG: system + user query + 3 retrieved docs with different confidence levels",
      "Assign rotation angles proportional to confidence (high confidence = small angle, low = large angle)",
      "Measure model dependence on each doc in generated response",
      "Verify: lower confidence docs receive less attention authority"
    ],
    "passes": false,
    "tier": 3
  }
]
```

---

## Paper Figure List (from proj.md)

Based on the follow-up experiments, the paper needs these core figures:

1. **Conceptual diagram** — standard transformer vs typed context (already have: Fig 12)
2. **Certified attention matrix** — 3x3 trust hierarchy with actual attention heatmap (Exp A1)
3. **delta-ASR curve** — theoretical certified bound vs empirical ASR scatter + fit (Exp B1)
4. **Security-utility frontier** — alpha or |S| as X, ASR + benign accuracy dual-Y (Exp A2 + D2)
5. **Adaptive attack comparison** — GCG steps vs ASR for rotation and special token (Exp C1)
6. **LoRA ablation bar chart** — already have (Fig 10), supplement with benign accuracy
7. **Per-attack-type breakdown** — already have (Fig 11), supplement adaptive attack version
8. **Bound tightness distribution** — rho_actual histogram with rho_min annotated (Exp G1)

---

## Agent Instructions

1. Read `activity.md` first to understand current state
2. Read `proj.md` for detailed experiment designs, hypotheses, and expected results
3. Read `proj_v1.md` for original code templates and Phase 1-4 methodology
4. Find next task with `"passes": false`, respecting dependency order and tier priority
5. Complete all steps for that task
6. Run the verification step to confirm correctness
7. Update task to `"passes": true`
8. Log completion in `activity.md`
9. Repeat until all tasks pass

**Priority order**: Work on Tier 1 tasks first (respecting dependencies), then Tier 2, then Tier 3.

**Dependency notes**:
- P1 (special token baseline) must be completed before C1, C2, B2
- A1 must be completed before A2, B1, G1
- D1 must be completed before D2, D3
- E1 must be completed before E3
- B1 + C1 must be completed before B2

**Important:**
- Each task should produce runnable code with `--synthetic` mode for CPU-only validation.
- Use `proj.md` experiment designs as specifications but adapt code as needed.
- If GPU is unavailable, implement code and verify on synthetic data.
- Save all figures to `outputs/` with filenames specified in steps.
- All new experiments build on the existing codebase (model/, utils.py, configs/).

---

## Completion Criteria

All Tier 1 tasks marked with `"passes": true` for paper submission readiness.
All Tier 2 tasks for paper strengthening.
Tier 3 tasks are optional extensions.
