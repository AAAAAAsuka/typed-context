# Typed Context via Persistent Rotation — Project Plan

## Overview

Research codebase for investigating and validating typed context encoding via RoPE rotation for prompt injection defense. The project progresses through 4 experimental phases: probing, RoPE analysis, ICL verification, and LoRA finetuning.

**Reference:** `proj.md` (full engineering document with theoretical motivation, expected results, and visualization specs)

---

## Task List

```json
[
  {
    "category": "setup",
    "description": "Project scaffolding and environment setup",
    "steps": [
      "Create directory structure: data/, analysis/, model/, experiments/, configs/, outputs/, screenshots/",
      "Create requirements.txt with: torch>=2.1, transformers>=4.40, peft>=0.10, accelerate, scikit-learn, numpy, matplotlib, seaborn, datasets, bitsandbytes",
      "Create configs/llama8b.yaml with model_name, head_dim, num_layers, rope_theta, max_position_embeddings",
      "Create a shared utils.py with: load_model helper (supports fp16/int8/int4), chat_template tokenizer wrapper, token-level source label assignment (system=0, user=1, external=2)",
      "Verify: python -c 'import torch; from transformers import AutoModelForCausalLM; print(\"OK\")' runs without error"
    ],
    "passes": true
  },
  {
    "category": "data",
    "description": "Build probing datasets with token-level source labels",
    "steps": [
      "Create data/build_probe_data.py",
      "Generate Dataset A: 500 normal conversations (diverse system prompts + benign user queries) saved as data/normal.jsonl",
      "Generate Dataset B: 500 PI attack samples using known attack templates (system prompt extraction, instruction override, role-play, payload smuggling) saved as data/pi_attacks.jsonl",
      "For each PI sample, run inference on the target model and label as pi_success or pi_fail based on keyword matching (does response contain secret / follow injected instruction)",
      "Split into data/pi_success.jsonl and data/pi_fail.jsonl",
      "Create data/build_probe_data.py that tokenizes each sample with chat template and outputs token-level source labels (0=system, 1=user) using the system-only tokenization boundary method from proj.md Section 4.3",
      "Create position-swapped control dataset data/swapped.jsonl (user content before system) for confound control",
      "Verify: each jsonl has correct count, token labels are correct by spot-checking 5 samples"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 1: Extract hidden states for probing",
    "steps": [
      "Create analysis/extract_hidden_states.py that loads model in fp16, runs forward pass with output_hidden_states=True, saves per-layer hidden states and source labels to .npz files in outputs/hidden_states/",
      "Process all 4 datasets: normal, pi_success, pi_fail, swapped",
      "Implement batched extraction (batch_size=4) to manage GPU memory",
      "Verify: outputs/hidden_states/ contains .npz files, shapes are (num_tokens, hidden_dim) per layer"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 1: Train linear probes and generate visualizations",
    "steps": [
      "Create analysis/linear_probe.py that loads hidden states, trains LogisticRegression per layer, reports accuracy",
      "Run probes on all 4 datasets independently (normal, pi_success, pi_fail, swapped)",
      "Generate Figure 1: layer-wise probe accuracy curves (3 lines: normal, pi_success, pi_fail) with 95% CI from 5-fold CV, saved as outputs/fig1_probe_accuracy.png",
      "Generate Figure 2: t-SNE at 3 layers (early/mid/late), colored by source type and shaped by condition, saved as outputs/fig2_tsne_{early,mid,late}.png",
      "Generate Figure 3: top-20 probe weight dimensions as bar chart per layer, saved as outputs/fig3_probe_weights.png",
      "Report swapped dataset accuracy separately as confound control",
      "Verify: all figures saved, probe accuracy on normal data > 0.7 at early layers (sanity check)"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 2: RoPE frequency spectrum analysis",
    "steps": [
      "Create analysis/rope_analysis.py",
      "Compute and print frequency spectrum: for each subspace, compute freq, wavelength, rotations within max context length",
      "Categorize subspaces into HIGH_FREQ (wavelength < 4), USEFUL, LOW_FREQ (wavelength > max_ctx)",
      "Generate Figure 4: frequency spectrum bar+line chart with candidate subspaces highlighted, saved as outputs/fig4_rope_spectrum.png",
      "Verify: output lists candidate subspaces for repurposing"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 2: RoPE dimension ablation study",
    "steps": [
      "Create analysis/rope_ablation.py that hooks into attention layers to zero out specified RoPE subspaces",
      "Compute perplexity on 100 WikiText sequences (2048 tokens each) for: baseline, ablate top-5 high-freq, ablate top-5 low-freq, ablate 5 random mid-freq, ablate top-5 highest-importance",
      "Generate Figure 5: ablation perplexity bar chart with error bars, saved as outputs/fig5_ablation_ppl.png",
      "Identify final set of target_subspaces where ablation causes < 1% relative PPL change",
      "Save target subspaces to configs/target_subspaces.json",
      "Verify: configs/target_subspaces.json exists and contains at least 2 subspace indices"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 3: Implement typed RoPE core module",
    "steps": [
      "Create model/typed_rope.py with: create_type_rotation(), apply_typed_rope() functions as specified in proj.md Section 6.2",
      "Create model/hook_attention.py with: install_type_rotation_hooks() that patches all attention layers at inference time to use typed RoPE for target subspaces",
      "Write unit test: verify that with type_id=0 the output matches standard RoPE exactly (no regression)",
      "Write unit test: verify that different type_ids produce different Q/K rotations in target subspaces only",
      "Verify: python -m pytest model/test_typed_rope.py passes"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 3: PPL tolerance sweep",
    "steps": [
      "Create experiments/ppl_sweep.py",
      "Sweep rotation_angle in [0.01, 0.05, 0.1, 0.2, 0.5, pi/8, pi/4, pi/2] with target_subspaces from configs/target_subspaces.json",
      "Also sweep num_subspaces in [1, 2, 4, 8] at angle=pi/4",
      "Generate Figure 6: PPL vs rotation angle line chart (one line per num_subspaces), saved as outputs/fig6_ppl_tolerance.png",
      "Determine max_safe_angle (largest angle where PPL increase < 5%) and save to configs/experiment.yaml",
      "Verify: configs/experiment.yaml contains max_safe_angle and chosen num_subspaces"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 3: Attention pattern analysis under type rotation",
    "steps": [
      "Create experiments/attention_analysis.py",
      "For 10 representative samples, compute attention weights with and without type rotation",
      "Measure attention gap: mean(same-type attn) - mean(cross-type attn) across all heads and layers",
      "Generate Figure 7: 2x2 attention heatmap grid (with/without rotation × most/least affected head), saved as outputs/fig7_attention_heatmaps.png",
      "Report which heads show largest shift (these are candidate 'type-sensitive' heads)",
      "Verify: attention gap is positive for at least some heads (type rotation creates measurable ingroup bias)"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 3: ICL experiment — delimiter baseline",
    "steps": [
      "Create experiments/icl_experiment.py with modular design: takes a 'condition' argument",
      "Implement Condition A (baseline): standard chat, no modifications",
      "Implement Condition B (delimiter_icl): [TRUSTED]/[UNTRUSTED] text markers + 3 few-shot demos as in proj.md Section 6.5",
      "Create data/pi_benchmark.jsonl: 200+ PI attacks covering 4 categories (extraction, override, role_play, smuggling)",
      "Implement 2-tier PI success judgment: keyword matching + LLM judge for ambiguous cases, as specified in proj.md Section 6.6",
      "Run Conditions A and B, save results to outputs/icl_results.json",
      "Verify: results contain strict_ASR, soft_ASR, benign_acc, degradation for both conditions"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 3: ICL experiment — rotation conditions",
    "steps": [
      "Extend experiments/icl_experiment.py with Condition C (rotation_only), D (rotation_icl), E (random_rotation_icl) as defined in proj.md Section 6.5-6.6",
      "Use max_safe_angle and target_subspaces from configs/",
      "For Condition E: apply random per-token rotations (not correlated with type) as control",
      "Run all conditions on the same pi_benchmark.jsonl",
      "Append results to outputs/icl_results.json",
      "Generate Figure 8: grouped bar chart of ASR and benign_acc across 5 conditions with significance markers, saved as outputs/fig8_icl_results.png",
      "Verify: all 5 conditions have complete results"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 3: Indirect PI experiment (3-type)",
    "steps": [
      "Create data/build_indirect_pi.py: generate 200 samples with system + user + external (retrieved doc containing hidden PI payload)",
      "Extend model/typed_rope.py to support 3 type categories (system=0, user=1, external=2)",
      "Run ICL experiment with 2-type (system vs rest) and 3-type (system/user/external) conditions",
      "Save results to outputs/indirect_pi_results.json",
      "Verify: results show 3-type has lower ASR than 2-type on indirect PI"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 3.5: Quantization robustness check",
    "steps": [
      "Create experiments/quantization_check.py",
      "Load model in fp16, int8, int4",
      "For each quantization × rotation angle, measure attention gap (same-type vs cross-type)",
      "Test type category capacity: find max k where adjacent types are distinguishable",
      "Generate Figure 9: quantization survival heatmap, saved as outputs/fig9_quant_heatmap.png",
      "Verify: report clearly states which quant levels preserve the signal"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 4: LoRA finetuning with typed RoPE",
    "steps": [
      "Create data/build_training_data.py: generate 5000 training samples (2000 normal + 2000 PI refusal + 1000 hard negatives) as specified in proj.md Section 8.2",
      "Create experiments/lora_train.py: LoRA config (r=16, target q/k/v/o proj), typed RoPE hooks active during training",
      "Train for 3 epochs with lr=2e-5, save adapter to outputs/lora_adapter/",
      "Verify: training loss decreases, adapter files saved"
    ],
    "passes": true
  },
  {
    "category": "experiment",
    "description": "Phase 4: LoRA evaluation with ablation",
    "steps": [
      "Create experiments/evaluate_lora.py",
      "Evaluate 4 conditions: baseline (no LoRA, no rotation), LoRA without rotation, LoRA with rotation, LoRA with rotation removed at test time",
      "Run on same pi_benchmark.jsonl + indirect PI benchmark",
      "Generate Figure 10: ASR bar chart with ablation, saved as outputs/fig10_lora_ablation.png",
      "Generate Figure 11: per-attack-type breakdown radar chart, saved as outputs/fig11_attack_breakdown.png",
      "Verify: ablation (rotation removed) shows ASR increase → model learned to use type signal"
    ],
    "passes": true
  },
  {
    "category": "finalize",
    "description": "Generate summary figure and compile results",
    "steps": [
      "Create Figure 12: conceptual diagram (standard transformer vs typed context) as SVG or PNG, saved as outputs/fig12_conceptual.png",
      "Create outputs/results_summary.md: compile all metrics, key findings, and figure references",
      "Ensure all 12 figures are in outputs/",
      "Verify: outputs/results_summary.md exists and references all figures"
    ],
    "passes": true
  }
]
```

---

## Agent Instructions

1. Read `activity.md` first to understand current state
2. Read `proj.md` for detailed specifications, expected results, and code templates
3. Find next task with `"passes": false`
4. Complete all steps for that task
5. Run the verification step to confirm correctness
6. Update task to `"passes": true`
7. Log completion in `activity.md`
8. Repeat until all tasks pass

**Important:**
- Only modify the `passes` field. Do not remove or rewrite tasks.
- Each task should produce runnable code. Use `proj.md` code templates as starting points but adapt as needed.
- If a step requires GPU and none is available, implement the code and verify it runs on a small synthetic input (2-3 samples, 1-2 layers).
- Save all figures to `outputs/` with exact filenames specified in the steps.
- When loading the model, check GPU availability first. If no GPU with enough VRAM, use CPU with a smaller slice of data for validation.

---

## Completion Criteria

All tasks marked with `"passes": true`
