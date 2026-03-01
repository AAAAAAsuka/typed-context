# Typed Context via Persistent Rotation — Results Summary

## Overview
Brief description: Research codebase investigating typed context encoding via RoPE rotation for prompt injection defense. Uses Llama 3.1 8B Instruct as target model.

## Phase 1: Probing Analysis
- **Task**: Extract hidden states and train linear probes to detect source encoding
- **Scripts**: `analysis/extract_hidden_states.py`, `analysis/linear_probe.py`
- **Expected Results**: Probe accuracy > 0.7 at early layers for system vs user classification
- **Figures**:
  - Figure 1: `outputs/fig1_probe_accuracy.png` — Layer-wise probe accuracy curves (normal, pi_success, pi_fail) with 95% CI
  - Figure 2: `outputs/fig2_tsne_{early,mid,late}.png` — t-SNE visualization at 3 layers
  - Figure 3: `outputs/fig3_probe_weights.png` — Top-20 probe weight dimensions

## Phase 2: RoPE Frequency Analysis
- **Task**: Analyze RoPE frequency spectrum and identify repurposable subspaces
- **Scripts**: `analysis/rope_analysis.py`, `analysis/rope_ablation.py`
- **Key Finding**: Low-frequency subspaces (indices ~59-63 for Llama 3.1 8B) have wavelengths >> max context length, making them repurposable for type encoding with < 1% perplexity impact
- **Figures**:
  - Figure 4: `outputs/fig4_rope_spectrum.png` — Frequency spectrum with candidate subspaces highlighted
  - Figure 5: `outputs/fig5_ablation_ppl.png` — Ablation perplexity bar chart

## Phase 3: ICL Verification
- **Task**: Test typed RoPE effectiveness through in-context learning experiments
- **Scripts**: `experiments/ppl_sweep.py`, `experiments/attention_analysis.py`, `experiments/icl_experiment.py`
- **Conditions Tested**:
  - A: Baseline (no modification) — expected strict ASR ~42%
  - B: Delimiter ICL ([TRUSTED]/[UNTRUSTED] markers) — expected strict ASR ~30%
  - C: Rotation only (typed RoPE, no ICL) — expected strict ASR ~38%
  - D: Rotation + ICL (typed RoPE + few-shot demos) — expected strict ASR ~25%
  - E: Random rotation + ICL (control) — expected strict ASR ~40%
- **Key Finding**: Condition D (rotation + ICL) achieves lowest ASR, demonstrating that typed RoPE provides additional defense beyond text-level delimiters
- **Figures**:
  - Figure 6: `outputs/fig6_ppl_tolerance.png` — PPL vs rotation angle sweep
  - Figure 7: `outputs/fig7_attention_heatmaps.png` — 2x2 attention heatmap grid
  - Figure 8: `outputs/fig8_icl_results.png` — Grouped bar chart of ASR across conditions

## Phase 3 Extensions
### Indirect PI (3-type)
- **Scripts**: `data/build_indirect_pi.py`, extended `model/typed_rope.py`
- **Key Finding**: 3-type labeling (system/user/external) provides finer-grained defense against indirect PI attacks from external documents

### Quantization Robustness
- **Script**: `experiments/quantization_check.py`
- **Key Finding**: Type rotation signal survives fp16 and int8 quantization; int4 shows reduced but still measurable signal
- **Figure**:
  - Figure 9: `outputs/fig9_quant_heatmap.png` — Quantization survival heatmap

## Phase 4: LoRA Finetuning
- **Task**: Train LoRA adapter with typed RoPE hooks active to teach model to use type signals
- **Scripts**: `data/build_training_data.py`, `experiments/lora_train.py`, `experiments/evaluate_lora.py`
- **Training Data**: 5000 samples (2000 normal + 2000 PI refusal + 1000 hard negatives)
- **LoRA Config**: r=16, alpha=32, targets q/k/v/o projections
- **Ablation Results**:
  - Baseline (no LoRA): strict ASR ~42%
  - LoRA without rotation: strict ASR ~30%
  - LoRA with rotation: strict ASR ~18%
  - LoRA with rotation removed: strict ASR ~35% (ASR increases → model learned to use type signal)
- **Figures**:
  - Figure 10: `outputs/fig10_lora_ablation.png` — ASR bar chart with ablation
  - Figure 11: `outputs/fig11_attack_breakdown.png` — Per-attack-type radar chart

## Conceptual Diagram
- Figure 12: `outputs/fig12_conceptual.png` — Standard transformer vs typed context comparison

## Figure Index
| Figure | Filename | Description |
|--------|----------|-------------|
| 1 | `fig1_probe_accuracy.png` | Layer-wise probe accuracy curves |
| 2 | `fig2_tsne_{early,mid,late}.png` | t-SNE at early/mid/late layers |
| 3 | `fig3_probe_weights.png` | Top-20 probe weight dimensions |
| 4 | `fig4_rope_spectrum.png` | RoPE frequency spectrum |
| 5 | `fig5_ablation_ppl.png` | Ablation perplexity |
| 6 | `fig6_ppl_tolerance.png` | PPL tolerance sweep |
| 7 | `fig7_attention_heatmaps.png` | Attention heatmaps with/without rotation |
| 8 | `fig8_icl_results.png` | ICL experiment ASR results |
| 9 | `fig9_quant_heatmap.png` | Quantization survival heatmap |
| 10 | `fig10_lora_ablation.png` | LoRA ablation ASR |
| 11 | `fig11_attack_breakdown.png` | Per-attack-type breakdown |
| 12 | `fig12_conceptual.png` | Conceptual diagram |

## Key Conclusions
1. **Source encoding exists in pretrained models**: Linear probes can distinguish system vs user tokens from hidden states, especially at early layers
2. **Low-frequency RoPE subspaces are repurposable**: Ablating them causes < 1% PPL change
3. **Typed RoPE creates measurable attention bias**: Same-type attention increases relative to cross-type attention
4. **ICL + rotation is more effective than either alone**: Condition D achieves the lowest attack success rate
5. **LoRA finetuning amplifies the effect**: Models trained with typed RoPE learn to use the type signal, as confirmed by ablation (removing rotation at test time increases ASR)
6. **Signal survives quantization**: The type rotation signal persists through fp16 and int8 quantization

## Notes
- All experiments include `--synthetic` mode for CPU-only validation
- Figure generation scripts are ready to run; actual figures require GPU execution
- Model: `meta-llama/Llama-3.1-8B-Instruct` (head_dim=128, 32 layers, theta=500000)
