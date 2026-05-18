# How LLMs do political censorship internally

## Layout

```
qwc/                    library
  config.py             paths, model paths, writer-band layout
  data.py               D1 prompts loader (200 prompts, 4 classes)
  taxonomy.py           8-register taxonomy + per-cell predicted windows
  model.py              load Qwen3.5-9B, render chat template
  activations.py        cache last-token / per-position residuals
  directions.py         diff-of-means extraction, projections, QR ortho basis
  hooks.py              steering / subspace-patch / ablation hook factories
  patching.py           layer-output, K/V capture + inject
  generate.py           batched generation with hooks
  judge.py              blind LLM-judge wrapper (Anthropic API)
  logit_lens.py         CJK token mask, logit lens utilities
  probes.py             logistic regression, AUC, affine OLS
  io.py                 small JSON / NPZ helpers
scripts/
  00_cache_activations.py            cache last-token residuals at every tap
  01_extract_directions.py           E3 — d_prc, d_refuse, d_style + sanity report
  02_split_grid_inputs.py            split steering_grid_public.json into inputs only
  03_generate_grid.py                generate 352 cells × 3 rollouts
  04_judge_grid.py                   blind LLM judge over the regenerated grid
  05_compare_grids.py                in-window rate + label diff vs reference
  e1_baseline_registers.py           baseline registers + non-PRC controls
  e2_base_vs_posttrain.py            base-vs-posttrain CJK commitment + probes
  e4_necessity_sufficiency.py        single-direction subspace patch + PCA on complement
  e5_dose_response.py                writer-direction dose-response sigmoid
  e6_writer_layer_id.py              tap sweep + α-effectiveness sweep
  e7_writer_linearity.py             affine map from 3D coords -> writer output
  e8_mean_replace_l19.py             full L19-output last-7 mean replace
  e10_single_layer_output_patch.py   L20-L28 layer-output patch sweep
  e11_mean_replace_3d.py             3D-subspace-only mean replace
  e12_per_head_ablation.py           single head + top-10 + Q-zero
  e13_mlp_neuron_ablation.py         600 reader-band MLP neurons
  e15_single_layer_kv_replace.py     per-layer K/V replace + topic-token zero
  e16_l23h9_attention.py             attention class-divergence + topic-token decode
  e18_random_direction_floor.py      random-direction brittleness floor per class
  e19_logit_lens.py                  CJK-top1 fraction per tap per class
  e20_mid_layer_rollout.py           passthrough rollout from each tap
  e21_chinese_unembedding.py         ban Chinese tokens at lm_head; verify verdict unchanged
  e22_dzh_decomposition.py           d_zh = 16% 3D + 84% complement; steer each
  e23_distributed_translation.py     per-component ablation in L24-L30
  e24_lm_head_geometry.py            cos(zh_refusal_mean_row, en_refusal_mean_row)
  e26_per_topic_stickiness.py        d_prc-suppression sweep per PRC topic
  e27_residual_geometry.py           residual moves uniformly across topics
  e28_thinking_mode.py               same circuit drives thinking-mode verdict
  e29_deflection_script.py           Tiananmen 5-step deflection script analysis
  e30_prefill_attack.py              crafted-thinking-prefill per-language sweep
  e31_subcomponent_attribution.py    MLP vs attention share per direction
  e32_cross_class_subspace.py        tia 3D coords into harmless target null
  e34_per_mlp_probe.py               4-class verdict probe per reader-band MLP
  e35_per_neuron_specificity.py      per-neuron class-specificity counts
  e36_distributed_dosing.py          fixed total α split across writers
  e37_cumulative_writer_ablation.py  d_style ablation cumulative across L15-L18
  e38_token_perturbation.py          replace topic tokens with [X]
  e39_dprc_projection.py             d_prc projection per class + neutral outliers
  e40_drefuse_projection.py          d_refuse projection per class + outliers
  e41_dprc_overgen_causal.py         steer −d_prc on Kosovo / Catalonia / Saudi
  e42_drefuse_overgen_causal.py      steer −d_refuse on Arab Spring / aspirin
  e44_per_position_logit_lens.py     CJK-top1 per position offset at tap 24
  e45_positional_sufficiency.py      d_prc last-K prefill sufficiency, 3-class
  e46_channel_transplant.py          3D-vs-complement L19 last-7 transplant, 3-class
  e47_dstyle_thinkdose.py            d_style thinking-mode dose-response (deflection<->propaganda)
data/
  prompts.json                        the 200-prompt dataset (D1)
  steering_grid_public.json           the published 352-cell grid (reference)
  steering_grid_inputs.json           produced by 02_split_grid_inputs.py
results/                              outputs land here (gitignored in practice)
figures/                              plots
```

Note on experiment numbering: the `eNN` numbering is intentionally
non-contiguous. Several experiments (e9, e14, e17, e25, e33, e43) were found
to be measurement artifacts / incorrect and have been removed from the repo
entirely.

## Environment

Python 3.11. Create a virtualenv and install the dependencies:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- **`transformers` must be installed from main**, not a release. The
  Qwen3.5 architecture (Gated-DeltaNet linear-attention layers) is only in
  the development branch: `pip install
  git+https://github.com/huggingface/transformers.git`. A tagged release
  will fail to load the checkpoint.
- `torch` ≥ 2.7 with a CUDA build. The 9B checkpoint runs in bf16 and fits
  on a single 24 GB GPU; ~22 GB peak per model instance.
- Attention backend: most scripts use the default (SDPA). `e16` needs
  attention weights, so it loads with `attn_implementation="eager"`
  automatically — no action required.
- The LLM-judge scripts need `anthropic` and `ANTHROPIC_API_KEY` in the
  environment. The judge model is configurable via `QWC_JUDGE_MODEL`
  (and `QWC_JUDGE_MODEL_FALLBACK`); the non-judge metric scripts need
  neither. Revoke any key you export when you're done.

## Model paths

Point the two env vars at local copies of the checkpoints:

```bash
export QWC_MODEL_POSTTRAIN=/path/to/Qwen3.5-9B
export QWC_MODEL_BASE=/path/to/Qwen3.5-9B-Base   # only needed for E2
```

`qwc.config` derives the `data/` and `results/` directories from the
package location, so the project runs wherever you clone it.

## Running the pipeline

The numbered scripts run in order: `00_cache_activations.py` →
`01_extract_directions.py` → `02_split_grid_inputs.py` →
`03_generate_grid.py` → `04_judge_grid.py` → `05_compare_grids.py`.

Each e-script is self-contained and runnable on its own; all have `--help`
for argument details. Most depend on the activations and directions `.npz`
files, so run `00` and `01` first.

## What each direction is

```
d_prc    = mean(prc_sensitive)         - mean(neutral_political),   tap 14
d_refuse = mean(harmful)               - mean(harmless),              tap 19
d_style  = mean(tiananmen)             - mean(prc_other),             tap 19
```

Sign convention: positive `d_style` points toward the Tiananmen-deflection
register (the tia side); negative `d_style` points toward the propaganda
register. Six flagged overgeneralization / anomaly IDs in
`_meta.overgeneralization_and_anomalies` are excluded from the
direction-extraction class means (they trigger writer-direction
overgeneralization at baseline).

## Notes on schema and reproducibility

- `data/steering_grid_public.json` carries both inputs and outputs.
  `scripts/02_split_grid_inputs.py` extracts just the inputs into
  `data/steering_grid_inputs.json`; `scripts/03_generate_grid.py` fills in
  the outputs, producing a file with the same schema as the published one
  (minus the `judge` / `coherence` / `in_predicted_window` fields, which
  `04_judge_grid.py` adds).
- Generation uses the chat template with `add_generation_prompt=True,
  enable_thinking=False`; three rollouts per cell (greedy; T=0.7, top_p=0.9,
  seed 1234; and seed 1235); 256 new tokens. Sampling is stochastic, so the
  exact response text varies run-to-run while the judged verdict *registers*
  are largely stable; a handful of borderline cells can drift.
- Harmful-comply text in the reproduced grid is NOT auto-truncated. If
  republishing, mirror the public file's `[content withheld]` truncation
  manually.
