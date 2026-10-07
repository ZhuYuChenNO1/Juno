# Reproducibility notes

## Scope

The public launcher covers the Bridge/Fractal policy-learning stage, using the flattened `juno-policy` implementation in this repository. It does not silently claim to reproduce the paper's JEPA pretraining or Juno-TTT stages.

## Fixed recipe

- Qwen3-VL-4B interface with BF16 training.
- 8 future frames, sampled with stride 2.
- 8 predictive reasoning queries.
- 7D delta end-effector actions with a 16-step horizon.
- 100,000 optimizer steps, 5,000-step warmup, cosine decay.
- DeepSpeed ZeRO-2 and gradient accumulation of 1.
- Frozen ViT-Base target, maximum alignment weight 0.2, input cross-attention scale 0.5.

These settings are machine-readable in `configs/bridge_fractal_jepa7_mot_1005.yaml`.

## External inputs

The data root, base VLM checkpoint, JEPA checkpoint, optional le-wm package, and CUDA environment are intentionally supplied at runtime. Keep them outside the repository and record their exact revisions in your experiment log.

## Reproducibility boundary

Reported paper scores are included in the README as reference values. A score should only be attributed to this code after running the prescribed dataset split and evaluation protocol; this repository does not bundle the evaluation datasets or model weights.
