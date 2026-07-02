# Installation

This package is a reproducible overlay for IsaacLab. It does not vendor the
full IsaacLab repository.

## Tested Workspace

- IsaacLab commit: `20f77074b0ac2e64f22bb63ef4f7070977a2dac0`
- Isaac Sim asset version in saved configs: `5.1`
- Local conda environment name: `issaaclabdog`
- Task scope: Isaac Sim low-level narrow-passage quadruped traversal and
  recovery validation

## Restore The Workspace

From a clean IsaacLab checkout:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
cd IsaacLab
git checkout 20f77074b0ac2e64f22bb63ef4f7070977a2dac0
```

Install IsaacLab following the upstream IsaacLab instructions for Isaac Sim 5.1.
Then copy this package over the IsaacLab root:

```bash
rsync -av narrow_gait_upload/source/ source/
rsync -av narrow_gait_upload/scripts/ scripts/
rsync -av narrow_gait_upload/logs/ logs/
```

Activate the environment used for the experiments:

```bash
conda activate issaaclabdog
```

## Verify Task Registration

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt \
  --widths 0.85 \
  --num_envs 4 \
  --max_steps 20 \
  --output logs/narrow_passage_eval/install_smoke.csv \
  --headless \
  --device cuda:0
```

If this command starts Isaac Sim and writes `install_smoke.csv`, the overlay is
installed correctly.
