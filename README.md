# Isaac Sim Narrow-Passage Low-Level Gait Work Package

This package contains the narrow-passage low-level gait work extracted from the
local IsaacLab workspace. It intentionally does not contain the full IsaacLab
repository.

## Contents

- `source/isaaclab_tasks/.../anymal_c_narrow/`
  - IsaacLab task registration and configs for narrow-passage experiments.
  - `narrow_gait_env_cfg.py` is the low-level gait task. The action is 12D joint
    position targets, not an upper-level navigation velocity command.
  - `mdp_narrow.py` contains geometry, memory, reward, and termination helpers.
- `scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py`
  - Offline evaluator for width sweeps, recovery scenarios, memory ablation, and
    Delta-D calibration outputs.
- `logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/`
  - `model_299.pt`: trained stage-1 low-level gait checkpoint.
  - `params/`: saved agent and environment configs.
- `logs/narrow_passage_eval/`
  - CSV and summary JSON files from offline evaluation.

## Main Task

```bash
Isaac-Narrow-Gait-Anymal-C-v0
```

Checkpoint:

```bash
logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt
```

## Re-run Width Evaluation

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt \
  --widths 0.75 0.85 0.95 \
  --num_envs 32 \
  --max_steps 600 \
  --output logs/narrow_passage_eval/low_level_gait_width_scan.csv \
  --headless \
  --device cuda:0
```

## Re-run Recovery Memory Ablation

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt \
  --widths 0.85 \
  --num_envs 32 \
  --max_steps 600 \
  --recovery_scenario left_wall \
  --output logs/narrow_passage_eval/low_level_gait_recovery_left_memory.csv \
  --headless \
  --device cuda:0
```

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt \
  --widths 0.85 \
  --num_envs 32 \
  --max_steps 600 \
  --recovery_scenario left_wall \
  --zero_recovery_memory \
  --output logs/narrow_passage_eval/low_level_gait_recovery_left_zero_memory.csv \
  --headless \
  --device cuda:0
```

## Current Offline Results

Width scan:

| width | SR | collision | wedge | time_mean | min_clearance_min | min_clearance_mean | osc_mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 0.9375 | 0.0625 | 0.0 | 8.498s | 0.223 | 0.339 | 98.125 |
| 0.85 | 1.0000 | 0.0000 | 0.0 | 8.535s | 0.386 | 0.397 | 96.219 |
| 0.95 | 1.0000 | 0.0000 | 0.0 | 8.496s | 0.437 | 0.447 | 94.406 |

Recovery left-wall scenario:

| memory | SR | collision | wedge | time_mean | min_clearance_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| on | 0.0 | 0.59375 | 0.0 | 6.404s | 0.183 |
| zeroed | 0.0 | 0.46875 | 0.0 | 8.147s | 0.181 |

Interpretation: the stage-1 checkpoint has learned straight narrow-passage
traversal, but has not yet learned in-corridor recovery from near-wall/yaw
misalignment. The next training stage should fine-tune from this checkpoint with
near-wall, yawed, and low-speed stuck reset states.
