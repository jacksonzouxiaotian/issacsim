# Isaac Sim Narrow-Passage Low-Level Gait Work Package

This package contains the narrow-passage low-level gait work extracted from the
local IsaacLab workspace. It intentionally does not contain the full IsaacLab
repository.

## Contents

- `INSTALL.md`
  - Reproducible IsaacLab overlay instructions and pinned workspace commit.
- `RUN.md`
  - Commands for width scans, oracle/sensor geometry comparison, recovery
    ablations, staged recovery training, and generalization evals.
- `ISAACLAB_VERSION.txt`
  - Minimal version lock for this work package.
- `make_eval_tables.py`
  - Builds compact markdown tables from evaluation CSV files.
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
- `logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_*_staged_memory_*_v1/`
  - staged recovery curriculum checkpoints for mild, medium, and hard reset
    distributions.
- `logs/narrow_passage_eval/`
  - CSV and summary JSON files from offline evaluation.

## Main Task

```bash
Isaac-Narrow-Gait-Anymal-C-v0
```

Recommended experiment split:

```bash
Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0
Isaac-Narrow-Gait-SensorEstimatedGeometry-Anymal-C-v0
```

Generalization scene entries:

```bash
Isaac-Narrow-Gait-Generalization-Doorway-Anymal-C-v0
Isaac-Narrow-Gait-Generalization-AsymmetricObstacle-Anymal-C-v0
Isaac-Narrow-Gait-Generalization-LCorridor-Anymal-C-v0
```

Scope statement: this package validates an Isaac Sim low-level
narrow-passage traversal and recovery module. It should not be presented as a
complete navigation benchmark by itself.

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

## Staged Recovery Curriculum

The recovery curriculum is split into explicit reset stages:

```bash
Isaac-Narrow-Gait-Recovery-Mild-Anymal-C-v0
Isaac-Narrow-Gait-Recovery-Medium-Anymal-C-v0
Isaac-Narrow-Gait-Recovery-Hard-Anymal-C-v0
```

The evaluator now writes both:

- `raw_success`: reached the goal.
- `success`: clean success, meaning goal reached without collision, wedge, or
  rejection. This prevents `success && collision` from being counted as success.

Staged memory checkpoints:

```bash
logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_09-59-07_staged_memory_mild_v1/model_498.pt
logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_10-02-47_staged_memory_medium_v1/model_697.pt
logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_10-07-24_staged_memory_hard_v1/model_896.pt
```

Training chain:

```bash
# stage-1 -> mild
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Recovery-Mild-Anymal-C-v0 \
  --headless --num_envs 2048 --max_iterations 200 \
  --resume --load_run 2026-06-30_15-57-59_low_level_gait_width085_stage1 \
  --checkpoint model_299.pt \
  --run_name staged_memory_mild_v1

# mild -> medium
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Recovery-Medium-Anymal-C-v0 \
  --headless --num_envs 2048 --max_iterations 200 \
  --resume --load_run 2026-07-02_09-59-07_staged_memory_mild_v1 \
  --checkpoint model_498.pt \
  --run_name staged_memory_medium_v1

# medium -> hard
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Recovery-Hard-Anymal-C-v0 \
  --headless --num_envs 2048 --max_iterations 200 \
  --resume --load_run 2026-07-02_10-02-47_staged_memory_medium_v1 \
  --checkpoint model_697.pt \
  --run_name staged_memory_hard_v1
```

Staged hard checkpoint, clean-success evaluation:

| scenario | raw SR | clean SR | collision | wedge | time_mean | clearance_mean | osc_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_wall | 0.8281 | 0.4531 | 0.4844 | 0.0 | 5.408s | 0.181 | 39.094 |
| yaw_left | 0.2656 | 0.1406 | 0.3125 | 0.0 | 8.615s | 0.343 | 73.281 |

Compared with the previous single hard-mixed stage, staged curriculum improves
clean success for `left_wall` and reduces collision in both `left_wall` and
`yaw_left`. The policy still needs additional hard-stage tuning before the
recovery results are publication-clean.

Additional recovery runs on 2026-07-02:

| checkpoint | scenario | raw SR | clean SR | collision | wedge | time_mean | clearance_mean | osc_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| staged_memory_hard_v2/model_1295.pt | left_wall | 0.9375 | 0.2656 | 0.6719 | 0.0 | 4.268s | 0.177 | 35.516 |
| staged_memory_hard_v2/model_1295.pt | yaw_left | 0.2656 | 0.1562 | 0.3281 | 0.0 | 8.390s | 0.339 | 74.328 |
| staged_memory_hard_clean_reward_v1/model_1494.pt | left_wall | 0.9375 | 0.3125 | 0.6406 | 0.0 | 4.004s | 0.177 | 36.828 |
| staged_memory_hard_clean_reward_v1/model_1494.pt | yaw_left | 0.2031 | 0.1094 | 0.3750 | 0.0 | 7.728s | 0.347 | 68.219 |

Interpretation: hard_v2 improves raw goal-reaching but learns collision-heavy
passage. Clean-reward fine-tuning slightly improves `left_wall` clean SR but
does not solve the core recovery problem and hurts `yaw_left`. Recovery should
still not be claimed as a main contribution.

Baseline naming note: the built-in `heuristic_dwb_like`,
`heuristic_rpp_like`, and `heuristic_mppi_like` controllers are lightweight
local heuristics. They are not real Nav2 DWB/RPP/MPPI results.
