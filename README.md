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

## Two-Layer Decision Setup

The high-level task entries:

```bash
Isaac-Navigation-Narrow-Anymal-C-v0
Isaac-Navigation-LCorridor-Anymal-C-v0
```

are now wired as decision policies above the exported narrow-gait policy. The
high-level action is a 3D velocity command; the low-level executor remains the
12D joint-target gait policy.

The wrapper requires a TorchScript/JIT low-level policy, not a raw RSL-RL
`model_*.pt` checkpoint. Export the selected low-level checkpoint first:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Narrow-Gait-Recovery-NearWall-Clean-Anymal-C-v0 \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_16-17-04_staged_memory_near_wall_clean_v1/model_1743.pt \
  --num_envs 1 \
  --headless \
  --device cuda:0
```

`play.py` writes:

```bash
logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_16-17-04_staged_memory_near_wall_clean_v1/exported/policy.pt
```

To use another exported low-level executor:

```bash
export ISAAC_NARROW_LOW_LEVEL_POLICY_PATH=/absolute/or/relative/path/to/exported/policy.pt
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

Important evaluator correction on 2026-07-02:

The evaluator now freezes each environment's metrics after its first episode
termination. Earlier recovery CSVs could mix a collision from the first episode
with a success after automatic reset in the same env slot. Use the
`first_episode_*` CSVs for recovery claims.

Corrected first-episode recovery ablation:

| condition | clean SR | collision | time | clearance | osc |
| --- | ---: | ---: | ---: | ---: | ---: |
| w/o recovery, left_wall | 0.0000 | 0.6250 | 5.968s | 0.184 | 62.17 |
| clean recovery, left_wall | 0.6406 | 0.3125 | 4.004s | 0.183 | 12.25 |
| near-wall clean, left_wall | 0.6406 | 0.3125 | 3.532s | 0.181 | 10.25 |
| w/o recovery, right_wall | 0.0000 | 0.9688 | 3.573s | 0.186 | 27.00 |
| clean recovery, right_wall | 0.5781 | 0.3750 | 2.181s | 0.183 | 9.34 |
| near-wall clean, right_wall | 0.6094 | 0.3438 | 2.257s | 0.184 | 9.97 |

Yaw recovery remains weak under the same first-episode protocol:

| checkpoint | scenario | clean SR | collision |
| --- | --- | ---: | ---: |
| clean_reward_v1/model_1494.pt | yaw_left | 0.1562 | 0.3281 |
| balanced_clean_v1/model_1793.pt | yaw_left | 0.1875 | 0.4219 |
| yaw_clean_v1/model_1500.pt | yaw_left | 0.1875 | 0.2969 |
| yaw_clean_v1/model_1793.pt | yaw_left | 0.1875 | 0.4844 |
| clean_reward_v1/model_1494.pt | yaw_right | 0.1562 | 0.2969 |

Interpretation: the strongest current contribution is near-wall recovery, not
large-yaw recovery. The policy shows a meaningful recovery gain over the
stage-1 gait checkpoint, but it should still be framed as incomplete because it
does not reach 70% clean SR across all recovery modes.

Baseline naming note: the built-in `heuristic_dwb_like`,
`heuristic_rpp_like`, and `heuristic_mppi_like` controllers are lightweight
local heuristics. They are not real Nav2 DWB/RPP/MPPI results.
