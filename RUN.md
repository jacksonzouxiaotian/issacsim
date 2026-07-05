# Running Low-Level Narrow-Passage Locomotion Experiments

This repository is for low-level PPO locomotion control only:

> RL-based low-level quadruped locomotion control for narrow-passage traversal.

It does not train memory, high-level decision, or full navigation policies.

## Reward Definition

Edit reward helpers in:

```bash
source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/anymal_c_narrow/mdp_narrow.py
```

Edit reward terms and weights in:

```bash
source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/anymal_c_narrow/narrow_gait_env_cfg.py
```

The main config class is `NarrowGaitEnvCfg`. The reward is composed of:

- `forward_progress`: positive x-direction progress;
- `success_bonus`: clean goal reached without collision/fall/wedge;
- `centerline_penalty`: lateral deviation from passage centerline;
- `unsafe_clearance`: near-wall safety penalty;
- `undesired_contacts` and `failure_termination`: collision avoidance;
- `yaw_alignment`: heading alignment with the corridor;
- `flat_orientation_l2` and `base_height`: stability;
- `action_rate_l2`, `dof_torques_l2`, `dof_acc_l2`: smoothness/effort;
- `time_penalty`: discourages slow traversal.

No reward term depends on failure memory or history counters.

## Curriculum Plan

Stage 1: wide straight passage.

- Width: `0.95m-1.10m`.
- Reset: entrance start, `x=(-0.9,-0.5)`, `y=(-0.03,0.03)`, `yaw=(-0.05,0.05)`.
- Reward: emphasize forward progress and stability, mild clearance penalty.
- Train: 800-1500 PPO iterations.
- Checkpoint name: `low_level_stage1_wide_straight`.

Stage 2: medium narrow passage.

- Width: `0.85m-0.95m`.
- Reset: entrance start, `y=(-0.05,0.05)`, `yaw=(-0.10,0.10)`.
- Reward: increase centerline and clearance weights.
- Train: 400-800 additional iterations from Stage 1.
- Checkpoint name: `low_level_stage2_medium_centerline`.

Stage 3: narrow passage.

- Width: `0.75m-0.85m`.
- Reset: stronger entrance lateral/yaw noise.
- Reward: stronger unsafe-clearance and collision penalties.
- Train: 600-1000 additional iterations.
- Checkpoint name: `low_level_stage3_narrow_safety`.

Stage 4: recovery-start training without memory.

- Reset: near-wall and yawed initial states inside the corridor.
- Observation: unchanged current-state proprioception + geometry only.
- Reward: current-state centerline, clearance, yaw, forward progress, stability.
- Train: 400-800 additional iterations.
- Checkpoint name: `low_level_stage4_recovery_starts_current_state`.

Stage 4b: focused right-wall and small-yaw recovery.

- Task: `Isaac-Narrow-Gait-Recovery-RightWall-SmallYaw-Anymal-C-v0`.
- Reset: mostly right-wall starts with small yaw, plus a small amount of
  entrance, left-wall, and center-yaw starts to reduce forgetting.
- Motivation: use this stage when fixed evaluation shows high
  `right_wall_start` collision rate while nominal traversal remains strong.
- Reward: stronger clean-success, contact-failure, and unsafe-clearance
  pressure, with local realignment rewards kept small.
- Train: 400-800 additional iterations from the best mild recovery checkpoint.
- Checkpoint name: `low_level_stage2d_right_wall_small_yaw`.

Stage 5: generalization testing.

- Scenes: doorway, asymmetric obstacle, L-corridor.
- No extra training required unless reporting a separate fine-tuned policy.
- Metrics: clean success, collision, wedge, min clearance, yaw error, oscillation,
  smoothness, and time to goal.

## Train

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name low_level_stage1_wide_straight \
  --device cuda:0
```

Continue with recovery-start reset distributions:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Recovery-Mild-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 400 \
  --resume \
  --load_run <previous_run> \
  --checkpoint <previous_checkpoint.pt> \
  --run_name low_level_stage4_recovery_mild_current_state \
  --device cuda:0
```

Focused right-wall/small-yaw recovery:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Recovery-RightWall-SmallYaw-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 500 \
  --resume \
  --load_run <previous_run> \
  --checkpoint <previous_checkpoint.pt> \
  --run_name low_level_stage2d_right_wall_small_yaw \
  --device cuda:0
```

## Baseline / Ablation Runs

Use the same PPO runner and train each ablation under a separate `run_name`.
These tasks keep the low-level controller, observation space, and action space
unchanged; only reward weights or reset curriculum are changed.

Full reward policy:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Ablation-FullReward-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 800 \
  --resume \
  --load_run <previous_run> \
  --checkpoint <previous_checkpoint.pt> \
  --run_name ablation_full_reward_policy \
  --device cuda:0
```

No clearance reward:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Ablation-NoClearanceReward-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 800 \
  --resume \
  --load_run <previous_run> \
  --checkpoint <previous_checkpoint.pt> \
  --run_name ablation_no_clearance_reward \
  --device cuda:0
```

No centerline reward:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Ablation-NoCenterlineReward-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 800 \
  --resume \
  --load_run <previous_run> \
  --checkpoint <previous_checkpoint.pt> \
  --run_name ablation_no_centerline_reward \
  --device cuda:0
```

No recovery curriculum:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Ablation-NoRecoveryCurriculum-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 800 \
  --resume \
  --load_run <previous_run> \
  --checkpoint <previous_checkpoint.pt> \
  --run_name ablation_no_recovery_curriculum \
  --device cuda:0
```

Scripted velocity-controller baseline interface:

```bash
python scripts/narrow_passage/evaluate_scripted_velocity_controller.py \
  --write_template \
  --output logs/narrow_passage_eval/scripted_velocity_controller.csv
```

The scripted controller template is not a completed baseline result. Report it
only after connecting the script to real Isaac Sim rollouts and filling per-trial
metrics.

## Evaluate

Width scan:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0 \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/<run>/model_<iter>.pt \
  --scenarios nominal \
  --widths 0.75 0.85 0.95 \
  --num_envs 64 \
  --max_steps 600 \
  --output logs/narrow_passage_eval/width_scan_oracle.csv \
  --headless \
  --device cuda:0
```

Recovery-start scenarios:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-Recovery-RightWall-SmallYaw-Anymal-C-v0 \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/<run>/model_<iter>.pt \
  --scenarios nominal left_wall_start right_wall_start yaw_left_start yaw_right_start \
  --widths 0.85 \
  --num_envs 64 \
  --max_steps 800 \
  --output logs/narrow_passage_eval/right_wall_small_yaw_eval.csv \
  --headless \
  --device cuda:0
```

Generalization:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0 \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/<run>/model_<iter>.pt \
  --scenarios doorway asymmetric_obstacle L_corridor \
  --widths 0.85 \
  --num_envs 64 \
  --max_steps 600 \
  --output logs/narrow_passage_eval/generalization.csv \
  --headless \
  --device cuda:0
```

The evaluator writes:

- per-trial CSV;
- summary JSON;
- markdown table.

For multi-width or multi-scenario runs, the evaluator launches each combination
in an isolated subprocess. This avoids Isaac Sim world-recreation hangs between
scenarios. `timeout_rate` is computed from the environment's `time_out`
termination term as well as unfinished episodes.

## Standardize Ablation CSV

Each ablation or baseline should be stored with the shared schema:

```text
method, scenario, width, trial, success, collision, wedge, rejected, oscillation_count, completion_time, min_clearance
```

Convert an evaluator CSV without modifying the original file:

```bash
python scripts/narrow_passage/standardize_eval_csv.py \
  --input logs/narrow_passage_eval/right_wall_small_yaw_eval.csv \
  --output logs/narrow_passage_eval/full_reward_policy.csv \
  --method full_reward_policy
```

Then aggregate method-grouped tables:

```bash
python make_eval_tables.py \
  --input_dir logs/narrow_passage_eval \
  --output logs/narrow_passage_eval/eval_tables.md
```

## Convenience Validation

```bash
LOW_LEVEL_CHECKPOINT=logs/rsl_rl/anymal_c_narrow_gait/<memory_free_run>/model_<iter>.pt \
  bash scripts/narrow_passage/validate_narrow_pipeline.sh
```

Fast check:

```bash
LOW_LEVEL_CHECKPOINT=logs/rsl_rl/anymal_c_narrow_gait/<memory_free_run>/model_<iter>.pt \
  NUM_ENVS=4 MAX_STEPS=80 WIDTHS="0.85" RUN_RECOVERY=0 RUN_GENERALIZATION=0 \
  bash scripts/narrow_passage/validate_narrow_pipeline.sh
```
