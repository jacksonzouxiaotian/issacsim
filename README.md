# Isaac Sim Narrow-Passage Low-Level Locomotion RL

This package is scoped as:

> RL-based low-level quadruped locomotion control for narrow-passage traversal.

It does not study memory, high-level navigation decisions, Nav2 planning, or
two-layer decision policies. The policy trained here is an ANYmal-C low-level
locomotion controller. Its action is a 12D joint position target.

## Research Scope

The task is to train a PPO locomotion policy that drives a quadruped from the
entrance of a narrow passage to the exit while staying centered, stable, smooth,
and collision-free.

Policy observations contain:

- robot proprioception: base linear/angular velocity, projected gravity, joint
  position, joint velocity, and previous action;
- local command: forward velocity/heading command from the locomotion task;
- compact local geometry: normalized progress, lateral offset, left/right wall
  clearance, distance to goal, Delta-D, and feasible clearance margin.

Policy observations do not contain memory counters such as stuck count, wedge
count, oscillation count, or failure history.

## Main Tasks

```bash
Isaac-Narrow-Gait-Anymal-C-v0
Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0
Isaac-Narrow-Gait-SensorEstimatedGeometry-Anymal-C-v0
```

Recovery-start curriculum tasks still exist, but they are reset distributions
only. The policy remains memory-free:

```bash
Isaac-Narrow-Gait-Recovery-Mild-Anymal-C-v0
Isaac-Narrow-Gait-Recovery-RightWall-SmallYaw-Anymal-C-v0
Isaac-Narrow-Gait-Recovery-Medium-Anymal-C-v0
Isaac-Narrow-Gait-Recovery-Hard-Anymal-C-v0
Isaac-Narrow-Gait-Recovery-NearWall-Clean-Anymal-C-v0
Isaac-Narrow-Gait-Recovery-Yaw-Clean-Anymal-C-v0
```

Generalization tasks:

```bash
Isaac-Narrow-Gait-Generalization-Doorway-Anymal-C-v0
Isaac-Narrow-Gait-Generalization-AsymmetricObstacle-Anymal-C-v0
Isaac-Narrow-Gait-Generalization-LCorridor-Anymal-C-v0
```

## PPO Task Design

Observation space:

- base linear velocity, base angular velocity, projected gravity;
- commanded base velocity;
- joint positions and velocities;
- previous joint target action;
- corridor geometry vector:
  `x_norm, y_norm, left_clearance_norm, right_clearance_norm, dist_to_goal_norm, delta_d_norm, feasible_margin_norm`.

Action space:

- 12D ANYmal-C joint position target from IsaacLab's joint position action term.

Reward terms:

- forward progress reward;
- clean goal reached bonus;
- centerline tracking penalty;
- unsafe clearance penalty;
- collision/contact penalty;
- yaw alignment penalty;
- base height and orientation stability penalties;
- action rate penalty;
- torque and joint acceleration penalties;
- time penalty.

Termination:

- timeout;
- clean goal reached;
- base contact;
- bad orientation or fall;
- base too low;
- sustained stuck condition for evaluation/training cutoff.

Evaluation metrics:

- clean success rate;
- raw success rate;
- collision rate;
- wedge rate;
- timeout rate;
- mean time to goal;
- min-clearance mean and minimum;
- yaw-error mean;
- oscillation count;
- action smoothness;
- fall rate.

## Important Files

- `source/isaaclab_tasks/.../anymal_c_narrow/narrow_gait_env_cfg.py`
  - low-level locomotion environment, reset curricula, scene variants, and
    reward weights.
- `source/isaaclab_tasks/.../anymal_c_narrow/mdp_narrow.py`
  - current-state geometry, reward, reset, and termination helpers.
- `source/isaaclab_tasks/.../anymal_c_narrow/__init__.py`
  - Gym task registration for low-level gait tasks.
- `scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py`
  - low-level checkpoint evaluator that writes CSV, summary JSON, and markdown
    tables. Multi-scenario or multi-width evaluations are launched as isolated
    subprocesses so Isaac Sim does not need to recreate multiple worlds inside
    one process. Timeout terminations are counted in `timeout_rate`.
- `scripts/narrow_passage/validate_narrow_pipeline.sh`
  - convenience validation entry point for width scans, recovery-start tests,
    and generalization tests.

## Quick Training

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name low_level_gait_width085_stage1 \
  --device cuda:0
```

## Quick Evaluation

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0 \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/<memory_free_run>/model_<iter>.pt \
  --scenarios nominal \
  --widths 0.75 0.85 0.95 \
  --num_envs 64 \
  --max_steps 600 \
  --output logs/narrow_passage_eval/width_scan_oracle.csv \
  --headless \
  --device cuda:0
```

Validation helper:

```bash
LOW_LEVEL_CHECKPOINT=logs/rsl_rl/anymal_c_narrow_gait/<memory_free_run>/model_<iter>.pt \
  bash scripts/narrow_passage/validate_narrow_pipeline.sh
```

Fast smoke validation:

```bash
LOW_LEVEL_CHECKPOINT=logs/rsl_rl/anymal_c_narrow_gait/<memory_free_run>/model_<iter>.pt \
  NUM_ENVS=4 MAX_STEPS=80 WIDTHS="0.85" RUN_RECOVERY=0 RUN_GENERALIZATION=0 \
  bash scripts/narrow_passage/validate_narrow_pipeline.sh
```

## Current Recovery Curriculum Notes

The recommended recovery sequence after a clean Stage 1 gait checkpoint is:

1. `Isaac-Narrow-Gait-Recovery-Mild-Anymal-C-v0`
   - mild entrance, small yaw, and light near-wall resets;
   - used to preserve nominal traversal while introducing recovery starts.
2. `Isaac-Narrow-Gait-Recovery-RightWall-SmallYaw-Anymal-C-v0`
   - focused right-wall and small-yaw resets;
   - intended for the common failure mode where `right_wall_start` has high
     collision rate.
3. `Isaac-Narrow-Gait-Recovery-NearWall-Clean-Anymal-C-v0` or
   `Isaac-Narrow-Gait-Recovery-Yaw-Clean-Anymal-C-v0`
   - use only after the focused stage no longer degrades nominal traversal.

Do not report a recovery policy as a main contribution unless fixed-start
evaluation reaches acceptable clean success and collision rates. The current
repository is designed to make this failure mode visible rather than hide it.
