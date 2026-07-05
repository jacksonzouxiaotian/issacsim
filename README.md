# Isaac Sim Narrow-Passage Low-Level Locomotion RL

This repository contains Isaac Sim / Isaac Lab experiments for narrow-passage
low-level quadruped locomotion control. The robot is ANYmal-C, and the policy is
trained with PPO to output low-level joint position targets. The task is local
narrow-passage traversal: move through a corridor while maintaining clearance,
heading alignment, stability, and smooth actions.

This repository does not train high-level memory, does not implement complete
navigation decision-making, and does not evaluate full Nav2-style planning. Its
scope is low-level locomotion control and Isaac Sim validation for narrow
passages.

## Current Results

The current nominal traversal result shows that the PPO locomotion policy can
reliably pass straight narrow corridors:

| Method | SR@75 | SR@85 | SR@95 | collision | wedge | reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO low-level gait | 1.000 | 0.984 | 1.000 | 0.005 | 0.000 | 0.000 |

Recovery-start scenarios remain much harder:

| Scenario | Success | Collision | Mean min clearance | Mean oscillation |
| --- | ---: | ---: | ---: | ---: |
| left_wall | 0.438 | 0.484 | 0.180 | 33.39 |
| right_wall | 0.000 | 1.000 | 0.187 | 7.44 |
| yaw_left | 0.188 | 0.812 | 0.348 | 10.34 |
| yaw_right | 0.156 | 0.500 | 0.330 | 45.00 |

See [RESULTS.md](RESULTS.md) for the full result summary and interpretation.
See [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md) for the recovery-start failure
mode analysis.

## Result Figures

The repository includes paper-style figures generated from
`logs/narrow_passage_eval/`:

![Success and collision rates across passage widths](figures/width_success_collision.png)

![Recovery-start success and collision comparison](figures/scenario_recovery_failure.png)

![Clearance and oscillation comparison](figures/clearance_oscillation.png)

Regenerate them with:

```bash
python scripts/plot_narrow_results.py \
  --input_dir logs/narrow_passage_eval \
  --output_dir figures
```

## Key Findings

- Nominal narrow-passage traversal is strong: the low-level PPO policy achieves
  high clean success on 0.75 m, 0.85 m, and 0.95 m straight corridors.
- Recovery-start scenarios are harder: near-wall and yawed initial states expose
  contact-dominated failures, especially on right-wall and yaw-left starts.
- Low-level RL alone is not a complete narrow-passage autonomy stack. These
  results motivate combining low-level locomotion control with a higher-level
  failure-aware navigation, recovery, or memory module.

## Task Design

Observation space:

- robot proprioception: base linear/angular velocity, projected gravity, joint
  position, joint velocity, and previous action;
- low-level command: forward velocity and heading command;
- compact local geometry: normalized progress, lateral offset, left/right
  clearance, distance to goal, Delta-D, and feasible clearance margin.

Action space:

- 12D ANYmal-C joint position target from Isaac Lab's joint position action
  term.

Reward terms emphasize:

- forward progress;
- clean goal reaching;
- centerline tracking;
- clearance safety;
- collision avoidance;
- yaw alignment;
- base height and orientation stability;
- action smoothness, torque, and joint acceleration regularization;
- time efficiency.

The policy observation does not include stuck counters, wedge counters,
oscillation history, failure memory, or high-level decision state.

## Main Tasks

Nominal and geometry variants:

```bash
Isaac-Narrow-Gait-Anymal-C-v0
Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0
Isaac-Narrow-Gait-SensorEstimatedGeometry-Anymal-C-v0
```

Recovery-start curriculum tasks:

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

Baseline and ablation tasks:

```bash
Isaac-Narrow-Gait-Ablation-FullReward-Anymal-C-v0
Isaac-Narrow-Gait-Ablation-NoClearanceReward-Anymal-C-v0
Isaac-Narrow-Gait-Ablation-NoCenterlineReward-Anymal-C-v0
Isaac-Narrow-Gait-Ablation-NoRecoveryCurriculum-Anymal-C-v0
```

The scripted velocity-controller baseline is exposed through:

```bash
scripts/narrow_passage/evaluate_scripted_velocity_controller.py
```

It is an evaluator interface and CSV schema template until a real Isaac Sim
scripted rollout is connected; no scripted baseline numbers are reported from
template rows.

## Repository Structure

- `source/`
  - Isaac Lab task registration, ANYmal-C narrow-passage environment
    configuration, reset curricula, reward terms, and narrow-passage MDP helper
    functions.
- `scripts/`
  - RSL-RL training and evaluation scripts, including
    `evaluate_narrow_passage.py`, which exports per-trial CSV files, summary
    JSON files, and markdown tables.
- `logs/`
  - local evaluation outputs when experiments are run in this workspace. Logs
    are used for result aggregation but are not required to understand the task
    code.
- `make_eval_tables.py`
  - utility for aggregating mixed-schema evaluation CSV files into
    `logs/narrow_passage_eval/eval_tables.md`, grouped by method, scenario, and
    width when a `method` column is available.
- `RUN.md`
  - reproducible training and evaluation commands.
- `RESULTS.md`
  - current nominal traversal and recovery-start result summary.

## Quick Start

Train a nominal low-level gait policy:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Anymal-C-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name low_level_stage1_wide_straight \
  --device cuda:0
```

Evaluate a checkpoint on a width scan:

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

Aggregate evaluation tables:

```bash
python make_eval_tables.py \
  --input_dir logs/narrow_passage_eval \
  --output logs/narrow_passage_eval/eval_tables.md
```

Standardize an existing evaluator CSV for an ablation method:

```bash
python scripts/narrow_passage/standardize_eval_csv.py \
  --input logs/narrow_passage_eval/right_wall_small_yaw_eval.csv \
  --output logs/narrow_passage_eval/full_reward_policy.csv \
  --method full_reward_policy
```

For staged curriculum commands and recovery-start evaluation protocols, see
[RUN.md](RUN.md).
