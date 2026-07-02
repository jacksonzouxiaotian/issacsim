# Running Experiments

This repository should be presented as:

> Isaac Sim low-level narrow-passage traversal and recovery validation.

It is not intended to carry the full evidence burden for a complete navigation
paper. The learned policy is a low-level gait/recovery module.

## Main Checkpoints

Stage-1 straight narrow-passage gait:

```bash
logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt
```

Staged recovery curriculum:

```bash
logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_09-59-07_staged_memory_mild_v1/model_498.pt
logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_10-02-47_staged_memory_medium_v1/model_697.pt
logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_10-07-24_staged_memory_hard_v1/model_896.pt
```

Current hard recovery checkpoint is not yet publication-clean as a main
contribution: clean SR is below 70% on the tested recovery scenarios.

## Export Low-Level Executor for High-Level Training

The high-level decision tasks use `PreTrainedPolicyAction`, which loads a
TorchScript/JIT `policy.pt`. It cannot load raw RSL-RL `model_*.pt`
checkpoints directly.

The complete one-command entry point is:

```bash
bash scripts/narrow_passage/train_decision_with_gait.sh
```

Useful overrides:

```bash
NUM_ENVS=1024 MAX_ITERATIONS=400 DEVICE=cuda:0 \
  bash scripts/narrow_passage/train_decision_with_gait.sh
```

Export the current low-level recovery executor:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/export_policy.py \
  --task Isaac-Narrow-Gait-Recovery-NearWall-Clean-Anymal-C-v0 \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_16-17-04_staged_memory_near_wall_clean_v1/model_1743.pt \
  --output_dir logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_16-17-04_staged_memory_near_wall_clean_v1/exported \
  --num_envs 1 \
  --headless \
  --device cuda:0
```

After export, high-level decision training can use:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Narrow-Anymal-C-v0 \
  --headless --num_envs 2048 --max_iterations 800 \
  --run_name decision_straight_with_narrow_gait_v1
```

For a different exported executor, set:

```bash
export ISAAC_NARROW_LOW_LEVEL_POLICY_PATH=/absolute/or/relative/path/to/exported/policy.pt
```

The runner for these high-level tasks is `NarrowDecisionPPORunnerCfg`, and logs
go under `logs/rsl_rl/anymal_c_narrow_decision/`.

## Validate Current Pipeline

Quick interface validation:

```bash
RUN_WIDTH_SCAN=0 RUN_RECOVERY=0 RUN_DECISION_SMOKE=1 \
  bash scripts/narrow_passage/validate_narrow_pipeline.sh
```

Low-level-only validation:

```bash
RUN_DECISION_SMOKE=0 NUM_ENVS=32 MAX_STEPS=600 \
  bash scripts/narrow_passage/validate_narrow_pipeline.sh
```

Full validation with the default width scan, left-wall recovery scenario, and
one-iteration high-level interface smoke:

```bash
bash scripts/narrow_passage/validate_narrow_pipeline.sh
```

Outputs are written to `logs/narrow_passage_validation/` by default. Use
`OUT_DIR=...`, `WIDTHS="0.75 0.85 0.95"`, `NUM_ENVS=...`, and `MAX_STEPS=...`
to change the evaluation scale.

## Geometry Experiments

Oracle geometry:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt \
  --widths 0.75 0.85 0.95 \
  --num_envs 64 \
  --max_steps 600 \
  --output logs/narrow_passage_eval/oracle_geometry_width_scan.csv \
  --headless \
  --device cuda:0
```

Sensor-estimated geometry:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-SensorEstimatedGeometry-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt \
  --widths 0.75 0.85 0.95 \
  --num_envs 64 \
  --max_steps 600 \
  --output logs/narrow_passage_eval/sensor_estimated_geometry_width_scan.csv \
  --headless \
  --device cuda:0
```

## Recovery Curriculum Training

Continue from stage-1 into staged recovery resets:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Recovery-Mild-Anymal-C-v0 \
  --headless --num_envs 2048 --max_iterations 200 \
  --resume --load_run 2026-06-30_15-57-59_low_level_gait_width085_stage1 \
  --checkpoint model_299.pt \
  --run_name staged_memory_mild_v2
```

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Recovery-Medium-Anymal-C-v0 \
  --headless --num_envs 2048 --max_iterations 200 \
  --resume --load_run <mild_run_name> \
  --checkpoint <mild_checkpoint.pt> \
  --run_name staged_memory_medium_v2
```

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Narrow-Gait-Recovery-Hard-Anymal-C-v0 \
  --headless --num_envs 2048 --max_iterations 400 \
  --resume --load_run <medium_run_name> \
  --checkpoint <medium_checkpoint.pt> \
  --run_name staged_memory_hard_v2
```

Do not promote recovery as the main contribution until clean SR is at least
70% on the clean recovery eval below.

## Recovery Ablation

w/o recovery:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-06-30_15-57-59_low_level_gait_width085_stage1/model_299.pt \
  --widths 0.85 \
  --num_envs 64 \
  --max_steps 600 \
  --recovery_scenario left_wall \
  --output logs/narrow_passage_eval/ablation_left_wall_wo_recovery.csv \
  --headless \
  --device cuda:0
```

w/ recovery:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-Recovery-Hard-NoMemory-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint <hard_no_memory_checkpoint.pt> \
  --widths 0.85 \
  --num_envs 64 \
  --max_steps 600 \
  --recovery_scenario left_wall \
  --output logs/narrow_passage_eval/ablation_left_wall_w_recovery.csv \
  --headless \
  --device cuda:0
```

w/ recovery + memory:

```bash
conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
  --task Isaac-Narrow-Gait-Recovery-Hard-Anymal-C-v0 \
  --controller checkpoint \
  --checkpoint logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_10-07-24_staged_memory_hard_v1/model_896.pt \
  --widths 0.85 \
  --num_envs 64 \
  --max_steps 600 \
  --recovery_scenario left_wall \
  --output logs/narrow_passage_eval/ablation_left_wall_w_recovery_memory.csv \
  --headless \
  --device cuda:0
```

## Generalization Table

Scene entries:

```bash
Isaac-Narrow-Gait-Generalization-Doorway-Anymal-C-v0
Isaac-Narrow-Gait-Generalization-AsymmetricObstacle-Anymal-C-v0
Isaac-Narrow-Gait-Generalization-LCorridor-Anymal-C-v0
```

Run each with the same evaluator and checkpoint, changing `--task` and
`--output`. The L-corridor entry is currently a generalization scaffold; report
it only after smoke testing and clean evaluation.

## Baselines

The evaluator includes `heuristic_dwb_like`, `heuristic_rpp_like`, and
`heuristic_mppi_like`. These are simple local heuristics and must not be
reported as real Nav2 DWB/RPP/MPPI. Use those names in tables unless real Nav2
results are integrated.

## Build Tables

```bash
python make_eval_tables.py \
  --input_dir logs/narrow_passage_eval \
  --output logs/narrow_passage_eval/eval_tables.md
```
