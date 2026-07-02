#!/usr/bin/env bash
set -euo pipefail

LOW_LEVEL_TASK="${LOW_LEVEL_TASK:-Isaac-Narrow-Gait-Recovery-NearWall-Clean-Anymal-C-v0}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_16-17-04_staged_memory_near_wall_clean_v1/model_1743.pt}"
LOW_LEVEL_EXPORT_DIR="${LOW_LEVEL_EXPORT_DIR:-$(dirname "${LOW_LEVEL_CHECKPOINT}")/exported}"

HIGH_LEVEL_TASK="${HIGH_LEVEL_TASK:-Isaac-Navigation-Narrow-Anymal-C-v0}"
HIGH_LEVEL_RUN_NAME="${HIGH_LEVEL_RUN_NAME:-decision_straight_with_narrow_gait_v1}"

NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-800}"
DEVICE="${DEVICE:-cuda:0}"

conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/export_policy.py \
  --task "${LOW_LEVEL_TASK}" \
  --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
  --output_dir "${LOW_LEVEL_EXPORT_DIR}" \
  --num_envs 1 \
  --headless \
  --device "${DEVICE}"

export ISAAC_NARROW_LOW_LEVEL_POLICY_PATH="${LOW_LEVEL_EXPORT_DIR}/policy.pt"

conda run -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${HIGH_LEVEL_TASK}" \
  --headless \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --run_name "${HIGH_LEVEL_RUN_NAME}" \
  --device "${DEVICE}"
