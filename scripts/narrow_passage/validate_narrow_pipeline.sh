#!/usr/bin/env bash
set -euo pipefail

: "${LOW_LEVEL_CHECKPOINT:?Set LOW_LEVEL_CHECKPOINT to a checkpoint trained with the current memory-free observation space.}"
OUT_DIR="${OUT_DIR:-logs/narrow_passage_validation}"
NUM_ENVS="${NUM_ENVS:-32}"
MAX_STEPS="${MAX_STEPS:-600}"
DEVICE="${DEVICE:-cuda:0}"
WIDTHS="${WIDTHS:-0.75 0.85 0.95}"
RUN_WIDTH_SCAN="${RUN_WIDTH_SCAN:-1}"
RUN_RECOVERY="${RUN_RECOVERY:-1}"
RUN_GENERALIZATION="${RUN_GENERALIZATION:-0}"

mkdir -p "${OUT_DIR}"

if [[ "${RUN_WIDTH_SCAN}" == "1" ]]; then
  echo "[1/3] Running low-level width scan..."
  # shellcheck disable=SC2086
  conda run --no-capture-output -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
    --task Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0 \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --scenarios nominal \
    --widths ${WIDTHS} \
    --num_envs "${NUM_ENVS}" \
    --max_steps "${MAX_STEPS}" \
    --output "${OUT_DIR}/width_scan_oracle.csv" \
    --headless \
    --device "${DEVICE}"
else
  echo "[1/3] Skipping width scan."
fi

if [[ "${RUN_RECOVERY}" == "1" ]]; then
  echo "[2/3] Running low-level recovery-start validation..."
  conda run --no-capture-output -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
    --task Isaac-Narrow-Gait-Recovery-NearWall-Clean-Anymal-C-v0 \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --scenarios left_wall_start right_wall_start yaw_left_start yaw_right_start \
    --widths 0.85 \
    --num_envs "${NUM_ENVS}" \
    --max_steps "${MAX_STEPS}" \
    --output "${OUT_DIR}/recovery_starts.csv" \
    --headless \
    --device "${DEVICE}"
else
  echo "[2/3] Skipping recovery-start validation."
fi

if [[ "${RUN_GENERALIZATION}" == "1" ]]; then
  echo "[3/3] Running generalization validation..."
  conda run --no-capture-output -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
    --task Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0 \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --scenarios doorway asymmetric_obstacle L_corridor \
    --widths 0.85 \
    --num_envs "${NUM_ENVS}" \
    --max_steps "${MAX_STEPS}" \
    --output "${OUT_DIR}/generalization.csv" \
    --headless \
    --device "${DEVICE}"
else
  echo "[3/3] Skipping generalization validation."
fi

echo "Validation complete."
echo "Checkpoint: ${LOW_LEVEL_CHECKPOINT}"
echo "Output directory: ${OUT_DIR}"
