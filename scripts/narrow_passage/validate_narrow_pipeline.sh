#!/usr/bin/env bash
set -euo pipefail

LOW_LEVEL_TASK="${LOW_LEVEL_TASK:-Isaac-Narrow-Gait-Recovery-NearWall-Clean-Anymal-C-v0}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-logs/rsl_rl/anymal_c_narrow_gait/2026-07-02_16-17-04_staged_memory_near_wall_clean_v1/model_1743.pt}"
LOW_LEVEL_EXPORT_DIR="${LOW_LEVEL_EXPORT_DIR:-$(dirname "${LOW_LEVEL_CHECKPOINT}")/exported}"
LOW_LEVEL_POLICY_PATH="${LOW_LEVEL_EXPORT_DIR}/policy.pt"

WIDTH_TASK="${WIDTH_TASK:-Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0}"
RECOVERY_TASK="${RECOVERY_TASK:-Isaac-Narrow-Gait-Recovery-NearWall-Clean-Anymal-C-v0}"
HIGH_LEVEL_TASK="${HIGH_LEVEL_TASK:-Isaac-Navigation-Narrow-Anymal-C-v0}"

OUT_DIR="${OUT_DIR:-logs/narrow_passage_validation}"
NUM_ENVS="${NUM_ENVS:-32}"
MAX_STEPS="${MAX_STEPS:-600}"
DEVICE="${DEVICE:-cuda:0}"
WIDTHS="${WIDTHS:-0.75 0.85 0.95}"
RUN_WIDTH_SCAN="${RUN_WIDTH_SCAN:-1}"
RUN_RECOVERY="${RUN_RECOVERY:-1}"
RUN_DECISION_SMOKE="${RUN_DECISION_SMOKE:-1}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${LOW_LEVEL_POLICY_PATH}" ]]; then
  echo "[1/4] Exporting low-level gait policy..."
  conda run --no-capture-output -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/export_policy.py \
    --task "${LOW_LEVEL_TASK}" \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --output_dir "${LOW_LEVEL_EXPORT_DIR}" \
    --num_envs 1 \
    --headless \
    --device "${DEVICE}"
else
  echo "[1/4] Reusing existing low-level gait policy: ${LOW_LEVEL_POLICY_PATH}"
fi

if [[ "${RUN_WIDTH_SCAN}" == "1" ]]; then
  echo "[2/4] Running low-level width scan..."
  # shellcheck disable=SC2086
  conda run --no-capture-output -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
    --task "${WIDTH_TASK}" \
    --controller checkpoint \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --widths ${WIDTHS} \
    --num_envs "${NUM_ENVS}" \
    --max_steps "${MAX_STEPS}" \
    --output "${OUT_DIR}/width_scan_oracle.csv" \
    --headless \
    --device "${DEVICE}"
else
  echo "[2/4] Skipping low-level width scan."
fi

if [[ "${RUN_RECOVERY}" == "1" ]]; then
  echo "[3/4] Running left-wall recovery validation..."
  conda run --no-capture-output -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/evaluate_narrow_passage.py \
    --task "${RECOVERY_TASK}" \
    --controller checkpoint \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --widths 0.85 \
    --num_envs "${NUM_ENVS}" \
    --max_steps "${MAX_STEPS}" \
    --recovery_scenario left_wall \
    --output "${OUT_DIR}/recovery_left_wall.csv" \
    --headless \
    --device "${DEVICE}"
else
  echo "[3/4] Skipping left-wall recovery validation."
fi

if [[ "${RUN_DECISION_SMOKE}" == "1" ]]; then
  echo "[4/4] Running high-level decision interface smoke validation..."
  ISAAC_NARROW_LOW_LEVEL_POLICY_PATH="${LOW_LEVEL_POLICY_PATH}" \
    conda run --no-capture-output -n issaaclabdog python scripts/reinforcement_learning/rsl_rl/train.py \
    --task "${HIGH_LEVEL_TASK}" \
    --headless \
    --num_envs 16 \
    --max_iterations 1 \
    --run_name validate_decision_with_narrow_gait \
    --device "${DEVICE}"
else
  echo "[4/4] Skipping high-level decision interface smoke validation."
fi

echo "Validation complete."
echo "Low-level JIT policy: ${LOW_LEVEL_POLICY_PATH}"
if [[ "${RUN_WIDTH_SCAN}" == "1" ]]; then
  echo "Width scan: ${OUT_DIR}/width_scan_oracle.csv"
fi
if [[ "${RUN_RECOVERY}" == "1" ]]; then
  echo "Recovery eval: ${OUT_DIR}/recovery_left_wall.csv"
fi
