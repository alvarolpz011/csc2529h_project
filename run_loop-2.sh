#!/bin/bash
set -euo pipefail
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.* ]] && continue
        [[ -z "$key" ]] && continue
        export "$key"="$(echo "$value" | tr -d '"')"
    done < "$ENV_FILE"
else
    echo "Missing .env"
    exit 1
fi
NUM_STEPS="${NUM_STEPS:-4}"
LAMBDA_PRIOR="${LAMBDA_PRIOR:-0.0}"
mkdir -p "${DEPTHSPLAT_OUT_ROOT}"
mkdir -p "${DIFIX_OUT_ROOT}"

run_depthsplat() {
  local in_dir="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  local cmd="${DEPTHSPLAT_CMD//INPUT_DIR/${in_dir}}"
  cmd="${cmd//OUTPUT_DIR/${out_dir}}"
  eval "${cmd}"
}

compute_psnr() {
  local pred_dir="$1"
  local tag="$2"
  python scripts/compute_psnr.py --gt_dir "${DATASET_GT_DIR}" --pred_dir "${pred_dir}" --tag "${tag}"
}

for (( STEP=0; STEP<=NUM_STEPS; STEP++ )); do
  DEPTH_OUT_STEP="${DEPTHSPLAT_OUT_ROOT}/step${STEP}"
  DIFIX_OUT_STEP="${DIFIX_OUT_ROOT}/step${STEP}"

  if [ "${STEP}" -eq 0 ]; then
    run_depthsplat "${DATASET_GT_DIR}" "${DEPTH_OUT_STEP}"
  else
    PREV="${DIFIX_OUT_ROOT}/step$((STEP - 1))"
    run_depthsplat "${PREV}" "${DEPTH_OUT_STEP}"
  fi

  compute_psnr "${DEPTH_OUT_STEP}" "depthsplat_step${STEP}"

  if [ "${STEP}" -eq 0 ]; then
    python feed_depthsplat_out_to_diffix.py       --input_dir "${DEPTH_OUT_STEP}"       --output_dir "${DIFIX_OUT_STEP}"       --prev_input_dir "${DATASET_GT_DIR}"       --lambda_prior "${LAMBDA_PRIOR}"
  else
    PREV_INPUT="${DEPTHSPLAT_OUT_ROOT}/step$((STEP - 1))"
    python feed_depthsplat_out_to_diffix.py       --input_dir "${DEPTH_OUT_STEP}"       --output_dir "${DIFIX_OUT_STEP}"       --prev_input_dir "${PREV_INPUT}"       --lambda_prior "${LAMBDA_PRIOR}"
  fi

  compute_psnr "${DIFIX_OUT_STEP}" "diffix_step${STEP}"

done
