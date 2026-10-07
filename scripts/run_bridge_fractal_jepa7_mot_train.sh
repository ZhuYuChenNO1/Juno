#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${BASE_VLM:?Set BASE_VLM to a local Qwen3-VL-4B-Instruct checkpoint}"
: "${OXE_DATA_ROOT:?Set OXE_DATA_ROOT to the Bridge/Fractal LeRobot data root}"
: "${JEPA_CKPT_PATH:?Set JEPA_CKPT_PATH to the frozen JEPA encoder checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/bridge_fractal_jepa7_mot_1005.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${REPO_ROOT}/runs}"
RUN_ID="${RUN_ID:-juno-policy_bridge_fractal_jepa7_mot}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-juno-policy}"
WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${RUN_ROOT_DIR}"
export PYTHONPATH="${REPO_ROOT}${LEWM_ROOT:+:${LEWM_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_PROJECT WANDB_MODE

exec "${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file "${REPO_ROOT}/juno/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/juno/training/train_juno.py" \
  --config_yaml "${CONFIG_PATH}" \
  --run_id "${RUN_ID}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.jepa.ckpt_path "${JEPA_CKPT_PATH}" \
  --datasets.vla_data.data_root_dir "${OXE_DATA_ROOT}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --is_resume "${IS_RESUME:-true}"
