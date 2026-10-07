#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
required=(
  "${ROOT}/juno/training/train_juno.py"
  "${ROOT}/juno/model/framework/VLM4A/juno_policy.py"
  "${ROOT}/juno/model/modules/world_model/JEPA.py"
  "${ROOT}/configs/bridge_fractal_jepa7_mot_1005.yaml"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "missing: ${path}" >&2; exit 1; }
done
bash -n "${ROOT}/scripts/run_bridge_fractal_jepa7_mot_train.sh"
"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import ast, pathlib, sys
root = pathlib.Path(sys.argv[1])
for path in root.joinpath("juno").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("[juno] Python AST and launcher checks passed")
PY
