#!/usr/bin/env bash
# Reproduce every experiment arm reported in the write-up.
#
#   bash run_matrix.sh
#
# Ten runs, all at matched processed targets (1200 steps x 32 x 256 = 9,830,400):
#   arm 1     the supplied baseline, via --implementation model
#   arms 2-10 student, one per positional-encoding mode x three seeds
#
# 'learned' is the ablation control that isolates position information; it is bit
# identical to model.py, so arm 1 and runs/learned-s17 should agree.
set -euo pipefail
cd "$(dirname "$0")/code"

PY=".venv/Scripts/python.exe"          # Linux/macOS: .venv/bin/python
SEEDS="17 18 19"
ARMS="learned none rope"

"$PY" train.py --implementation model --device cuda --seed 17 --eval-every 300 \
  --run-dir runs/baseline-model-s17

for mode in $ARMS; do
  for seed in $SEEDS; do
    "$PY" train.py --implementation student --config "configs/pos-${mode}.json" \
      --device cuda --seed "$seed" --eval-every 300 --run-dir "runs/${mode}-s${seed}"
  done
done

echo "all runs finished"
