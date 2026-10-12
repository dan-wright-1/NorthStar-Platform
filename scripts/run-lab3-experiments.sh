#!/usr/bin/env bash
# Runs the Lab 3 churn experiment grid. Every run logs to the MLflow App and
# writes its evidence (metrics JSON, feature-importance plot) to
# docs/lab_3/results/<run-name>/, because teardown deletes the S3 copies.
#
# Usage: bash scripts/run-lab3-experiments.sh        (from the repo root, stack up)
# Requires: .venv from models/churn/requirements.txt, AWS credentials exported
#   (eval "$(aws configure export-credentials --format env)" for `aws login`).
set -euo pipefail

cd "$(dirname "$0")/../infrastructure/environments/dev"
BUCKET=$(terraform output -raw s3_bucket_name)
cd ../../..
: "${MLFLOW_APP_ARN:?set MLFLOW_APP_ARN to the MLflow App ARN}"

PY=./.venv/Scripts/python
[ -x "$PY" ] || PY=./.venv/bin/python

run() {
  name=$1; shift
  PYTHONIOENCODING=utf8 "$PY" models/churn/train.py \
    --artifacts-bucket "$BUCKET" --mlflow-arn "$MLFLOW_APP_ARN" \
    --run-name "$name" --model-dir .build/model \
    --output-data-dir "docs/lab_3/results/$name" "$@"
}

# Vary one thing at a time against the depth-6 reference.
run xgb-d6-eta0.1            --max-depth 6 --eta 0.1
run xgb-d4-eta0.1            --max-depth 4 --eta 0.1
run xgb-d3-eta0.05           --max-depth 3 --eta 0.05 --num-round 400
# Leakage check: same config as the reference, risk score included.
run xgb-d6-eta0.1-risk       --max-depth 6 --eta 0.1 --include-risk-score
