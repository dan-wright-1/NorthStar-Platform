#!/usr/bin/env bash
# teardown-lab3.sh
# Deletes everything Lab 3 can leave running and bills for.
#
# Endpoints are the Lab 3 equivalent of Lab 2's NAT Gateway: they bill hourly
# until explicitly deleted, they are easy to create from a notebook and forget,
# and nothing in SageMaker will remind you. Training jobs are bounded and stop
# on their own; endpoints do not.
#
# Usage: bash scripts/teardown-lab3.sh
set -uo pipefail
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
mkdir -p docs

echo "==> Lab 3 teardown  (account $(aws sts get-caller-identity --query Account --output text))"

echo "[1/4] Deleting SageMaker endpoints"
for e in $(aws sagemaker list-endpoints --query 'Endpoints[*].EndpointName' --output text 2>/dev/null | tr '\t' '\n'); do
  [ -z "$e" ] && continue
  aws sagemaker delete-endpoint --endpoint-name "$e" >/dev/null 2>&1 && echo "      deleted endpoint $e"
done

echo "[2/4] Deleting endpoint configs"
for c in $(aws sagemaker list-endpoint-configs --query 'EndpointConfigs[*].EndpointConfigName' --output text 2>/dev/null | tr '\t' '\n'); do
  [ -z "$c" ] && continue
  aws sagemaker delete-endpoint-config --endpoint-config-name "$c" >/dev/null 2>&1 && echo "      deleted config $c"
done

echo "[3/4] Stopping in-flight training / processing / transform jobs"
for j in $(aws sagemaker list-training-jobs --status-equals InProgress --query 'TrainingJobSummaries[*].TrainingJobName' --output text 2>/dev/null | tr '\t' '\n'); do
  [ -z "$j" ] && continue
  aws sagemaker stop-training-job --training-job-name "$j" >/dev/null 2>&1 && echo "      stopped training job $j"
done
for j in $(aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[*].ProcessingJobName' --output text 2>/dev/null | tr '\t' '\n'); do
  [ -z "$j" ] && continue
  aws sagemaker stop-processing-job --processing-job-name "$j" >/dev/null 2>&1 && echo "      stopped processing job $j"
done
for j in $(aws sagemaker list-transform-jobs --status-equals InProgress --query 'TransformJobSummaries[*].TransformJobName' --output text 2>/dev/null | tr '\t' '\n'); do
  [ -z "$j" ] && continue
  aws sagemaker stop-transform-job --transform-job-name "$j" >/dev/null 2>&1 && echo "      stopped transform job $j"
done

echo "[4/4] Verifying"
{
  fail=0
  # A failed AWS call is a FAILED check, never "OK". This used to be
  # `$(eval "$2" || echo 0)`, so expired credentials or a wrong profile read
  # as zero resources everywhere and produced a clean-looking teardown file
  # that had verified nothing (2026-10-01).
  check () {
    if ! n=$(eval "$2" 2>/dev/null); then
      printf "      %-28s CHECK FAILED (AWS call errored; credentials/region?)\n" "$1"; fail=1; return
    fi
    # The CLI applies --query PER PAGE, so length() on a paginated list call
    # prints one count per page ("0\n0" on a clean account). Sum them. Before
    # this, every clean account reported "STILL PRESENT: 0 0" and exited 1.
    n=$(printf '%s\n' "$n" | awk '{s+=$1} END {print s+0}')
    if [ "$n" = "0" ]; then printf "      %-28s OK\n" "$1"
    else printf "      %-28s STILL PRESENT: %s\n" "$1" "$n"; fail=1; fi
  }
  check "SageMaker endpoints"   "aws sagemaker list-endpoints --query 'length(Endpoints)' --output text"
  check "Endpoint configs"      "aws sagemaker list-endpoint-configs --query 'length(EndpointConfigs)' --output text"
  check "Training jobs running" "aws sagemaker list-training-jobs --status-equals InProgress --query 'length(TrainingJobSummaries)' --output text"
  check "Processing jobs"       "aws sagemaker list-processing-jobs --status-equals InProgress --query 'length(ProcessingJobSummaries)' --output text"
  # The MLflow TRACKING SERVER bills $0.60/hr until deleted -- $10 budget gone
  # in 16.7 hours, ~$43 over a weekend. It is not an endpoint, so nothing else
  # catches it. The serverless MLflow APP is free and deliberately NOT checked:
  # students keep it for Labs 4 and 6.
  check "MLflow TRACKING SERVERS" "aws sagemaker list-mlflow-tracking-servers --query 'length(TrackingServerSummaries)' --output text"
  echo ""
  if [ "$fail" = "0" ]; then
    echo "==> Lab 3 teardown complete. No billable inference or training resources remain."
    echo "    Model Registry entries cost nothing - kept."
    echo "    Your MLflow APP is serverless and free - kept on purpose; Labs 4 and 6 log to it."
    echo "    Run scripts/teardown-lab2.sh as well if you are done with the data platform."
  else
    echo "==> WARNING: teardown NOT verified. Fix anything marked STILL PRESENT or"
    echo "    CHECK FAILED and re-run. Do not submit this file as evidence."
    echo "    An MLflow TRACKING SERVER is \$0.60/hr. Delete it now:"
    echo "      aws sagemaker delete-mlflow-tracking-server --tracking-server-name <name>"
    echo "    Stopping is NOT deleting - a stopped server can be restarted and still holds storage."
    exit 1
  fi
} | tee docs/lab3-teardown-output.txt
