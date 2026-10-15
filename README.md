# NorthStar Retail — AI Platform (CS 401R)

Coursework repository for CS 401R. Builds the NorthStar Retail AI platform on AWS
across seven labs; each lab extends the previous one rather than replacing it.

## Layout

```
infrastructure/
  modules/        vpc, storage, iam, sagemaker, glue, feature_store
                  — reusable, environment-agnostic, no hardcoded names
  environments/
    dev/          the real AWS deployment (S3 remote state + DynamoDB lock)
    agent/        Lab 3 Track C AgentCore stack (own root: aws provider 6.x, own state key)
    local/        the same modules against LocalStack (no sagemaker, NAT, or
                  lifecycle rules — not emulated)
glue-scripts/
  transform.py         Glue ETL: raw/customers (CSV) -> processed/customers (Parquet)
  feature_engineer.py  Glue ETL: processed/customers -> features/customers + Feature Store
northstar-raw-sample.csv  synthetic transactions (~163k rows) for the Lab 2 pipeline
scripts/
  bootstrap-state.sh   one-time remote-state bucket + lock table setup
  verify-lab1.sh       Lab 1 rubric checks against a live stack
  verify-lab2.sh       Lab 2 rubric checks, incl. data-quality checks on S3 output
  run-lab3-experiments.sh  Lab 3 churn experiment grid (5 MLflow runs)
  collect-lab3-evidence.sh (+ collect_lab3_evidence.py)  exports Athena/MLflow/registry/AgentCore
                       evidence into docs/lab3-evidence/ and cross-checks it
  teardown-lab3.sh     Lab 3 teardown: endpoints, jobs, tracking servers, the AgentCore stack
  teardown-lab2.sh     full Lab 2 teardown, incl. resources Terraform can't see
  check-secrets.sh     scans all git history for committed credentials
models/churn/
  train.py             Lab 3 churn training: Athena -> XGBoost -> evaluation -> MLflow -> registry
  compare_models.py    XGBoost vs logistic regression vs MLP on the same split
  test_feature_query.py  unit tests (no AWS): query dedupe/soft-delete, preprocessing, evidence files
  requirements.txt     training dependencies
evaluation/
  evaluation_harness.py  course harness (Track C scoring via LocalAgentEvaluator)
agent/                 Lab 3 Track C (graded): AgentCore tools + Lambda handler, harness_client.py,
                       policy_tests.py, run_eval.py, rescore_history.py
agent-converse/        earlier client-side ReAct loop over Bedrock Converse (superseded, kept for reference)
notebooks/lab3-demo.ipynb  Track C demo, committed with outputs from one clean run
northstar-policy-docs/ policy corpus the agent retrieves from (bundled into the tools Lambda)
docs/                  Lab 2 graded outputs (lab2-*), data contract, lineage diagram
  lab_1/               Lab 1 diagrams, screenshots, ADR, cost estimate, run outputs
  lab_2/               Lab 2 handout and diagram source
  lab_3/               Lab 3 handout; results/ (earlier experiment grid), agent/ (Converse-agent runs)
  lab3-evidence/       Lab 3 graded evidence (see "Where artifacts land")
  lab3-evaluation-report.md  Task 1 evaluation report
  lab3-model-design.md Lab 3 design justification
  lab3-teardown-output.txt   Lab 3 teardown evidence
```

## Lab 2 additions — data and feature engineering

**Network.** A private subnet (`10.0.1.0/24`) with a NAT Gateway (and its Elastic IP in
the public subnet) for outbound-only internet access. The SageMaker domain now runs in the
private subnet with `VpcOnly` networking, and Glue job workers attach there through a Glue
`NETWORK` connection. The shared security group has a self-referencing all-ports ingress
rule, which Glue requires. `enable_nat_gateway = false` skips the NAT for LocalStack.

**Storage.** Five S3 lifecycle rules expire data by prefix. `raw/` objects expire after 90
days and `datacapture/` after 7. Old (noncurrent) versions expire after 30 days under
`raw/` and `processed/`, and after 60 days under `features/`. `force_destroy` is on in
dev because the data is synthetic and regenerable. It would be the wrong setting for real
customer records.

**IAM.** Two new roles alongside MLEngineer:
- `northstar-dev-DataEngineer` is trusted by Glue, Lambda and SageMaker. It runs the crawler
  and both Glue jobs, and it is the Feature Group's execution role. It can read and write
  `raw/`, `processed/` and `features/`, but it can only read `artifacts/`.
- `northstar-dev-ModelMonitor` is read-only on S3 and can write CloudWatch metrics.

**`modules/glue`.** Catalog database `northstar_dev`, the network connection, crawler
`northstar-dev-raw-crawler` over `raw/customers/`, both scripts uploaded to
`artifacts/glue/` on apply (re-uploaded when the file changes), and two Glue 4.0 jobs:
- `northstar-dev-transform` trims, casts, drops null `customer_id`, imputes medians and
  `unknown`, and deduplicates on `transaction_id`. The output keeps transaction grain.
- `northstar-dev-feature-engineer` splits at `FEATURE_CUTOFF` (2026-04-01). It computes 11
  RFM features from the observation window, plus loyalty tier and a rule-based churn risk
  score. `churn_label` comes only from the 90-day outcome window. The job writes one row per
  customer to `features/customers/` and calls `PutRecord` into the Feature Store.

**`modules/feature_store`.** Feature Group `northstar-dev-customer-features` with 16
definitions (`customer_id`, `event_time` Fractional, 13 features, `churn_label` Integral).
The online store is enabled, and the offline store lives under `features/offline-store/`.

The contract for `processed/customers/` is in
[`docs/lab2-data-contract.md`](docs/lab2-data-contract.md), and the end-to-end flow is in
[`docs/lab2-data-lineage.png`](docs/lab2-data-lineage.png).

### Running the data pipeline end to end

```bash
cd infrastructure/environments/dev
terraform apply                                   # creates jobs, crawler, Feature Group
BUCKET=$(terraform output -raw s3_bucket_name)
cd ../../..

aws s3 cp northstar-raw-sample.csv s3://$BUCKET/raw/customers/northstar-raw-sample.csv

aws glue start-crawler --name northstar-dev-raw-crawler
aws glue get-crawler --name northstar-dev-raw-crawler --query 'Crawler.State'   # until READY
aws glue get-table --database-name northstar_dev --name customers

aws glue start-job-run --job-name northstar-dev-transform
aws glue get-job-runs --job-name northstar-dev-transform \
  --query 'JobRuns[0].JobRunState'                                               # until SUCCEEDED

aws glue start-job-run --job-name northstar-dev-feature-engineer
aws glue get-job-runs --job-name northstar-dev-feature-engineer \
  --query 'JobRuns[0].JobRunState'                                               # until SUCCEEDED

bash scripts/verify-lab2.sh            # rubric + data-quality checks
bash scripts/teardown-lab2.sh          # when done: the NAT Gateway bills hourly
```

Both scripts can be tested locally before you pay for a Glue run. Install `pyspark`,
stub the `awsglue` imports, and call `cast_types`, `impute_nulls`, `deduplicate` and the
feature functions on `northstar-raw-sample.csv`.

## Lab 3 — model development

Task 1: churn model (XGBoost on the Feature Store, tracked in a SageMaker MLflow App, registered
as `PendingManualApproval`). Task 2, Track C: a customer service agent on Amazon Bedrock
AgentCore (Harness + Gateway + Cedar policy engine). Design notes: `docs/lab3-model-design.md`.

### Setup

Needs the Lab 1-2 stack up and the Feature Store populated (after any teardown: `terraform
apply`, upload the raw CSV, run the crawler and both Glue jobs; see Lab 2 above). Python 3.12:

```bash
python -m venv .venv && . .venv/Scripts/activate        # Windows Git Bash; use bin/ on macOS/Linux
pip install -r models/churn/requirements.txt -r agent/requirements.txt
aws login
```

`aws login` credentials last 15 minutes and refresh only inside the AWS CLI. boto3 reads them
directly (that is what the `botocore[crt]` extra is for). Terraform's AWS provider does not, and
a one-off `aws configure export-credentials` expires mid-apply. Add a profile that re-runs the
export whenever credentials expire, and use it for Terraform and the scripts:

```ini
# ~/.aws/config
[profile tf]
region = us-east-1
credential_process = aws configure export-credentials --profile default --format process
```

```bash
export AWS_PROFILE=tf
```

### Train and evaluate the churn model

`MLFLOW_APP_ARN` is the ARN of the SageMaker MLflow **App** (serverless, free; never a Tracking
Server, which bills hourly): `aws sagemaker list-mlflow-apps --query 'Summaries[].Arn'`. It was
created once outside Terraform, with the Lab 1 MLEngineer role and an artifact store under
`artifacts/mlflow/`.

```bash
export MLFLOW_APP_ARN=arn:aws:sagemaker:us-east-1:<account>:mlflow-app/<id>
BUCKET=$(terraform -chdir=infrastructure/environments/dev output -raw s3_bucket_name)

bash scripts/run-lab3-experiments.sh                    # experiment grid, each run logged to MLflow

# The graded run: its console log, metrics and plots are the Task 1 evidence.
# PYTHONIOENCODING: on Windows a piped stdout is cp1252, which cannot print the log's ✓/─.
PYTHONIOENCODING=utf8 python models/churn/train.py --artifacts-bucket "$BUCKET" \
  --mlflow-arn "$MLFLOW_APP_ARN" --max-depth 4 --eta 0.1 --run-name xgb-d4-eta0.1-final \
  --expected-customers 9999 --register --model-dir .build/model \
  --output-data-dir docs/lab3-evidence 2>&1 | tee docs/lab3-evidence/train-final.log

python models/churn/compare_models.py --artifacts-bucket "$BUCKET"   # vs logistic regression / MLP
python models/churn/test_feature_query.py                # unit tests, no AWS needed
```

Features come from the Feature Store offline table through Athena (the log prints the
`Athena QueryExecutionId`), deduplicated to the newest record per customer, excluding
soft-deleted customers. The 30% test set is untouched until final scoring; early stopping and
configuration choice use an inner validation split. Gates: precision@top-10% >= 0.50,
recall@top-10% >= 0.25, and a paired-bootstrap 95% CI on AUC lift over the recency-only baseline
that excludes zero. A run that misses a gate is still logged to MLflow but is not registered.

### Deploy and evaluate the agent (Track C)

```bash
cd infrastructure/environments/agent
cp backend.tf.example backend.tf            # then put your account ID in it
terraform init && terraform plan -out=tfplan && terraform apply tfplan
cd ../../..

python agent/policy_tests.py                # calls the gateway directly: the $50 cap, 7 cases
python agent/run_eval.py --runs 3           # 5 scenarios + bonus injection, each in its own session
jupyter nbconvert --to notebook --execute --inplace notebooks/lab3-demo.ipynb
```

The system prompt lives in `infrastructure/environments/agent/system_prompt.md`; change it and
re-apply. Bedrock needs the Anthropic use-case form submitted on the account (see
`docs/bedrock-access-verification.txt`). Three runs of the six scenarios cost about $0.20
estimated: $0.14 of model tokens and about $0.06 of runtime (18 sessions at an estimated
$0.0033 each). Idle sessions are billed, which is why the harness idle timeout stays at 60 s.
`python agent/rescore_history.py` rescores every saved prompt version with the final rules.

### Evidence and submission

```bash
python scripts/collect_lab3_evidence.py --track C   # or: bash scripts/collect-lab3-evidence.sh --track C
bash scripts/teardown-lab3.sh                       # writes docs/lab3-teardown-output.txt
bash scripts/teardown-lab2.sh                       # when finished with the data platform
git tag lab3-submit && git push origin lab3-submit
```

Run the collector while everything is still up: it exports AWS state that teardown deletes.
`collect-lab3-evidence.sh` calls `python3`, so on Windows run the `.py` with the venv's Python
and save its output to `docs/lab3-evidence/collect-output.txt`. On Git Bash, set `MSYS_NO_PATHCONV=1` for any AWS CLI
command whose argument starts with `/`: Git Bash rewrites it into a Windows path. `teardown-lab3.sh`
now does this itself, so it really deletes the harness's conversation log group.

### Where artifacts land

| Artifact | Location |
|---|---|
| Graded training run: console log, metrics, plots, slice table | `docs/lab3-evidence/` (`train-final.log`, `evaluation_metrics.json`, `confusion-matrix.png`, `feature-importance.png`, `slice-evaluation.csv`) |
| Account identity, Athena query, MLflow App and runs, Model Registry, AgentCore config (exported) | `docs/lab3-evidence/` (`account.json`, `athena-query.json`, `mlflow-app.json`, `mlflow-runs.csv`, `model-registry.json`, `agentcore-config.json`, `collect-output.txt`) |
| Model package | Model Registry group `northstar-churn-models` (`PendingManualApproval`); artifact under `s3://<bucket>/artifacts/models/churn/` (deleted by teardown) |
| Agent evaluation | `docs/lab3-evidence/` (`agent_eval.csv`, `agent-traces.jsonl`, `agent-cost.csv`, `policy-tests.json`, `track-c-failures.md`; runs of the earlier prompt versions in `history/v1_*`, `v2_*`, `v3_*`, every version's `*_system_prompt.md` (v4 is final), and `history/rescored-summary.csv`) |
| Reports | `docs/lab3-evaluation-report.md` (Task 1), `docs/lab3-model-design.md` (Task 3) |
| Earlier experiment grid and model comparison | `docs/lab_3/results/` |
| Teardown evidence | `docs/lab3-teardown-output.txt` |

The MLflow App and the Model Registry group are kept (free; later labs use them). Teardown
deletes the S3 bucket, so the registered model's artifact is gone until the next retrain.

## Usage

```bash
bash scripts/bootstrap-state.sh          # once per account
cd infrastructure/environments/dev
terraform init && terraform plan && terraform apply
bash ../../../scripts/verify-lab1.sh     # from repo root, while the stack is up
terraform destroy                        # don't leave it running
make local-validate                      # LocalStack check, costs nothing
```

## Labs

| Lab | Scope | Tag |
|---|---|---|
| 1 | Platform foundation — VPC, S3, IAM, SageMaker Domain, Terraform IaC | `lab1-submit` |
| 2 | Data and feature engineering — private subnet + NAT, Glue pipeline, Feature Store, data contract | `lab2-submit` |
| 3 | Model development — XGBoost churn model with MLflow and registry; AgentCore customer service agent | `lab3-submit` |

Never commit credentials. `.tfvars`, `.tfstate`, and `.env` are gitignored; keep it
that way.
