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
  run-lab3-experiments.sh  Lab 3 churn experiment grid (4 MLflow runs)
  teardown-lab3.sh     Lab 3 teardown: endpoints, endpoint configs, jobs, tracking servers
  teardown-lab2.sh     full Lab 2 teardown, incl. resources Terraform can't see
  check-secrets.sh     scans all git history for committed credentials
models/churn/
  train.py             Lab 3 churn training: Athena -> XGBoost -> evaluation -> MLflow -> registry
  compare_models.py    XGBoost vs logistic regression vs MLP on the same split
  requirements.txt     training dependencies
evaluation/
  evaluation_harness.py  course harness (Track C scoring via LocalAgentEvaluator)
agent/                 Lab 3 Track C customer service agent (ReAct loop over Bedrock Converse)
northstar-policy-docs/ policy corpus the agent retrieves from
docs/                  Lab 2 graded outputs (lab2-*), data contract, lineage diagram
  lab_1/               Lab 1 diagrams, screenshots, ADR, cost estimate, run outputs
  lab_2/               Lab 2 handout and diagram source
  lab_3/               Lab 3 handout; results/ (MLflow export, registry, per-run metrics), agent/ (traces)
  lab3-model-design.md Lab 3 design justification
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

Track C: churn model (XGBoost) plus a customer service agent on Bedrock.

### Setup

Needs the Lab 2 stack up and the Feature Store populated (re-run the feature-engineering
Glue job after any teardown). Python 3.12:

```bash
python -m venv .venv && . .venv/Scripts/activate        # Windows Git Bash; use bin/ on macOS/Linux
pip install -r models/churn/requirements.txt -r agent/requirements.txt
aws login                                               # then:
eval "$(aws configure export-credentials --format env)" # exported creds are short-lived; re-run if they expire
```

### Train and evaluate the churn model

`MLFLOW_APP_ARN` is the ARN of your SageMaker MLflow **App** (serverless, free; never a
Tracking Server, which bills hourly). It is created outside Terraform.

```bash
export MLFLOW_APP_ARN=arn:aws:sagemaker:...            # your MLflow App
bash scripts/run-lab3-experiments.sh                    # 4 configs, each logged to MLflow

# or one run by hand (all paths and hyperparameters are arguments):
python models/churn/train.py \
  --artifacts-bucket "$(cd infrastructure/environments/dev && terraform output -raw s3_bucket_name)" \
  --mlflow-arn "$MLFLOW_APP_ARN" --max-depth 4 --eta 0.1 --run-name my-run
#   add --register to register the model (PendingManualApproval) if the gates pass
python models/churn/compare_models.py --artifacts-bucket <bucket>   # vs logistic regression / MLP
```

Features are read from the Feature Store offline table through Athena, deduplicated to the
newest record per customer. The 30% test set is untouched until final scoring; early
stopping and configuration choice use an inner validation split. Gates: precision@top-10%
>= 0.50, recall@top-10% >= 0.25, and a paired-bootstrap 95% CI on AUC lift over the
recency-only baseline that excludes zero. Gate failures are logged, not raised.

### Run the agent

```bash
python agent/run_eval.py                                # 5 required scenarios + bonus injection
jupyter nbconvert --to notebook --execute agent/demo.ipynb   # demo notebook
```

Bedrock needs the Anthropic use-case form submitted on the account (model ID
`us.anthropic.claude-haiku-4-5-20251001-v1:0`); see `docs/bedrock-access-verification.txt`.
A full six-scenario run costs about $0.04.

### Where artifacts land

| Artifact | Location |
|---|---|
| Per-run metrics JSON + feature-importance plot | `docs/lab_3/results/<run-name>/` (kept in git; teardown deletes the S3 copies) |
| MLflow runs and registry export | `docs/lab_3/results/mlflow-search-runs.csv`, `model-registry.json` |
| Model package | SageMaker Model Registry group `northstar-churn-models` (PendingManualApproval); artifact under `s3://<bucket>/artifacts/models/churn/` |
| Model comparison | `docs/lab_3/results/model-comparison.json` |
| Agent scores and traces | `docs/lab_3/agent/` (`agent_eval.csv`, `agent_traces.json`, `README.md` with failure modes) |
| Design justification | `docs/lab3-model-design.md` |

### Teardown

```bash
bash scripts/teardown-lab3.sh        # writes docs/lab3-teardown-output.txt
bash scripts/teardown-lab2.sh        # when finished with the stack
```

The MLflow App is kept (free; later labs log to it). Teardown deletes the model artifact in
S3, so retrain after the next `terraform apply`.

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
| 3 | Model development — XGBoost churn model with MLflow and registry; Bedrock customer service agent | `lab3-submit` |

Never commit credentials. `.tfvars`, `.tfstate`, and `.env` are gitignored; keep it
that way.
