# Lab 4 — XOps Assessment

## Champion-Challenger

The champion is the newest model package in `northstar-churn-models` with status
`Approved` (v4 at submission, AUC 0.7760). A challenger is the package a CI run has just
registered as `PendingManualApproval`.

**Rule: the challenger replaces the champion if and only if both of these hold.**

1. **It clears the absolute gates**, enforced today by `tests/test_model.py`:
   precision@top10% ≥ 0.50, recall@top10% ≥ 0.25, and the paired-bootstrap 95% CI on
   (challenger AUC − recency-only baseline AUC) has a lower bound above 0.
2. **It beats the champion with statistical confidence:** the paired-bootstrap 95% CI on
   (challenger AUC − champion AUC), computed on the same held-out rows with 2,000
   resamples and seed 42, has a **lower bound > 0**.

There is no fixed "better by 0.01" margin on purpose. Lab 4's own history retired a fixed
lift threshold because it was smaller than the metric's run-to-run noise. A confidence
interval adapts to the test-set size and the models' correlation. The check is binary: the
lower bound is above zero or it is not.

Separately, `test_new_model_does_not_regress_against_champion` blocks a candidate from
being promoted if its AUC is below the champion's by more than **0.02**. That is a
*do-no-harm* floor, not the promotion criterion.

**Enforced vs designed.** Condition 1 and the 0.02 floor run in every build. Condition 2 is
designed but not yet automated. Today the human at the `ManualApproval` stage applies it.
`bootstrap_lift_ci()` in `models/churn/churn_model.py` already computes a paired AUC-difference
interval, so automating condition 2 means passing the champion's scores in place of the
baseline's.

**Worked example from this lab.** v5 (final green run) logged
`champion v4 auc=0.7760; candidate auc=0.7760` in `docs/lab4-build-green.log`. The AUC
difference is zero, so its interval contains zero and v5 does **not** replace v4. v4 stays the
champion and v5 stays `PendingManualApproval` (`docs/lab4-model-registry.json`).

## Retraining Triggers

Both triggers start the same thing: the `northstar-pipeline` CodePipeline, which retrains
through the SageMaker Pipeline and registers a challenger judged by the rule above. Neither
needs a human to decide to retrain. Both are designed, not deployed.

**(a) Scheduled: weekly.**

| | |
|---|---|
| Schedule | `cron(0 6 ? * MON *)`, Mondays 06:00 UTC |
| Fires it | Amazon EventBridge Scheduler, target `codepipeline:StartPipelineExecution` on `northstar-pipeline` |
| Precondition | An AWS Glue scheduled trigger, `cron(0 5 ? * MON *)`, runs `northstar-dev-feature-engineer` an hour earlier |

The Glue trigger is part of the design because `tests/test_data.py` fails any feature set
whose newest `event_time` is more than 48 hours old. A scheduled retrain on stale features
would fail the freshness test by construction. Weekly cadence with a refresh the same morning
keeps data under 48 hours every run.

**(b) Performance-based: production AUC drops.**

| | |
|---|---|
| Threshold | Rolling 30-day production AUC **< 0.726** (champion AUC 0.7760 − 0.05) on 2 consecutive daily evaluations |
| Measured by | SageMaker Model Monitor model-quality schedule, comparing logged predictions to churn labels as they mature, publishing `auc` to CloudWatch |
| Fires it | A CloudWatch alarm on that metric, then an EventBridge rule on the alarm's `ALARM` state change, target `codepipeline:StartPipelineExecution` |

The threshold is tied to numbers in this repo. The recency-only baseline scores about 0.72
(`baseline_auc` 0.7206 in the v5 metadata), so 0.726 is the point where the model has lost
nearly all its advantage. A drop of 0.05 is more than the width of the lift interval in the
gate (0.036 to 0.076), so ordinary noise does not trigger it. Requiring two consecutive days
keeps a single bad day from retraining.

**Known limit.** Churn labels need a 90-day window to mature, so this trigger lags by
months. A faster proxy trigger, population stability index above 0.2 on
`days_since_last_purchase` from Model Monitor's data-quality schedule, would catch input drift
sooner. It would start the same pipeline.

## Experiment Tracking and Lineage

- **MLflow runs:** `docs/lab4-mlflow-runs.txt`. Three runs in experiment
  `northstar-churn-lab4` on the MLflow App `northstar-mlflow`
  (`arn:aws:sagemaker:us-east-1:767397742230:mlflow-app/app-3PIYYOBQRQWE`), varying
  `max_depth` and `eta` (6/0.10, 4/0.05, 8/0.20), all with metrics logged. These ran
  Lab 3's `models/churn/train.py`. They do not include the monotone recency constraint
  that CI's `train()` has.
- **Model lineage:** `docs/lab4-model-package.json`. Package v5's
  `CustomerMetadataProperties` hold the training data URI and timestamp, the commit SHA
  (`15cf378444422ebbacc2b8c6c5ae86b8377b8e35`, in `main` history), and the evaluation metrics.

## Pipeline Stage Mapping

The five required phases run in three CodePipeline stages (`pipeline.yaml`). CodeBuild runs one
project, `northstar-ci`, using `buildspec.yml`.

| Phase | CodePipeline stage | Where it runs |
|---|---|---|
| 1. Source | `Source` | CodeStar connection to GitHub `main`; `DetectChanges: true` starts a run on every push |
| 2. Test | `Build` | `buildspec.yml` `pre_build`: `pytest tests/test_data.py` and `tests/test_features.py`. A failure ends the build here, and `BUILD` never starts |
| 3. Build | `Build` | `buildspec.yml` `build`: upserts and runs the SageMaker Pipeline; `TrainChurnModel` runs `churn_model.train()` on spot capacity |
| 4. Evaluate | `Build` | `train()` raises if a quality gate fails; then `build` runs `tests/test_model.py` and `tests/test_fairness.py` against the emitted metrics |
| 5. Register | `Build` | SageMaker Pipeline step `RegisterChurnModel` creates the package as `PendingManualApproval` |
| (human decision) | `ManualApproval` | SNS-notified approval action; the approver sets the package to Approved or Rejected |

**Rationale.** A CodePipeline stage is a hand-off boundary between artifacts, or a place to
insert a person. Phases 2 to 5 pass no artifact between each other, so giving each its own
stage would mean extra CodeBuild projects and copying the model between them for no gain. Only
the approval is a real boundary, so it gets its own stage. I used CodePipeline and CodeBuild
because the SageMaker, CloudWatch and SNS permissions are plain IAM roles in the same account
and stack, with no outside credentials to manage.

**One ordering detail.** Registration happens before the CodeBuild-only gate. A package that
passes training but fails a later test is already in the registry, so `post_build` sets it to
`Rejected`. Evidence: v3 in `docs/lab4-model-registry.json` was registered by build
`472e99d4` (commit `e692189`), which then failed `test_model_monotone_on_recency`; its
`northstar-ci:472e99d4-…` entry is in `docs/lab4-pipeline-runs.txt`.

## DataOps Maturity

**Current level: 1 (Repeatable)**

**Evidence:** The data pipeline is defined as code and repeatable: the Glue crawler and both ETL
jobs in `infrastructure/modules/glue/main.tf`, with logic in `glue-scripts/transform.py` and
`glue-scripts/feature_engineer.py`. Validation is automated on every push:
`tests/test_data.py` covers schema, grain, leakage, bounds and freshness, and
`docs/lab4-build-green.log` shows 29 PASSED lines for it. `scripts/data-bucket.sh` keeps the
data bucket across teardowns.

It is not Level 2 because nothing triggers the pipeline. The Glue module's own comment reads
"On-demand only (no schedule)", and a person starts the chain with `aws glue start-job-run`.
Forgetting to means `test_data.py` fails the freshness check once features pass 48 hours.

**Gap to Level 2:** No automatic trigger for the crawler → transform → feature-engineer chain.
Staleness is caught only after the fact, when a CI build fails. A manual re-run on 2026-10-30
also exposed an IAM gap that only an overwrite triggers (`features_$folder$`, noted in
`infrastructure/modules/iam/main.tf`), which a regular schedule would have surfaced earlier.

**Top priority investment:** Add an AWS Glue workflow with a scheduled trigger
(`aws_glue_trigger`, `cron(0 5 ? * MON *)`) in `infrastructure/modules/glue/`, so features are
refreshed weekly without anyone remembering.

## MLOps Maturity

**Current level: 2 (Automated)**

**Evidence:** A push to `main` triggers the whole path with no manual step:
`pipeline.yaml` and `buildspec.yml` run test, train, gate and register.
`docs/lab4-pipeline-runs.txt` lists the runs. The gates are enforced in code:
`docs/lab4-build-green.log` shows 26 passed in the model gate, including the regression test
(`champion v4 auc=0.7760`) and the fairness flag (32.0-point recall gap). The failure path is
measured, not claimed: `docs/lab4-failure-demo.txt` shows `PRE_BUILD FAILED`, the registry
unchanged at v4, and the alarm going to ALARM and back to OK, with the email in
`docs/lab4-alarm-email.png`. `docs/lab4-model-package.json` records lineage for v5.

**Gap to Level 3:** Retraining is not automated on any drift signal. No Model Monitor schedule
is deployed, so nothing but a git push can start a run. The triggers above are designed, not
built. Promotion also still depends on a person applying the champion-challenger interval at
`ManualApproval`.

**Top priority investment:** Deploy a SageMaker Model Monitor data-quality schedule baselined
on `features/customers/`, with a CloudWatch alarm and an EventBridge rule that starts
`northstar-pipeline` (Lab 6).
