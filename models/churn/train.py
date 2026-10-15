"""
NorthStar Retail — Churn Prediction Training Script (Skeleton)
CS 401R Lab 3 Starter Kit

This script is the SageMaker XGBoost training entry point for the NorthStar
churn prediction model. It reads features from the SageMaker Feature Store,
trains an XGBoost classifier, tracks the run in a SageMaker MLflow App, and
registers the model in the Model Registry.

Experiment tracking needs two packages that are NOT in the training container:

    pip install mlflow sagemaker-mlflow

`sagemaker-mlflow` is the SigV4 auth plugin for arn:aws:sagemaker:... tracking
URIs. Install `mlflow` alone and you get a connection error that never mentions
credentials. See track_run() for the cost warning about the OTHER MLflow
product -- read it before you create anything.

Usage (local testing):
    python models/churn/train.py \
        --feature-group-name northstar-dev-customer-features \
        --artifacts-bucket northstar-dev-data-<account> \
        --max-depth 6 --eta 0.1 --num-round 200 \
        --mlflow-arn arn:aws:sagemaker:us-east-1:<account>:mlflow-app/app-XXXX \
        --run-name xgb-depth6-eta0.1

This lab trains locally (see models/churn/requirements.txt); a SageMaker Training
Job is optional. Evidence files (evaluation_metrics.json, feature-importance.png,
confusion-matrix.png, slice-evaluation.csv)
are written to --output-data-dir, because teardown deletes the S3 copies.
"""

import argparse
import csv
import hashlib
import json
import os
import tarfile
import time
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd

# No `sagemaker` SDK import. The skeleton used FeatureGroup.athena_query(), but
# sagemaker 3.x removed the feature_store package, and all that call does is look
# up the offline-store table and run SQL. boto3 does the same two things
# directly (describe_feature_group + Athena), with one fewer fragile dependency.

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Argument Parsing ───────────────────────────────────────────────────────────
# SageMaker passes hyperparameters as CLI arguments.

def parse_args():
    parser = argparse.ArgumentParser(description="NorthStar churn prediction training")

    # Data configuration
    # Names come from Lab 2's Terraform: `terraform output -raw feature_group_name`
    # and `terraform output -raw s3_bucket_name`. Do not invent new ones.
    parser.add_argument("--feature-group-name", type=str,
                        default="northstar-dev-customer-features",
                        help="SageMaker Feature Store feature group name (Lab 2 output)")
    # These dates bound the INGEST batch, not the feature window. Lab 2 stamps
    # event_time with the wall-clock time its Glue job ran (time.time()), not
    # with the feature cutoff T = 2026-04-01 -- T was already enforced when the
    # features were computed. A window that ends before your Lab 2 run (the old
    # default here was 2026-06-01) matches zero rows. These defaults match the
    # reference implementation and cover any Lab 2 run during the semester.
    parser.add_argument("--training-start-date", type=str, default="2026-01-01",
                        help="Earliest Lab 2 ingest date to include (YYYY-MM-DD)")
    parser.add_argument("--training-end-date", type=str, default="2027-01-01",
                        help="Latest Lab 2 ingest date to include (YYYY-MM-DD)")
    parser.add_argument("--artifacts-bucket", type=str, required=True,
                        help="Your Lab 1 data bucket, northstar-dev-data-<account>. "
                             "Athena results and model artifacts go under its artifacts/ "
                             "prefix, the only prefix your MLEngineer role can write")
    parser.add_argument("--expected-customers", type=int, default=None,
                        help="Fail if the load returns a different number of customers. Guards "
                             "against a re-ingest or window change silently altering the data")
    parser.add_argument("--athena-timeout", type=int, default=ATHENA_TIMEOUT_S,
                        help="Seconds to wait for the Athena feature query before stopping it")
    parser.add_argument("--local-data-path", type=str, default=None,
                        help="Path to local feature CSV for development testing (bypasses Feature Store)")

    # XGBoost hyperparameters
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--num-round", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=20,
                        help="Stop when the inner validation AUC has not improved for this many rounds")
    parser.add_argument("--verbose-eval", type=int, default=25,
                        help="Print training metrics every N rounds")
    parser.add_argument("--min-child-weight", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--scale-pos-weight", type=float, default=None,
                        help=(
                            "XGBoost positive-class weight. Leave unset and it is computed "
                            "from YOUR training split as negatives/positives, which is what "
                            "you want. The reference 10,000-customer dataset is 22.0%% "
                            "churners, giving (1 - 0.220) / 0.220 = 3.545. Passing a value "
                            "tuned for a different class balance will quietly skew your "
                            "precision/recall trade-off, so only override this deliberately."
                        ))
    # Experiment tracking (Task 1, 5 points)
    parser.add_argument("--mlflow-arn", type=str,
                        default=os.environ.get("MLFLOW_APP_ARN"),
                        help=(
                            "ARN of your SageMaker MLflow APP, e.g. "
                            "arn:aws:sagemaker:us-east-1:<account>:mlflow-app/app-XXXX. "
                            "Create it once with `aws sagemaker create-mlflow-app`. "
                            "This is NOT an MLflow Tracking Server -- see the warning "
                            "on track_run() below. Omit to skip tracking."
                        ))
    parser.add_argument("--mlflow-experiment", type=str, default="northstar-churn",
                        help="MLflow experiment name; runs group under this")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run. Make it descriptive -- 'run3' tells "
                             "you nothing three weeks later.")

    # Leakage control (handout: "report results with and without it").
    parser.add_argument("--include-risk-score", action="store_true",
                        help="Add churn_risk_score (a pure recency heuristic) to the "
                             "feature set. Run once with and once without, and report both.")
    parser.add_argument("--inner-val-size", type=float, default=0.15,
                        help="Fraction of the TRAINING rows held out for early stopping. "
                             "The 30%% test set is never used to choose a stopping round.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Split + model seed. Keep it fixed when comparing "
                             "hyperparameters, or you are comparing seeds.")

    # Registry (Task 1, 5 points). Off by default so experiment runs do not each
    # add a version -- pass --register on the one final run.
    parser.add_argument("--register", action="store_true",
                        help="Register the model as PendingManualApproval if every "
                             "gate passes. Never auto-approves.")
    parser.add_argument("--model-package-group", type=str,
                        default="northstar-churn-models",
                        help="Lab 5 deploys from this group by name; do not rename it")

    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--output-data-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))

    return parser.parse_args()


# ── Feature Store Loading ──────────────────────────────────────────────────────

# These are exactly the features your Lab 2 pipeline wrote to the Feature
# Group. If a name here does not match your feature group, the Athena query
# below will fail - check `aws sagemaker describe-feature-group` first.
#
# Note what is NOT in this list: churn_label (the target) and customer_id
# (an identifier, not a signal). Including either as an input is leakage.
FEATURE_COLUMNS = [
    # Recency and tenure
    "days_since_last_purchase",
    "customer_tenure_days",
    # Frequency
    "purchase_frequency_30d",
    "purchase_frequency_90d",
    "purchase_frequency_180d",
    # Monetary
    "avg_order_value",
    "total_spend_90d",
    "total_lifetime_value",
    "avg_basket_size_6m",
    # Behavioural - these are the features that catch churners who still
    # look active on recency alone. Drop them and your model collapses
    # toward the recency-only baseline.
    "category_diversity_score",
    "online_to_store_ratio",
]

# churn_risk_score is available in the feature group but deliberately excluded
# here: it is a pure recency heuristic and doubles as the baseline you must
# beat. If you add it, report your metrics both with and without it.
BASELINE_COLUMN = "churn_risk_score"

# The recency-only baseline model uses just this one feature. Task 1 requires
# you to train it and show your full model beats it by a margin whose 95%
# confidence interval excludes zero. There is no fixed AUC-lift threshold: a
# ">= 0.03 lift" gate lived here until 2026-08-02 and was removed because 0.03
# is smaller than the metric's own run-to-run standard deviation, so it failed
# on 21% of splits regardless of model quality. See EVAL_THRESHOLDS below.
BASELINE_FEATURE = "days_since_last_purchase"

LABEL_COLUMN = "churn_label"

# Not a feature. This is the column Task 1's slice evaluation groups by, so it
# has to come back from the query even though it never enters the model. Keep it
# out of FEATURE_COLUMNS: loyalty_tier correlates with spend, and feeding it in
# both leaks and makes the per-tier fairness check circular.
SLICE_COLUMN = "loyalty_tier"

# The four tiers in the Lab 2 data contract. A value outside this set would create a
# bogus slice (or, if null, silently drop rows from slice evaluation).
LOYALTY_TIERS = ("Bronze", "Silver", "Gold", "Platinum")

# Seconds to wait for the Athena feature query before stopping it and failing.
ATHENA_TIMEOUT_S = 300


def resolve_feature_columns(include_risk_score: bool) -> list[str]:
    """The model's input columns. churn_risk_score is opt-in (see BASELINE_COLUMN)."""
    return FEATURE_COLUMNS + ([BASELINE_COLUMN] if include_risk_score else [])


def build_feature_query(table: str,
                        feature_columns: list[str],
                        start_epoch: float,
                        end_epoch: float) -> str:
    """SQL that returns one row per customer: the newest record, unless that record
    is a soft-delete marker, in which case the customer is excluded.

    Ranking happens BEFORE the is_deleted filter. A soft delete is a new row with
    is_deleted = true and the newest event_time, so filtering first would drop the
    marker and promote the customer's older live row, resurrecting a deleted
    customer into the training set. Kept separate from the AWS calls so it can be
    unit tested (see test_feature_query.py).
    """
    cols = ", ".join(feature_columns)
    return f"""
        WITH ranked AS (
            SELECT
                customer_id,
                {cols},
                {LABEL_COLUMN},
                {SLICE_COLUMN},
                is_deleted,
                ROW_NUMBER() OVER (
                    PARTITION BY customer_id
                    ORDER BY event_time DESC, write_time DESC
                ) AS rn
            FROM {table}
            WHERE event_time BETWEEN {start_epoch} AND {end_epoch}
        )
        SELECT
            customer_id,
            {cols},
            {LABEL_COLUMN},
            {SLICE_COLUMN}
        FROM ranked
        WHERE rn = 1
          AND NOT is_deleted
        ORDER BY customer_id
    """


def _wait_for_athena(athena, qid: str, timeout_s: int) -> dict:
    """Poll until the query finishes. Stops the query and raises on timeout."""
    deadline = time.monotonic() + timeout_s
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if status["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return status
        if time.monotonic() > deadline:
            athena.stop_query_execution(QueryExecutionId=qid)
            raise TimeoutError(f"Athena query {qid} still {status['State']} after {timeout_s}s; "
                               f"stopped it. Check Athena query history for the cause.")
        time.sleep(1)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of the loaded rows (already ordered by customer_id), so a run records
    exactly which data it saw even if the Feature Store is re-ingested later."""
    row_hashes = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False)
    return hashlib.sha256(row_hashes.values.tobytes()).hexdigest()


def load_features_from_feature_store(feature_group_name: str,
                                     start_date: str,
                                     end_date: str,
                                     artifacts_bucket: str,
                                     feature_columns: list[str],
                                     timeout_s: int = ATHENA_TIMEOUT_S) -> pd.DataFrame:
    """
    Load features from the SageMaker Feature Store offline store using Athena.

    Runs under the caller's own AWS credentials (not MLEngineer, which has no
    Athena or Glue-catalog access by design). Athena writes its result set to
    s3://<artifacts_bucket>/artifacts/athena-results/.

    event_time is Fractional (epoch seconds), so the ISO date bounds are
    converted before comparison; ISO strings fail with TYPE_MISMATCH.
    The offline store is append-only and soft-deleting; build_feature_query keeps
    the newest record per customer and excludes customers whose newest record is a
    delete marker. It ORDERs BY customer_id because Athena returns rows in
    split-completion order, and train_test_split is only deterministic for a given
    row order.

    The returned frame carries df.attrs["athena_query_id"] and
    df.attrs["data_sha256"] so the run can log which data it used.
    """
    start_epoch = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    end_epoch = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()

    sm = boto3.client("sagemaker")
    athena = boto3.client("athena")

    # The offline table name carries a numeric suffix (..._1791746975) that
    # changes on every rebuild, so look it up instead of hardcoding it.
    cfg = sm.describe_feature_group(FeatureGroupName=feature_group_name)
    catalog = cfg["OfflineStoreConfig"]["DataCatalogConfig"]
    table = f'"{catalog["Database"]}"."{catalog["TableName"]}"'

    query_string = build_feature_query(table, feature_columns, start_epoch, end_epoch)

    qid = athena.start_query_execution(
        QueryString=query_string,
        ResultConfiguration={
            "OutputLocation": f"s3://{artifacts_bucket}/artifacts/athena-results/"},
    )["QueryExecutionId"]
    # The evidence script (collect-lab3-evidence.sh) finds this exact line in
    # train-final.log and looks the query up in Athena, so keep the format.
    print(f"Athena QueryExecutionId: {qid}")
    print(f"  table: {table}")

    status = _wait_for_athena(athena, qid, timeout_s)
    if status["State"] != "SUCCEEDED":
        raise RuntimeError(f"Athena query {status['State']}: "
                           f"{status.get('StateChangeReason', 'no reason given')}")

    rows, token = [], None
    while True:
        kwargs = {"QueryExecutionId": qid, "MaxResults": 1000}
        if token:
            kwargs["NextToken"] = token
        page = athena.get_query_results(**kwargs)
        rows.extend([[c.get("VarCharValue") for c in r["Data"]]
                     for r in page["ResultSet"]["Rows"]])
        token = page.get("NextToken")
        if not token:
            break
    header, body = rows[0], rows[1:]
    df = pd.DataFrame(body, columns=header)

    if df.empty:
        raise RuntimeError(
            f"Athena query {qid} returned 0 customers. --training-start-date/--training-end-date "
            f"bound the Lab 2 INGEST time ({start_date} to {end_date}), not the feature window; "
            f"a window that ends before your Lab 2 run matches nothing. Also confirm the feature "
            f"group was populated (re-run the feature-engineering Glue job after a teardown).")

    # Athena returns everything as text; restore real types. errors="coerce" would turn
    # an unparseable value into NaN silently, so count NaNs before and after: Athena
    # NULLs are already missing and are reported later; any NEW NaN is a parse failure.
    numeric = feature_columns + [LABEL_COLUMN]
    nan_before = int(df[numeric].isna().sum().sum())
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    unparseable = int(df[numeric].isna().sum().sum()) - nan_before
    if unparseable:
        bad = [c for c in numeric if df[c].isna().sum() > 0]
        raise ValueError(f"{unparseable} Athena values could not be parsed as numbers "
                         f"(columns with missing values: {bad}); refusing to train on silently "
                         f"coerced data.")

    df.attrs["athena_query_id"] = qid
    df.attrs["data_sha256"] = dataset_fingerprint(df)
    print(f"  loaded {len(df):,} customers, {df['customer_id'].nunique():,} distinct")
    print(f"  data sha256: {df.attrs['data_sha256'][:16]}...")
    return df


def load_features_local(path: str, feature_columns: list[str]) -> pd.DataFrame:
    """Load features from a local CSV file (development mode)."""
    print(f"Loading features from local file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in feature_columns + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Local data missing required columns: {missing}")
    # Not fatal - the model trains fine without it - but Task 1's slice
    # evaluation cannot be done at all, so fail the task rather than the run.
    if SLICE_COLUMN not in df.columns:
        print(f"WARNING: '{SLICE_COLUMN}' not in local data — Task 1 slice "
              f"evaluation will not be possible with this file")
    df.attrs["athena_query_id"] = "local-csv"
    df.attrs["data_sha256"] = dataset_fingerprint(df)
    return df


# ── Feature Preprocessing ──────────────────────────────────────────────────────

def preprocess_features(df: pd.DataFrame,
                        feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Clean and prepare features for XGBoost training.
    Returns (X, y, df) where df is the cleaned frame, row-aligned with X and y
    so the loyalty tier and customer_id can follow each row through the split.

    Missing values: Lab 2's own verification found zero nulls in the feature
    table, and we report the count here rather than assume it. NaN is passed
    through untouched -- XGBoost learns a default branch direction for missing
    values -- instead of the skeleton's fillna(-1), which would have silently
    turned "missing" into a real value (-1 days since last purchase).
    """
    print(f"Preprocessing {len(df):,} samples...")

    n_before = len(df)
    df = df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
    n_unlabeled = n_before - len(df)
    if n_unlabeled:
        print(f"  WARNING: dropped {n_unlabeled:,} rows with no {LABEL_COLUMN}")

    # astype(int) below would silently truncate a label like 0.7 to 0.
    bad_labels = sorted(set(df[LABEL_COLUMN].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"{LABEL_COLUMN} must be 0 or 1; found {bad_labels}")

    # Slice evaluation groups by tier, so an unexpected or null tier must not pass quietly.
    # (A local CSV may lack the column; load_features_local already warns about that.)
    if SLICE_COLUMN in df.columns:
        bad_tiers = df[SLICE_COLUMN][~df[SLICE_COLUMN].isin(LOYALTY_TIERS)]
        if len(bad_tiers):
            raise ValueError(f"{len(bad_tiers):,} rows have a {SLICE_COLUMN} outside "
                             f"{list(LOYALTY_TIERS)}: {sorted(bad_tiers.astype(str).unique())}")

    n_missing = int(df[feature_columns].isna().sum().sum())
    print(f"  Missing feature cells: {n_missing}"
          + ("" if n_missing == 0 else "  (left as NaN for XGBoost)"))

    X = df[feature_columns].values.astype(float)
    y = df[LABEL_COLUMN].values.astype(int)

    print(f"  Positive class rate: {y.mean():.1%}")
    print(f"  Feature matrix shape: {X.shape}")

    return X, y, df


# ── Model Training ─────────────────────────────────────────────────────────────

def predict_proba(model: xgb.Booster, X: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    """Score with the best early-stopping round, stated explicitly."""
    best = getattr(model, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    return model.predict(xgb.DMatrix(X, feature_names=feature_columns), iteration_range=rng)


def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                X_stop: np.ndarray,
                y_stop: np.ndarray,
                args,
                feature_columns: list[str]) -> xgb.Booster:
    """Train the XGBoost churn model.

    (X_stop, y_stop) is an inner validation set carved out of the TRAINING rows
    and used only to pick the early-stopping round. The 30% test set is never
    seen here, so the test metrics are not tuned to it.
    """

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dstop = xgb.DMatrix(X_stop, label=y_stop, feature_names=feature_columns)

    # Class imbalance, computed from the training split rather than hardcoded.
    # negatives/positives is the XGBoost-documented starting point: it makes the
    # total weight of each class equal, so the loss is not dominated by the
    # 78% majority. On the reference dataset this lands at 3.545.
    spw = (args.scale_pos_weight if args.scale_pos_weight is not None
           else float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))

    # Record the RESOLVED value so track_run() can log what was actually used.
    # Logging args.scale_pos_weight instead would log None on every run that
    # let it compute itself -- which is most of them.
    args.resolved_scale_pos_weight = spw

    params = {
        # eval_metric ORDER MATTERS. XGBoost early-stops on the LAST metric in
        # this list, not the first. Keep "auc" last (defect 49): with logloss
        # last, scale_pos_weight keeps "improving" logloss long after ranking
        # quality peaks, and training runs all rounds.
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": args.max_depth,
        "eta": args.eta,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": spw,
        "seed": args.seed,
        "verbosity": 1,
    }

    print(f"\nTraining XGBoost with params: {json.dumps(params, indent=2)}")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dstop, "stop")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )
    print(f"  best iteration: {model.best_iteration}")

    # Early stopping halts early_stopping_rounds AFTER the best round, so the booster still
    # holds that many extra trees, and a plain predict() (e.g. in the SageMaker container)
    # would use them and score differently from what we evaluated. Keep only the best trees.
    # Slicing drops best_iteration / best_score, which main() and MLflow read, so restore them.
    best_iteration, best_score = model.best_iteration, model.best_score
    model = model[: best_iteration + 1]
    model.set_attr(best_iteration=str(best_iteration), best_score=str(best_score))
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

EVAL_THRESHOLDS = {
    "precision_top10":   0.50,   # Minimum precision @ top 10% scored customers
    "recall_top10":      0.25,   # Minimum recall @ top 10% scored customers
    # Note the recall ceiling: with ~22% positives, targeting the top 10% of
    # customers caps achievable recall near 0.45. Hitting 0.25 means capturing
    # roughly half of what is reachable within that contact budget.
}

# There is deliberately NO absolute AUC threshold. One used to live here
# (auc_roc >= 0.72) and it was removed on 2026-08-02: measured across 200
# random train/test splits of the same data, the reference model fell below
# 0.72 on 58% of them. A gate the reference clears by luck of the shuffle
# grades your random seed, not your model. Report AUC; do not gate on it.
#
# The baseline gate is an INTERVAL, not a threshold: the 95% CI on
# (your AUC - recency-only baseline AUC) must exclude zero.
# BASELINE_FEATURE is defined once, near FEATURE_COLUMNS above.
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

# A slice with fewer positives than this cannot support a claim about its AUC or
# recall. The handout's cautionary tale: ~2 positives swung a tier's AUC between
# 0.00 and 1.00 on nothing but the split.
MIN_SLICE_POSITIVES = 30
SLICE_BOOTSTRAP_N = 500


def train_recency_baseline(X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           feature_columns: list[str],
                           seed: int) -> np.ndarray:
    """Train the recency-only baseline and score the validation split.

    The baseline is a model over `days_since_last_purchase` ALONE, trained on
    the same training rows and scored on the same test rows as the full model.
    That is what makes the comparison fair.

    It is deliberately NOT a constant predictor. A constant prediction has an
    AUC of exactly 0.5 by construction, so "beating" it proves only that your
    model is better than a coin flip. The question Lab 3 asks is whether your
    feature engineering beats the rule the business already has for free:
    "contact whoever has not purchased in a while."
    """
    i = feature_columns.index(BASELINE_FEATURE)
    dtr = xgb.DMatrix(X_train[:, [i]], label=y_train, feature_names=[BASELINE_FEATURE])
    dva = xgb.DMatrix(X_val[:, [i]], feature_names=[BASELINE_FEATURE])
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 4, "eta": 0.1, "subsample": 0.9,
        "colsample_bytree": 0.9, "scale_pos_weight": spw, "seed": seed,
    }
    return xgb.train(params, dtr, num_boost_round=200).predict(dva)


def bootstrap_lift_ci(y_true: np.ndarray,
                      proba: np.ndarray,
                      baseline_proba: np.ndarray,
                      n: int = BOOTSTRAP_N,
                      seed: int = BOOTSTRAP_SEED,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for (model AUC - baseline AUC), by resampling the test set.

    PAIRED: each replicate resamples row indices once and scores BOTH models on
    those same rows, so the interval is on the difference and the correlation
    between the two models is preserved. Resampling them independently breaks
    the pairing and inflates the interval, which would make a real improvement
    look inconclusive.

    Replicates whose resample happens to be single-class are skipped, because
    AUC is undefined there.

    The seed is fixed so the gate is reproducible: the same split must always
    produce the same verdict.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    idx = np.arange(len(y_true))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        ys = y_true[s]
        if len(np.unique(ys)) < 2:
            continue
        diffs.append(roc_auc_score(ys, proba[s]) - roc_auc_score(ys, baseline_proba[s]))
    if len(diffs) < n // 2:
        raise ValueError(
            "Too few usable bootstrap replicates to form a confidence interval. "
            "Your test set is too small or too imbalanced to support this gate."
        )
    d = np.sort(diffs)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def bootstrap_auc_ci(y_true: np.ndarray, proba: np.ndarray,
                     n: int = SLICE_BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED):
    """95% percentile CI for a single AUC, or None if it cannot be formed."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        vals.append(roc_auc_score(y_true[s], proba[s]))
    if len(vals) < n // 2:
        return None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def top_fraction_mask(proba: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Boolean mask of exactly the top ceil(frac * n) rows by score.

    Selects by RANK, not by `proba >= percentile`. The percentile version lets
    every row tied with the cutoff in, so a model with coarse or repeated scores
    silently contacts more than 10% of customers and its precision/recall are
    measured on a bigger list than the one the business pays for (observed:
    a depth-3 model selected ~411 of 3,000 for a "top 10%"). A stable sort
    breaks ties by row order, which is deterministic because rows are ordered
    by customer_id.
    """
    k = int(np.ceil(frac * len(proba)))
    order = np.argsort(-proba, kind="stable")
    mask = np.zeros(len(proba), dtype=bool)
    mask[order[:k]] = True
    return mask


def _confusion(y_true: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_model(y_val: np.ndarray,
                   y_pred_proba: np.ndarray,
                   baseline_proba: np.ndarray) -> tuple[dict, list[str]]:
    """
    Score the model against the required gates.

    Returns (metrics, failures). It does NOT raise on a failed gate: a run that
    misses a gate is still a legitimate experiment and must still be logged to
    MLflow. Only registration is blocked by a failure (see main()).
    """
    auc_roc = roc_auc_score(y_val, y_pred_proba)

    # Precision and recall @ top 10%
    top10_mask = top_fraction_mask(y_pred_proba, 0.10)
    top10_cutoff = float(y_pred_proba[top10_mask].min())
    precision_top10 = y_val[top10_mask].mean() if top10_mask.sum() > 0 else 0.0
    recall_top10 = y_val[top10_mask].sum() / y_val.sum() if y_val.sum() > 0 else 0.0

    # Baseline: a model trained on days_since_last_purchase ALONE. See
    # train_recency_baseline for why a constant predictor would be meaningless.
    baseline_auc = roc_auc_score(y_val, baseline_proba)
    auc_vs_baseline = auc_roc - baseline_auc

    lift_ci_low, lift_ci_high = bootstrap_lift_ci(
        y_val, y_pred_proba, baseline_proba)

    metrics = {
        "auc_roc": round(float(auc_roc), 4),
        "precision_top10": round(float(precision_top10), 4),
        "recall_top10": round(float(recall_top10), 4),
        "auc_vs_baseline": round(float(auc_vs_baseline), 4),
        "baseline_auc": round(float(baseline_auc), 4),
        "lift_ci_low": round(float(lift_ci_low), 4),
        "lift_ci_high": round(float(lift_ci_high), 4),
        "positive_rate_val": round(float(y_val.mean()), 4),
        "n_val_samples": int(len(y_val)),
        "n_top10": int(top10_mask.sum()),
        "top10_cutoff": round(top10_cutoff, 4),
        # Two operating points: the business one (contact the top 10%) and the
        # conventional 0.5 probability cutoff. With scale_pos_weight the
        # probabilities are inflated, so 0.5 over-flags -- which is why top-10%
        # is the point the retention team will actually use.
        "confusion_top10": _confusion(y_val, top10_mask.astype(int)),
        "confusion_p050": _confusion(y_val, (y_pred_proba > 0.5).astype(int)),
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print("\n── Evaluation Results ──────────────────────────────")
    failures = []
    for metric, threshold in EVAL_THRESHOLDS.items():
        value = metrics[metric]
        status = "✓ PASS" if value >= threshold else "✗ FAIL"
        print(f"  {metric:25s}: {value:.4f}  (threshold: {threshold:.4f})  {status}")
        if value < threshold:
            failures.append(f"{metric}={value:.4f} < {threshold}")
    print(f"  {'auc_roc (not gated)':25s}: {metrics['auc_roc']:.4f}")
    print(f"  {'baseline_auc':25s}: {metrics['baseline_auc']:.4f}")

    # The baseline gate: the interval must exclude zero.
    ci_status = "✓ PASS" if lift_ci_low > 0 else "✗ FAIL"
    print(f"  {'auc_lift 95% CI':25s}: [{lift_ci_low:.4f}, {lift_ci_high:.4f}]"
          f"  (must exclude 0)  {ci_status}")
    if lift_ci_low <= 0:
        failures.append(
            f"auc_lift 95% CI [{lift_ci_low:.4f}, {lift_ci_high:.4f}] includes "
            f"zero - no evidence the model beats the recency-only baseline")

    print("  All gates passed ✓" if not failures
          else f"  GATES FAILED: {'; '.join(failures)}")
    return metrics, failures


# ── Slice Evaluation ───────────────────────────────────────────────────────────

def evaluate_slices(y_val: np.ndarray,
                    y_pred_proba: np.ndarray,
                    slice_column: pd.Series,
                    top10_mask: np.ndarray,
                    aggregate_recall: float) -> dict:
    """
    Evaluate model performance by slice (loyalty tier).

    Two recalls per tier, because "recall@10% for a slice" has two readings and
    they answer different questions:

      recall_top10_global  recall at the SAME global top-10% score cutoff the
                           business would use. The retention team contacts one
                           ranked list, so this is what each tier actually gets.
                           It mixes ranking quality with base rate: a tier that
                           churns less will have fewer customers above a global
                           cutoff no matter how well it is ranked.
      recall_top10_within  recall if you contacted the top 10% OF THAT TIER.
                           Isolates ranking quality inside the tier.

    Each is flagged when it falls more than 10pp below the aggregate recall.

    Every row carries n and n_pos. Slices with fewer than MIN_SLICE_POSITIVES
    positives are marked too_small: their AUC and recall are reported but must
    not be read as findings.
    """
    slice_values = np.asarray(slice_column)
    out = {}
    for tier in sorted(pd.unique(slice_values)):
        mask = slice_values == tier
        y_s, p_s = y_val[mask], y_pred_proba[mask]
        n, n_pos = int(mask.sum()), int(y_s.sum())
        has_both = 0 < n_pos < n
        auc_s = float(roc_auc_score(y_s, p_s)) if has_both else None
        ci = bootstrap_auc_ci(y_s, p_s) if has_both else None
        recall_s = float(y_s[top10_mask[mask]].sum() / n_pos) if n_pos else None
        recall_w = float(y_s[top_fraction_mask(p_s, 0.10)].sum() / n_pos) if n_pos else None
        out[str(tier)] = {
            "n": n,
            "n_pos": n_pos,
            "churn_rate": round(n_pos / n, 4) if n else None,
            "auc": round(auc_s, 4) if auc_s is not None else None,
            "auc_ci_low": round(ci[0], 4) if ci else None,
            "auc_ci_high": round(ci[1], 4) if ci else None,
            "recall_top10_global": round(recall_s, 4) if recall_s is not None else None,
            "recall_top10_within": round(recall_w, 4) if recall_w is not None else None,
            "too_small": n_pos < MIN_SLICE_POSITIVES,
            "flagged_global": bool(recall_s is not None
                                   and recall_s < aggregate_recall - 0.10),
            "flagged_within": bool(recall_w is not None
                                   and recall_w < aggregate_recall - 0.10),
        }

    # Weakest tier = lowest AUC among tiers big enough to support a claim. AUC is
    # the base-rate-free measure of ranking quality; picking by global recall
    # would just name whichever tier churns least. Reported separately so a tiny
    # tier can never be named "weakest" by accident.
    eligible = {t: r for t, r in out.items()
                if not r["too_small"] and r["auc"] is not None}
    weakest = min(eligible, key=lambda t: eligible[t]["auc"]) if eligible else None

    print("\n── Slice evaluation by loyalty tier ────────────────")
    print(f"  aggregate recall@10% = {aggregate_recall:.4f} "
          f"(flag if below {aggregate_recall - 0.10:.4f})")
    print(f"  {'tier':10s} {'n':>5s} {'n_pos':>6s} {'AUC':>7s} {'AUC 95% CI':>17s} "
          f"{'rec@glob':>9s} {'rec@within':>10s}  notes")
    for tier, r in out.items():
        ci = (f"[{r['auc_ci_low']:.3f}, {r['auc_ci_high']:.3f}]"
              if r["auc_ci_low"] is not None else "n/a")
        notes = []
        if r["too_small"]:
            notes.append(f"TOO SMALL (<{MIN_SLICE_POSITIVES} positives)")
        if r["flagged_global"]:
            notes.append("FLAG(global)")
        if r["flagged_within"]:
            notes.append("FLAG(within)")
        if tier == weakest:
            notes.append("weakest AUC")
        auc_txt = f"{r['auc']:.4f}" if r["auc"] is not None else "n/a"
        g_txt = f"{r['recall_top10_global']:.4f}" if r["recall_top10_global"] is not None else "n/a"
        w_txt = f"{r['recall_top10_within']:.4f}" if r["recall_top10_within"] is not None else "n/a"
        print(f"  {tier:10s} {r['n']:5d} {r['n_pos']:6d} {auc_txt:>7s} {ci:>17s} "
              f"{g_txt:>9s} {w_txt:>10s}  {'; '.join(notes)}")

    return {"tiers": out, "weakest_tier": weakest,
            "aggregate_recall_top10": round(float(aggregate_recall), 4),
            "min_slice_positives": MIN_SLICE_POSITIVES}


def plot_feature_importance(model: xgb.Booster, path: str,
                            feature_columns: list[str]) -> dict:
    """Save a gain-based feature importance bar chart; return the gains.

    get_score omits any feature that was never used in a split. Those are exactly the
    "decorative" features the design doc asks about, so list them explicitly at 0.
    """
    gain = {c: 0.0 for c in feature_columns}
    gain.update(model.get_score(importance_type="gain"))
    items = sorted(gain.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#2a6f97")
    ax.set_xlabel("Average gain per split")
    ax.set_title("Churn model: feature importance (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {k: round(float(v), 4) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}


def plot_confusion_matrix(confusion: dict, cutoff: float, n_flagged: int, path: str) -> None:
    """Save the confusion matrix at the chosen operating point (top 10% by score)."""
    cm = np.array([[confusion["tn"], confusion["fp"]],
                   [confusion["fn"], confusion["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["Stayed", "Churned"])
    ax.set_xlabel("Model decision")
    ax.set_ylabel("Actual outcome")
    ax.set_title(f"Confusion matrix at top-10% cutoff\n"
                 f"(score >= {cutoff:.3f}, {n_flagged:,} customers flagged)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_slice_csv(slices: dict, path: str) -> None:
    """One row per loyalty tier: n, positives, AUC (with CI), recall, too-small flag."""
    fields = ["tier", "n", "positives", "churn_rate", "auc", "auc_ci_low", "auc_ci_high",
              "recall_top10_global", "recall_top10_within", "too_small"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tier, r in slices["tiers"].items():
            w.writerow({"tier": tier, "positives": r["n_pos"],
                        **{k: r[k] for k in fields if k not in ("tier", "positives")}})


# ── Experiment Tracking (MLflow App) ───────────────────────────────────────────

def track_run(args, metrics: dict, slices: dict, feature_columns: list[str],
              split_sizes: dict, artifact_paths: list[str]) -> None:
    """
    Log this training run to your SageMaker MLflow App.

    Worth 5 points in Task 1: >=3 runs, each with logged params AND metrics,
    retrievable via mlflow.search_runs.

    ###########################################################################
    #  THERE ARE TWO MLflow PRODUCTS ON SAGEMAKER. USE THE APP.               #
    #                                                                         #
    #    CreateMlflowApp            serverless   NO ADDITIONAL CHARGE   <-- ok #
    #    CreateMlflowTrackingServer $0.60/hour   until you delete it    <-- NO #
    #                                                                         #
    #  $0.60/hr breaches the entire $10 course budget in 16.7 hours and costs #
    #  about $43 over a weekend. If anything asks you to choose a size         #
    #  (Small/Medium), you are on the wrong product.                          #
    ###########################################################################

    Needs BOTH `mlflow` and `sagemaker-mlflow` installed; the second is the
    SigV4 auth plugin for arn:aws:sagemaker:... tracking URIs.
    """
    if not args.mlflow_arn:
        print("\n(no --mlflow-arn given; skipping experiment tracking)")
        return

    import mlflow

    mlflow.set_tracking_uri(args.mlflow_arn)   # the App ARN, verbatim
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        args.mlflow_run_id = run.info.run_id
        # XGBoost version as a param: the SageMaker container is 1.7 and the
        # reference numbers are 3.2.0, so runs on different versions are not
        # directly comparable and the log must say which is which.
        mlflow.log_params({
            "max_depth": args.max_depth,
            "eta": args.eta,
            "num_round": args.num_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": round(args.resolved_scale_pos_weight, 4),
            "include_risk_score": args.include_risk_score,
            "n_features": len(feature_columns),
            "seed": args.seed,
            "athena_query_id": args.athena_query_id,
            "data_sha256": args.data_sha256,
            "n_customers": args.n_customers,
            "best_iteration": args.best_iteration,
            "xgboost_version": xgb.__version__,
            **split_sizes,
        })
        flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        for tier, r in slices["tiers"].items():
            if r["auc"] is not None:
                flat[f"slice_auc_{tier}"] = r["auc"]
            if r["recall_top10_global"] is not None:
                flat[f"slice_recall_global_{tier}"] = r["recall_top10_global"]
            if r["recall_top10_within"] is not None:
                flat[f"slice_recall_within_{tier}"] = r["recall_top10_within"]
        flat["gates_passed"] = float(args.gates_passed)
        mlflow.log_metrics(flat)
        mlflow.set_tag("feature_set", "with_risk_score" if args.include_risk_score
                       else "without_risk_score")
        # Only the files this run wrote: --output-data-dir may be a shared evidence folder.
        for path in artifact_paths:
            mlflow.log_artifact(path)

    print(f"\n✓ Logged run '{args.run_name}' to MLflow experiment "
          f"'{args.mlflow_experiment}'")


# ── Model Saving & Registry ────────────────────────────────────────────────────

# AWS-owned ECR account for the built-in XGBoost container in us-east-1. The
# registry needs an inference image to describe the package; nothing is
# deployed in Lab 3, so this is metadata until Lab 5.
XGBOOST_IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"


def save_model(model: xgb.Booster, metrics: dict, feature_columns: list[str], args) -> str:
    """Write the model and its metadata locally. Returns the model path."""
    os.makedirs(args.model_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, "model.json")
    model.save_model(model_path)
    print(f"\n✓ Model saved to {model_path}")

    metadata = {
        "feature_columns": feature_columns,
        "label_column": LABEL_COLUMN,
        "hyperparameters": {
            "max_depth": args.max_depth,
            "eta": args.eta,
            "num_round": args.num_round,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": args.resolved_scale_pos_weight,
            "best_iteration": args.best_iteration,
        },
        "xgboost_version": xgb.__version__,
        "evaluation_metrics": metrics,
        "training_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return model_path


def register_model(model_path: str, metrics: dict, args) -> str:
    """Register in the SageMaker Model Registry as PendingManualApproval.

    The group name must stay "northstar-churn-models": Lab 5 deploys from that
    group by name. Status is hard-coded, never a parameter, so no flag on this
    script can approve a model.
    """
    sm = boto3.client("sagemaker")
    s3 = boto3.client("s3")

    try:
        sm.create_model_package_group(
            ModelPackageGroupName=args.model_package_group,
            ModelPackageGroupDescription="NorthStar 90-day churn models (XGBoost)",
        )
        print(f"  created model package group {args.model_package_group}")
    except sm.exceptions.ClientError as e:
        if "already exist" not in str(e):
            raise

    # SageMaker expects model.tar.gz containing the artifact under a known name.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tar_path = os.path.join(args.model_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="xgboost-model")
    key = f"artifacts/models/churn/{stamp}/model.tar.gz"
    s3.upload_file(tar_path, args.artifacts_bucket, key)

    keep = ["auc_roc", "precision_top10", "recall_top10", "baseline_auc",
            "auc_vs_baseline", "lift_ci_low", "lift_ci_high", "n_val_samples"]
    resp = sm.create_model_package(
        ModelPackageGroupName=args.model_package_group,
        ModelPackageDescription=(
            f"Churn model {stamp}: AUC {metrics['auc_roc']}, "
            f"P@10% {metrics['precision_top10']}, R@10% {metrics['recall_top10']}, "
            f"lift over recency baseline {metrics['auc_vs_baseline']} "
            f"[{metrics['lift_ci_low']}, {metrics['lift_ci_high']}] "
            f"(XGBoost {xgb.__version__}, risk score "
            f"{'included' if args.include_risk_score else 'excluded'})"),
        InferenceSpecification={
            "Containers": [{
                "Image": XGBOOST_IMAGE,
                "ModelDataUrl": f"s3://{args.artifacts_bucket}/{key}",
            }],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties={
            **{k: str(metrics[k]) for k in keep},
            "xgboost_version": xgb.__version__,
            "include_risk_score": str(args.include_risk_score),
            "mlflow_run_name": str(args.run_name),
            "mlflow_run_id": str(getattr(args, "mlflow_run_id", "none")),
        },
    )
    arn = resp["ModelPackageArn"]
    print(f"✓ Registered {arn} as PendingManualApproval")
    return arn


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    feature_columns = resolve_feature_columns(args.include_risk_score)
    if args.run_name is None:
        args.run_name = (f"xgb-d{args.max_depth}-eta{args.eta}"
                         f"{'-risk' if args.include_risk_score else ''}")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    print("=" * 60)
    print("NorthStar Churn Prediction — Training Script")
    print(f"run: {args.run_name}   xgboost {xgb.__version__}   "
          f"risk score {'IN' if args.include_risk_score else 'out'}")
    print("=" * 60)

    # 1. Load features
    if args.local_data_path:
        df = load_features_local(args.local_data_path, feature_columns)
    else:
        df = load_features_from_feature_store(
            args.feature_group_name,
            args.training_start_date,
            args.training_end_date,
            args.artifacts_bucket,
            feature_columns,
            timeout_s=args.athena_timeout,
        )
    # Which data this run saw, logged to MLflow and the evidence JSON so a later
    # re-ingest cannot be mistaken for the same dataset.
    args.athena_query_id = df.attrs.get("athena_query_id", "unknown")
    args.data_sha256 = df.attrs.get("data_sha256", "unknown")
    args.n_customers = len(df)
    if args.expected_customers is not None and len(df) != args.expected_customers:
        raise RuntimeError(f"Loaded {len(df):,} customers but --expected-customers is "
                           f"{args.expected_customers:,}. The Feature Store contents or the "
                           f"ingest window changed; check before training.")

    # 2. Preprocess
    X, y, df = preprocess_features(df, feature_columns)

    # 3. Split. test_size=0.30 matches the reference run (6,999 train / 3,000
    # test on the 10k dataset); the bootstrap lift CI is calibrated on a test
    # set that size. Splitting row INDICES keeps each customer's loyalty tier
    # aligned with its prediction. Early stopping then uses a validation set
    # carved from the training rows, so the test set is touched exactly once.
    idx = np.arange(len(y))
    fit_idx, test_idx = train_test_split(
        idx, test_size=0.30, random_state=args.seed, stratify=y)
    train_idx, stop_idx = train_test_split(
        fit_idx, test_size=args.inner_val_size, random_state=args.seed, stratify=y[fit_idx])
    X_train, y_train = X[train_idx], y[train_idx]
    X_stop, y_stop = X[stop_idx], y[stop_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    tier_test = df[SLICE_COLUMN].values[test_idx]
    split_sizes = {"n_train": len(train_idx), "n_stop": len(stop_idx),
                   "n_test": len(test_idx)}
    print(f"\nTrain: {len(train_idx):,} | Early-stop: {len(stop_idx):,} | "
          f"Test: {len(test_idx):,}")

    # 4. Train
    model = train_model(X_train, y_train, X_stop, y_stop, args, feature_columns)
    args.best_iteration = int(model.best_iteration)

    # 5. Evaluate on the untouched test set
    proba = predict_proba(model, X_test, feature_columns)
    baseline_proba = train_recency_baseline(
        X_train, y_train, X_test, feature_columns, args.seed)
    metrics, failures = evaluate_model(y_test, proba, baseline_proba)
    # AUC on the inner early-stopping set. This, not the test set, is the number
    # to compare configurations by: choosing a config on test metrics would tune
    # it to the 3,000 rows the final report is graded on.
    metrics["val_auc_inner"] = round(float(model.best_score), 4)
    args.gates_passed = not failures

    slices = evaluate_slices(y_test, proba, tier_test,
                             top_fraction_mask(proba, 0.10), metrics["recall_top10"])
    out = args.output_data_dir
    importance_png = os.path.join(out, "feature-importance.png")
    confusion_png = os.path.join(out, "confusion-matrix.png")
    slices_csv = os.path.join(out, "slice-evaluation.csv")
    metrics_json = os.path.join(out, "evaluation_metrics.json")
    importance = plot_feature_importance(model, importance_png, feature_columns)
    plot_confusion_matrix(metrics["confusion_top10"], metrics["top10_cutoff"],
                          metrics["n_top10"], confusion_png)
    write_slice_csv(slices, slices_csv)

    # 6. Save locally and write the evidence files. Everything the design doc
    #    cites is written here, because teardown deletes the S3 copies.
    model_path = save_model(model, metrics, feature_columns, args)
    # The metrics sit at the TOP LEVEL of this file: scripts/collect-lab3-evidence.sh reads
    # auc_roc, precision_top10, recall_top10, baseline_auc, lift_ci_* and n_val_samples
    # from here and cross-checks them against MLflow and the registry.
    report = {
        **metrics,
        "run_name": args.run_name,
        "xgboost_version": xgb.__version__,
        "include_risk_score": args.include_risk_score,
        "data": {"athena_query_id": args.athena_query_id, "data_sha256": args.data_sha256,
                 "n_customers": args.n_customers},
        "feature_columns": feature_columns,
        "hyperparameters": {
            "max_depth": args.max_depth, "eta": args.eta, "num_round": args.num_round,
            "min_child_weight": args.min_child_weight, "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": args.resolved_scale_pos_weight,
            "best_iteration": args.best_iteration, "seed": args.seed},
        "split_sizes": split_sizes,
        "gates_passed": not failures,
        "gate_failures": failures,
        "slices": slices,
        "feature_importance_gain": importance,
    }
    with open(metrics_json, "w") as f:
        json.dump(report, f, indent=2)

    # 7. Track this run in your MLflow App (Task 1, 5 pts). Logged even when a
    #    gate fails: a run that misses is still an experiment worth comparing.
    track_run(args, metrics, slices, feature_columns, split_sizes,
              artifact_paths=[metrics_json, importance_png, confusion_png, slices_csv])

    # 8. Register only on request, and only if every gate passed.
    if args.register:
        if failures:
            print("\n✗ Not registering: " + "; ".join(failures))
            raise SystemExit(1)
        register_model(model_path, metrics, args)

    print("\n" + "=" * 60)
    print(f"Training complete. AUC-ROC: {metrics['auc_roc']:.4f}  "
          f"gates {'PASSED' if not failures else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
