"""Your Lab 3 churn model, packaged for the Lab 4 pipeline.

YOU WRITE THIS FILE. The pipeline's training job runs train_sagemaker.py
(beside this file; do not edit it), which imports this module and calls
train(). Everything the job needs from you goes through the three names below.

Contract (train_sagemaker.py checks it in the first seconds of the job):

    FEATURE_COLUMNS   list[str]   model input columns, in order
    LABEL             str         "churn_label"
    train(df, seed=42) -> (xgboost.Booster, dict)

`df` is the full labeled feature table: every column of your Lab 2 output,
including `loyalty_tier` for slice evaluation. It is already loaded. Do not
query Athena or the Feature Store from here; the training container stages
the data for you and has no business making service calls for its input.

The metrics dict must include the four numbers the CI gate in
tests/test_model.py reads. Your Lab 3 evaluate_model() already returns them:

    precision_top10, recall_top10      gated at 0.50 / 0.25
    lift_ci_low, lift_ci_high          bootstrap 95% CI on AUC lift over the
                                       recency-only baseline; lift_ci_low > 0

Include anything else you reported in Lab 3 (auc_roc, baseline_auc,
positive_rate_val, slice results). It is all written to evaluation_metrics.json
and attached to the registered model.

What runs in the container: the SageMaker XGBoost 1.7-1 image ships pandas,
numpy, scikit-learn, xgboost and boto3. It does NOT ship mlflow or sagemaker,
so do not import them at the top of this file. MLflow tracking stays in your
Lab 3 script.

Only this directory (models/churn/) is uploaded to the training job, so the
code must live here. `import churn_training_skeleton` will fail inside the
container even though it works on your laptop. Copy the functions you need.

Test it locally before pushing:

    python models/churn/train_sagemaker.py \
        --train-channel <dir containing your features CSV or Parquet> \
        --model-dir /tmp/model --output-data-dir /tmp/out
"""

# Keeps Lab 3 annotations like `-> tuple[np.ndarray, np.ndarray]` from being
# evaluated, whatever Python the training container runs.
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Must match the columns your Lab 3 model was trained on, in the same order.
FEATURE_COLUMNS = [
    "days_since_last_purchase",
    "customer_tenure_days",
    "purchase_frequency_30d",
    "purchase_frequency_90d",
    "purchase_frequency_180d",
    "avg_order_value",
    "total_spend_90d",
    "total_lifetime_value",
    "avg_basket_size_6m",
    "category_diversity_score",
    "online_to_store_ratio",
]

LABEL = "churn_label"

# The recency-only baseline model uses just this one feature.
BASELINE_FEATURE = "days_since_last_purchase"
SLICE_COLUMN = "loyalty_tier"
LOYALTY_TIERS = ("Bronze", "Silver", "Gold", "Platinum")

# Lab 3 CLI defaults, frozen. The pipeline runs this with no flags, so these are
# the hyperparameters of every CI model.
HYPERPARAMS = {
    "max_depth": 6, "eta": 0.1, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "num_round": 200, "early_stopping_rounds": 20,
}
# Domain constraint, not a tuned hyperparameter: a customer who has been silent
# longer must never score as LESS likely to churn, all else equal. Without it the
# unconstrained model dipped 0.063 between 60 and 90 days on a fixed profile and
# failed test_model_monotone_on_recency in CI (2026-10-30).
MONOTONE_INCREASING = ("days_since_last_purchase",)
INNER_VAL_SIZE = 0.15   # share of training rows held out to pick the early-stopping round

# Gates. There is deliberately no absolute AUC gate (see tests/test_model.py).
EVAL_THRESHOLDS = {"precision_top10": 0.50, "recall_top10": 0.25}
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
SLICE_BOOTSTRAP_N = 500
MIN_SLICE_POSITIVES = 30   # a tier with fewer positives cannot support a claim


# ── Model training ─────────────────────────────────────────────────────────────

def preprocess_features(df: pd.DataFrame):
    """Validate the labeled table and return (X, y, df).

    Rejects labels that are not 0/1 (astype(int) would silently truncate 0.7 to
    0) and tiers outside the data contract (they would create a bogus slice).
    NaN features pass through: XGBoost learns a default branch for missing
    values, whereas fillna(-1) would turn "missing" into a real value.
    """
    df = df.dropna(subset=[LABEL]).reset_index(drop=True)
    bad_labels = sorted(set(df[LABEL].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"{LABEL} must be 0 or 1; found {bad_labels}")
    bad_tiers = df[SLICE_COLUMN][~df[SLICE_COLUMN].isin(LOYALTY_TIERS)]
    if len(bad_tiers):
        raise ValueError(f"{len(bad_tiers):,} rows have a {SLICE_COLUMN} outside "
                         f"{list(LOYALTY_TIERS)}: {sorted(bad_tiers.astype(str).unique())}")
    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df[LABEL].values.astype(int)
    return X, y, df


def train_model(X_train, y_train, X_stop, y_stop, seed: int) -> xgb.Booster:
    """Train the XGBoost churn model with early stopping on (X_stop, y_stop).

    (X_stop, y_stop) is carved out of the TRAINING rows, so the 30% test set is
    never seen here and the test metrics are not tuned to it.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dstop = xgb.DMatrix(X_stop, label=y_stop, feature_names=FEATURE_COLUMNS)
    # negatives / positives equalises the total weight of the two classes.
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "objective": "binary:logistic",
        # ORDER MATTERS: XGBoost early-stops on the LAST metric, so "auc" goes last.
        "eval_metric": ["logloss", "auc"],
        "max_depth": HYPERPARAMS["max_depth"], "eta": HYPERPARAMS["eta"],
        "min_child_weight": HYPERPARAMS["min_child_weight"],
        "subsample": HYPERPARAMS["subsample"],
        "colsample_bytree": HYPERPARAMS["colsample_bytree"],
        "scale_pos_weight": spw, "seed": seed, "verbosity": 1,
        "monotone_constraints": "(" + ",".join(
            "1" if c in MONOTONE_INCREASING else "0" for c in FEATURE_COLUMNS) + ")",
    }
    model = xgb.train(params, dtrain, num_boost_round=HYPERPARAMS["num_round"],
                      evals=[(dtrain, "train"), (dstop, "stop")],
                      early_stopping_rounds=HYPERPARAMS["early_stopping_rounds"],
                      verbose_eval=25)
    # Early stopping leaves extra trees after the best round; a plain predict()
    # would use them. Keep only the best trees, and restore the attributes that
    # slicing drops.
    best_iteration, best_score = model.best_iteration, model.best_score
    model = model[: best_iteration + 1]
    model.set_attr(best_iteration=str(best_iteration), best_score=str(best_score))
    return model


def predict_proba(model: xgb.Booster, X) -> np.ndarray:
    """Score with the best early-stopping round, stated explicitly."""
    best = getattr(model, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    return model.predict(xgb.DMatrix(X, feature_names=FEATURE_COLUMNS), iteration_range=rng)


def train_recency_baseline(X_train, y_train, X_val, seed: int) -> np.ndarray:
    """Train a model on days_since_last_purchase ALONE and score the test rows.

    Same training rows and same test rows as the full model, which is what makes
    the comparison fair. Deliberately not a constant predictor: that scores an
    AUC of exactly 0.5, so beating it would only prove "better than a coin flip".
    The real question is whether the features beat the free rule "contact whoever
    has not purchased lately". Parameters are fixed so a run cannot move the
    goalposts it is scored against.
    """
    i = FEATURE_COLUMNS.index(BASELINE_FEATURE)
    dtr = xgb.DMatrix(X_train[:, [i]], label=y_train, feature_names=[BASELINE_FEATURE])
    dva = xgb.DMatrix(X_val[:, [i]], feature_names=[BASELINE_FEATURE])
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {"objective": "binary:logistic", "eval_metric": "auc",
              "max_depth": 4, "eta": 0.1, "subsample": 0.9,
              "colsample_bytree": 0.9, "scale_pos_weight": spw, "seed": seed}
    return xgb.train(params, dtr, num_boost_round=200).predict(dva)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def top_fraction_mask(proba: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Boolean mask of exactly the top ceil(frac * n) rows by score.

    Selects by RANK, not `proba >= percentile`: the percentile form lets every
    row tied with the cutoff in, so coarse scores silently contact more than
    10% of customers. A stable sort breaks ties by row order.
    """
    k = int(np.ceil(frac * len(proba)))
    order = np.argsort(-proba, kind="stable")
    mask = np.zeros(len(proba), dtype=bool)
    mask[order[:k]] = True
    return mask


def bootstrap_lift_ci(y_true, proba, baseline_proba, n: int = BOOTSTRAP_N,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05):
    """Percentile CI for (model AUC - baseline AUC), by resampling the test set.

    PAIRED: each replicate resamples row indices once and scores BOTH models on
    those rows, so the interval is on the difference and keeps the correlation
    between the models. Single-class resamples are skipped (AUC is undefined).
    The seed is fixed so the same split always gives the same verdict.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    idx = np.arange(len(y_true))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[s], proba[s]) - roc_auc_score(y_true[s], baseline_proba[s]))
    if len(diffs) < n // 2:
        raise ValueError("Too few usable bootstrap replicates to form a confidence "
                         "interval: the test set is too small or too imbalanced.")
    d = np.sort(diffs)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def bootstrap_auc_ci(y_true, proba, n: int = SLICE_BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED):
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


def _confusion(y_true, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_model(y_val, y_pred_proba, baseline_proba):
    """Score the model against the gates. Returns (metrics, failures).

    Does not raise: train() decides what a failure means. Both metric
    vocabularies are emitted (baseline_auc and baseline_auc_roc, etc.) because
    the handout and tests/test_model.py use different names for the same numbers.
    """
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    top10 = top_fraction_mask(y_pred_proba, 0.10)
    precision_top10 = y_val[top10].mean() if top10.sum() > 0 else 0.0
    recall_top10 = y_val[top10].sum() / y_val.sum() if y_val.sum() > 0 else 0.0
    baseline_auc = roc_auc_score(y_val, baseline_proba)
    lo, hi = bootstrap_lift_ci(y_val, y_pred_proba, baseline_proba)

    metrics = {
        "auc_roc": round(float(auc_roc), 4),
        "precision_top10": round(float(precision_top10), 4),
        "recall_top10": round(float(recall_top10), 4),
        "baseline_auc": round(float(baseline_auc), 4),
        "baseline_auc_roc": round(float(baseline_auc), 4),
        "auc_vs_baseline": round(float(auc_roc - baseline_auc), 4),
        "lift_ci_low": round(lo, 4), "lift_ci_high": round(hi, 4),
        "auc_lift_ci_low": round(lo, 4), "auc_lift_ci_high": round(hi, 4),
        "positive_rate_val": round(float(y_val.mean()), 4),
        "n_val_samples": int(len(y_val)),
        "n_top10": int(top10.sum()),
        "confusion_top10": _confusion(y_val, top10.astype(int)),
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    failures = [f"{m}={metrics[m]:.4f} < {t}" for m, t in EVAL_THRESHOLDS.items()
                if metrics[m] < t]
    if lo <= 0:
        failures.append(f"auc_lift 95% CI [{lo:.4f}, {hi:.4f}] includes zero - no evidence "
                        f"the model beats the recency-only baseline")
    return metrics, failures


def evaluate_slices(y_val, y_pred_proba, slice_column, top10_mask, aggregate_recall) -> dict:
    """Per-loyalty-tier AUC and recall@top10%, for the fairness check.

    recall_top10_global is recall at the SAME global top-10% cutoff the business
    would use (what each tier actually gets); recall_top10_within is recall if
    you contacted the top 10% OF THAT TIER (ranking quality inside the tier).
    Every row carries n and n_pos, and tiers under MIN_SLICE_POSITIVES are marked
    too_small: reported, but not to be read as findings.
    """
    values = np.asarray(slice_column)
    out = {}
    for tier in sorted(pd.unique(values)):
        mask = values == tier
        y_s, p_s = y_val[mask], y_pred_proba[mask]
        n, n_pos = int(mask.sum()), int(y_s.sum())
        has_both = 0 < n_pos < n
        auc_s = float(roc_auc_score(y_s, p_s)) if has_both else None
        ci = bootstrap_auc_ci(y_s, p_s) if has_both else None
        recall_g = float(y_s[top10_mask[mask]].sum() / n_pos) if n_pos else None
        recall_w = float(y_s[top_fraction_mask(p_s, 0.10)].sum() / n_pos) if n_pos else None
        out[str(tier)] = {
            "n": n, "n_pos": n_pos,
            "churn_rate": round(n_pos / n, 4) if n else None,
            "auc": round(auc_s, 4) if auc_s is not None else None,
            "auc_ci_low": round(ci[0], 4) if ci else None,
            "auc_ci_high": round(ci[1], 4) if ci else None,
            "recall_top10_global": round(recall_g, 4) if recall_g is not None else None,
            "recall_top10_within": round(recall_w, 4) if recall_w is not None else None,
            "too_small": n_pos < MIN_SLICE_POSITIVES,
        }
    return {"tiers": out, "aggregate_recall_top10": round(float(aggregate_recall), 4),
            "min_slice_positives": MIN_SLICE_POSITIVES}


# ── Entry point ────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, seed: int = 42):
    """Train the churn model on `df` and return (booster, metrics).

    TODO (Lab 4): implement this from your Lab 3 code. The usual shape:

      1. X, y from FEATURE_COLUMNS and LABEL (handle missing values the way
         your Lab 3 preprocess_features() did)
      2. train_test_split(..., test_size=0.30, random_state=seed, stratify=y)
         -- carry the tier through the same call if you report slices. Pass
         df["loyalty_tier"] (a Series) or .to_numpy(dtype=object); NOT
         .values, which under pandas 3 is an Arrow array that
         train_test_split cannot index
      3. train the XGBoost model (your Lab 3 train_model())
      4. train the recency-only baseline (train_recency_baseline())
      5. metrics = evaluate_model(...)   # keep Lab 3's raise on a failed gate:
                                         # it fails the training step, so a
                                         # model that misses a gate is never
                                         # registered
      6. return model, metrics

    Return an xgboost.Booster (what xgb.train() gives you). If you used
    XGBClassifier, return clf.get_booster().
    """
    X, y, df = preprocess_features(df)

    # Two nested stratified splits, by row index so the tier follows each row:
    # 70% fit / 30% test, then the fit rows give up 15% to pick the stopping round.
    idx = np.arange(len(y))
    fit_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    train_idx, stop_idx = train_test_split(fit_idx, test_size=INNER_VAL_SIZE,
                                           random_state=seed, stratify=y[fit_idx])

    model = train_model(X[train_idx], y[train_idx], X[stop_idx], y[stop_idx], seed)
    proba = predict_proba(model, X[test_idx])
    baseline_proba = train_recency_baseline(X[train_idx], y[train_idx], X[test_idx], seed)

    metrics, failures = evaluate_model(y[test_idx], proba, baseline_proba)
    metrics["slices"] = evaluate_slices(
        y[test_idx], proba, df[SLICE_COLUMN].iloc[test_idx],
        top_fraction_mask(proba, 0.10), metrics["recall_top10"])
    metrics["n_train"] = int(len(train_idx))

    # A model that misses a gate must fail the training step, so it is never registered.
    if failures:
        raise RuntimeError("Model failed gates: " + "; ".join(failures))
    return model, metrics
