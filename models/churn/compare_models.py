"""Compare the XGBoost churn model against logistic regression and a small neural net.

Same data, same split, same metrics as train.py, so the numbers in the design doc are
measured rather than assumed. Writes docs/lab_3/results/model-comparison.json.

    python models/churn/compare_models.py --artifacts-bucket "$(terraform output -raw s3_bucket_name)"
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train as T


def topk(y, proba):
    mask = T.top_fraction_mask(proba, 0.10)
    tp = int(y[mask].sum())
    return {"precision_top10": round(tp / mask.sum(), 4), "recall_top10": round(tp / y.sum(), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-bucket", required=True)
    ap.add_argument("--feature-group-name", default="northstar-dev-customer-features")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(Path(__file__).resolve().parents[2]
                                            / "docs" / "lab_3" / "results" / "model-comparison.json"))
    a = ap.parse_args()

    cols = T.resolve_feature_columns(False)
    df = T.load_features_from_feature_store(a.feature_group_name, "2026-01-01", "2027-01-01",
                                            a.artifacts_bucket, cols)
    X, y, df = T.preprocess_features(df, cols)

    # identical split to train.py
    idx = np.arange(len(y))
    fit_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=a.seed, stratify=y)
    train_idx, stop_idx = train_test_split(fit_idx, test_size=0.15, random_state=a.seed, stratify=y[fit_idx])
    Xtr, ytr, Xte, yte = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    out = {"n_train": len(train_idx), "n_test": len(test_idx), "models": {}}

    # XGBoost: the config train.py selected (d4 / eta 0.1)
    xargs = SimpleNamespace(max_depth=4, eta=0.1, num_round=200, min_child_weight=5, subsample=0.8,
                            colsample_bytree=0.8, scale_pos_weight=None, seed=a.seed)
    booster = T.train_model(Xtr, ytr, X[stop_idx], y[stop_idx], xargs, cols)
    p_xgb = T.predict_proba(booster, Xte, cols)

    # Logistic regression: standardized, class-balanced
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=2000, random_state=a.seed))
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    coefs = dict(zip(cols, lr[-1].coef_[0].round(3).tolist()))

    # Small MLP (sklearn; 2 hidden layers, early stopping on an internal 15% split)
    mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True,
                                      validation_fraction=0.15, max_iter=500, random_state=a.seed))
    mlp.fit(Xtr, ytr)
    p_mlp = mlp.predict_proba(Xte)[:, 1]

    for name, p in [("xgboost", p_xgb), ("logistic_regression", p_lr), ("mlp_32_16", p_mlp)]:
        out["models"][name] = {"auc": round(float(roc_auc_score(yte, p)), 4), **topk(yte, p)}

    for name, p in [("logistic_regression", p_lr), ("mlp_32_16", p_mlp)]:
        lo, hi = T.bootstrap_lift_ci(yte, p_xgb, p)
        out["models"][name]["xgboost_auc_minus_this_ci95"] = [round(lo, 4), round(hi, 4)]
    out["logistic_regression_standardized_coefficients"] = coefs

    Path(a.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
