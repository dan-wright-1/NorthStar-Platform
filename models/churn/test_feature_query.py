"""Unit tests for build_feature_query, run against in-memory sqlite (no AWS).

    python models/churn/test_feature_query.py        (or: pytest models/churn)

sqlite supports the window function and CTE the Athena query uses, so this checks
the dedupe and soft-delete logic, not Athena syntax.
"""
import sqlite3

import train as T

COLS = ["days_since_last_purchase"]
LO, HI = 100.0, 200.0

# customer_id, days_since_last_purchase, churn_label, loyalty_tier,
# event_time, write_time, is_deleted
ROWS = [
    ("A", 10, 0, "Gold",   150.0, 1.0, 0),   # A re-ingested: the newer row must win
    ("A", 11, 0, "Gold",   160.0, 2.0, 0),
    ("B", 20, 1, "Silver", 150.0, 1.0, 0),   # B soft-deleted: newest row is a marker
    ("B", 20, 1, "Silver", 170.0, 2.0, 1),   # -> B must NOT come back from its older row
    ("C", 30, 0, "Bronze", 150.0, 1.0, 1),   # C only ever a delete marker
    ("D", 40, 1, "Bronze", 50.0,  1.0, 0),   # D outside the ingest window
    ("E", 50, 0, "Gold",   150.0, 1.0, 0),   # E: same event_time, write_time breaks the tie
    ("E", 51, 0, "Gold",   150.0, 5.0, 0),
    ("F", 60, 1, "Silver", 150.0, 1.0, 1),   # F deleted, then re-created later
    ("F", 61, 1, "Silver", 180.0, 9.0, 0),
]


def run_query():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE features (customer_id TEXT, days_since_last_purchase REAL, "
               "churn_label INT, loyalty_tier TEXT, event_time REAL, write_time REAL, is_deleted INT)")
    db.executemany("INSERT INTO features VALUES (?,?,?,?,?,?,?)", ROWS)
    return db.execute(T.build_feature_query("features", COLS, LO, HI)).fetchall()


def test_one_row_per_customer_newest_wins():
    got = {r[0]: r for r in run_query()}
    assert got["A"][1] == 11, "newer re-ingest should win"
    assert got["E"][1] == 51, "write_time should break an event_time tie"
    assert got["F"][1] == 61, "a customer re-created after a delete should be present"


def test_soft_deleted_customers_are_excluded():
    ids = {r[0] for r in run_query()}
    assert "B" not in ids, "newest record is a delete marker; the older live row must not resurrect B"
    assert "C" not in ids


def test_ingest_window_applied():
    assert "D" not in {r[0] for r in run_query()}


def test_ordered_by_customer_id_and_unique():
    ids = [r[0] for r in run_query()]
    assert ids == sorted(ids) and len(ids) == len(set(ids))


def test_fingerprint_is_stable_and_sensitive():
    import pandas as pd
    a = pd.DataFrame({"customer_id": ["A", "B"], "x": [1.0, 2.0]})
    assert T.dataset_fingerprint(a) == T.dataset_fingerprint(a.copy())
    b = a.copy()
    b.loc[1, "x"] = 2.5
    assert T.dataset_fingerprint(a) != T.dataset_fingerprint(b)


def _frame(**overrides):
    import pandas as pd
    base = {c: [1.0, 2.0, 3.0, 4.0] for c in T.FEATURE_COLUMNS}
    base.update({"customer_id": ["A", "B", "C", "D"], "churn_label": [0, 1, 0, 1],
                 "loyalty_tier": ["Bronze", "Silver", "Gold", "Platinum"]})
    base.update(overrides)
    return pd.DataFrame(base)


def test_preprocess_accepts_valid_frame():
    X, y, df = T.preprocess_features(_frame(), T.FEATURE_COLUMNS)
    assert X.shape == (4, len(T.FEATURE_COLUMNS)) and list(y) == [0, 1, 0, 1] and len(df) == 4


def test_preprocess_reports_dropped_unlabeled_rows(capsys=None):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, y, df = T.preprocess_features(_frame(churn_label=[0, 1, None, 1]), T.FEATURE_COLUMNS)
    assert "dropped 1 rows" in buf.getvalue() and len(df) == 3 and len(y) == 3


def test_preprocess_rejects_non_binary_label():
    try:
        T.preprocess_features(_frame(churn_label=[0, 1, 0.7, 1]), T.FEATURE_COLUMNS)
    except ValueError as e:
        assert "0.7" in str(e)
    else:
        raise AssertionError("a label of 0.7 should have been rejected")


def test_preprocess_rejects_unknown_or_null_tier():
    for tiers in (["Bronze", "Silver", "Gold", "Diamond"], ["Bronze", "Silver", "Gold", None]):
        try:
            T.preprocess_features(_frame(loyalty_tier=tiers), T.FEATURE_COLUMNS)
        except ValueError as e:
            assert "loyalty_tier" in str(e)
        else:
            raise AssertionError(f"tiers {tiers} should have been rejected")


def test_train_model_keeps_only_best_trees():
    """The returned booster must hold exactly the best trees, so plain predict() equals
    what we evaluated, and best_iteration / best_score must survive the slice."""
    import io, contextlib
    import numpy as np
    import xgboost as xgb
    from types import SimpleNamespace
    from sklearn.datasets import make_classification

    X, y = make_classification(3000, 11, n_informative=4, flip_y=0.25, random_state=0)
    cols = [f"f{i}" for i in range(11)]
    args = SimpleNamespace(max_depth=4, eta=0.1, num_round=300, min_child_weight=5, subsample=0.8,
                           colsample_bytree=0.8, scale_pos_weight=None, seed=42,
                           early_stopping_rounds=20, verbose_eval=0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = T.train_model(X[:2000], y[:2000], X[2000:2500], y[2000:2500], args, cols)
    assert model.num_boosted_rounds() == model.best_iteration + 1
    assert 0.5 < model.best_score <= 1.0
    dtest = xgb.DMatrix(X[2500:], feature_names=cols)
    assert np.allclose(model.predict(dtest), T.predict_proba(model, X[2500:], cols))


def test_main_writes_evidence_files_the_collector_expects():
    """Run main() end to end on a synthetic CSV (no AWS, no MLflow) and check the evidence
    files: exact names, and metrics at the TOP LEVEL of evaluation_metrics.json."""
    import contextlib, csv, io, json, os, sys, tempfile
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({c: rng.normal(size=n) for c in T.FEATURE_COLUMNS})
    df["days_since_last_purchase"] = rng.gamma(2.0, 30.0, size=n)
    logit = 0.02 * df["days_since_last_purchase"] - 2.2 + 0.5 * df["purchase_frequency_90d"]
    df["churn_label"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["loyalty_tier"] = rng.choice(T.LOYALTY_TIERS, size=n, p=[0.4, 0.3, 0.2, 0.1])
    df["customer_id"] = [f"C{i:05d}" for i in range(n)]

    with tempfile.TemporaryDirectory() as d:
        data_csv = os.path.join(d, "features.csv")
        df.to_csv(data_csv, index=False)
        out = os.path.join(d, "evidence")
        argv = ["train.py", "--artifacts-bucket", "unused", "--local-data-path", data_csv,
                "--model-dir", os.path.join(d, "model"), "--output-data-dir", out,
                "--verbose-eval", "0"]
        old = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                T.main()
        finally:
            sys.argv = old

        for name in ("evaluation_metrics.json", "feature-importance.png",
                     "confusion-matrix.png", "slice-evaluation.csv"):
            assert os.path.getsize(os.path.join(out, name)) > 0, name
        m = json.load(open(os.path.join(out, "evaluation_metrics.json")))
        for k in ("auc_roc", "precision_top10", "recall_top10", "baseline_auc",
                  "lift_ci_low", "lift_ci_high", "n_val_samples"):
            assert k in m, f"{k} must be a top-level key"
        assert "metrics" not in m
        rows = list(csv.DictReader(open(os.path.join(out, "slice-evaluation.csv"), newline="")))
        assert {r["tier"] for r in rows} == set(T.LOYALTY_TIERS)
        assert set(rows[0]) >= {"tier", "n", "positives", "auc", "recall_top10_global", "too_small"}
        assert set(m["feature_importance_gain"]) == set(T.FEATURE_COLUMNS), "unused features listed at 0"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} passed")
