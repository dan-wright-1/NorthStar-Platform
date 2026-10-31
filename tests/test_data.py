"""
NorthStar Retail — Data Validation Tests
CS 401R Lab 4 Starter Kit

Validates the three datasets in the Lab 2 pipeline:

    raw/customers/        the source CSV, deliberately dirty
    processed/customers/  cleaned Parquet, TRANSACTION grain
    features/customers/   engineered Parquet, CUSTOMER grain, with churn_label

These run in the CodeBuild pre_build phase and MUST pass before any model
training is triggered.

Run locally:
    export NORTHSTAR_RAW=./data/northstar-raw-sample.csv
    export NORTHSTAR_PROCESSED=./data/processed
    export NORTHSTAR_FEATURES=./data/features
    pytest tests/test_data.py -v

Or point them at the pipeline output after a run:
    aws s3 cp s3://$BUCKET/processed/customers/ ./data/processed --recursive
    aws s3 cp s3://$BUCKET/features/customers/  ./data/features  --recursive

DESIGN PRINCIPLE — read this before you edit the fixtures.

A missing dataset is a FAILURE, not a skip. An earlier version of this file
called pytest.skip() when a file was absent, which meant `pytest tests/` went
green having tested nothing at all. A CI gate that cannot fail is not a gate,
and a green build that validated no data is worse than a red one because it
gets trusted.

If you need to work on other tests before your pipeline has produced output,
set ALLOW_MISSING_DATA=1 explicitly. Never make absence silent by default.
"""

import glob
import os
import time
from pathlib import Path

import pandas as pd
import pytest

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_PATH = os.environ.get("NORTHSTAR_RAW", "./data/northstar-raw-sample.csv")
PROCESSED_DIR = os.environ.get("NORTHSTAR_PROCESSED", "./data/processed")
FEATURES_DIR = os.environ.get("NORTHSTAR_FEATURES", "./data/features")

ALLOW_MISSING = os.environ.get("ALLOW_MISSING_DATA") == "1"

# Must match the Lab 2 feature job. Windows are measured backwards from the
# cutoff; the label comes from the holdout after it.
FEATURE_CUTOFF = pd.Timestamp("2026-04-01")
WINDOW_START = pd.Timestamp("2025-04-01")
SNAPSHOT = pd.Timestamp("2026-06-30")

RAW_COLUMNS = [
    "transaction_id", "customer_id", "purchase_date", "order_value",
    "num_items", "payment_method", "channel", "store_id", "product_category",
]

FEATURE_COLUMNS = [
    "days_since_last_purchase", "customer_tenure_days",
    "purchase_frequency_30d", "purchase_frequency_90d", "purchase_frequency_180d",
    "avg_order_value", "total_spend_90d", "total_lifetime_value",
    "avg_basket_size_6m", "category_diversity_score", "online_to_store_ratio",
]

VALID_TIERS = {"Bronze", "Silver", "Gold", "Platinum"}
VALID_CHANNELS = {"store", "online", "unknown"}


def _require(path_desc, found):
    """Fail loudly unless the student explicitly opted into deferral."""
    if found:
        return
    msg = (
        f"{path_desc} not found. Run the Lab 2 pipeline and download its output, "
        f"or set the NORTHSTAR_* environment variables. "
        f"Set ALLOW_MISSING_DATA=1 only if you are deliberately deferring this."
    )
    if ALLOW_MISSING:
        pytest.skip(msg)
    pytest.fail(msg, pytrace=False)


MAX_FEATURE_AGE_HOURS = 48


def feature_age_hours(features, now=None):
    """Age in hours of the newest feature record. event_time is epoch seconds."""
    now = time.time() if now is None else now
    return (now - float(features["event_time"].max())) / 3600.0


def _read_parquet_dir(directory):
    files = glob.glob(str(Path(directory) / "*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    _require(f"Raw CSV ({RAW_PATH})", Path(RAW_PATH).exists())
    return pd.read_csv(RAW_PATH)


@pytest.fixture(scope="module")
def processed_df():
    df = _read_parquet_dir(PROCESSED_DIR)
    _require(f"Processed Parquet ({PROCESSED_DIR})", df is not None)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    return df


@pytest.fixture(scope="module")
def features_df():
    df = _read_parquet_dir(FEATURES_DIR)
    _require(f"Features Parquet ({FEATURES_DIR})", df is not None)
    return df


# ── Raw data ──────────────────────────────────────────────────────────────────

class TestRawData:
    """The raw file is SUPPOSED to be dirty. These assert its shape and confirm
    the defects the transform is meant to clean are actually present - a raw
    file that is already clean means the pipeline is never exercised."""

    def test_schema(self, raw_df):
        missing = [c for c in RAW_COLUMNS if c not in raw_df.columns]
        assert not missing, f"Raw CSV missing columns: {missing}"

    def test_row_count_plausible(self, raw_df):
        """Shape, not absolute size.

        This used to assert `5_000 < len(raw_df) < 60_000`, a bound derived from
        the retired 1,200-customer sample. The 2026-08-02 rebase to 10,000
        customers produced 163,255 raw rows and this test failed on the
        correct dataset (defect 51).

        Transactions-per-customer is the property actually worth asserting: it
        survives any regeneration of the sample at a different `--customers`
        value, which an absolute row count cannot.
        """
        n = len(raw_df)
        customers = raw_df["customer_id"].nunique()
        assert n > 5_000, f"Raw file is implausibly small: {n} rows"
        per_customer = n / max(customers, 1)
        assert 5 < per_customer < 40, (
            f"Raw file averages {per_customer:.1f} transactions per customer "
            f"({n:,} rows / {customers:,} customers). The generator produces "
            f"~14. A value outside [5, 40] means the sample was generated with "
            f"different parameters than the labs assume."
        )

    def test_contains_defects_to_clean(self, raw_df):
        nulls = raw_df["customer_id"].isna().sum()
        dupes = raw_df["transaction_id"].duplicated().sum()
        assert nulls > 0, "Expected null customer_id values in the raw data"
        assert dupes > 0, "Expected duplicate transaction_id rows in the raw data"

    def test_contains_mixed_date_formats(self, raw_df):
        slashed = raw_df["purchase_date"].astype(str).str.contains("/").sum()
        assert slashed > 0, "Expected some MM/DD/YYYY dates to force real parsing"


# ── Processed data (transaction grain) ────────────────────────────────────────

class TestProcessedData:

    def test_schema(self, processed_df):
        missing = [c for c in RAW_COLUMNS if c not in processed_df.columns]
        assert not missing, f"Processed data missing columns: {missing}"

    def test_no_null_customer_id(self, processed_df):
        n = processed_df["customer_id"].isna().sum()
        assert n == 0, f"{n} rows with null customer_id survived the transform"

    def test_no_duplicate_transactions(self, processed_df):
        n = processed_df["transaction_id"].duplicated().sum()
        assert n == 0, f"{n} duplicate transaction_id rows survived deduplication"

    def test_all_dates_parsed(self, processed_df):
        n = processed_df["purchase_date"].isna().sum()
        assert n == 0, f"{n} unparsed purchase_date values"

    def test_no_null_order_value(self, processed_df):
        n = processed_df["order_value"].isna().sum()
        assert n == 0, f"{n} null order_value survived imputation"

    def test_order_value_in_range(self, processed_df):
        assert processed_df["order_value"].min() >= 0, "Negative order_value found"
        assert processed_df["order_value"].max() < 10_000, "Implausible order_value"

    def test_customer_id_whitespace_trimmed(self, processed_df):
        stripped = processed_df["customer_id"].str.strip()
        n = (processed_df["customer_id"] != stripped).sum()
        assert n == 0, f"{n} customer_id values still carry whitespace"

    def test_channel_values_valid(self, processed_df):
        bad = set(processed_df["channel"].unique()) - VALID_CHANNELS
        assert not bad, f"Unexpected channel values: {bad}"

    def test_transaction_grain_preserved(self, processed_df):
        """The single most damaging Lab 2 error is deduplicating on customer_id.
        It looks clean and makes the Task 3 aggregates impossible to compute."""
        rows, customers = len(processed_df), processed_df["customer_id"].nunique()
        assert rows > customers * 2, (
            f"{rows} rows for {customers} customers. Processed data must stay at "
            f"TRANSACTION grain - deduplicate on transaction_id, not customer_id."
        )


# ── Features (customer grain) ─────────────────────────────────────────────────

class TestFeatures:

    def test_schema(self, features_df):
        required = FEATURE_COLUMNS + ["customer_id", "loyalty_tier", "churn_label"]
        missing = [c for c in required if c not in features_df.columns]
        assert not missing, f"Feature set missing columns: {missing}"

    def test_one_row_per_customer(self, features_df):
        assert len(features_df) == features_df["customer_id"].nunique(), (
            "Feature set must be exactly one row per customer"
        )

    def test_no_nulls(self, features_df):
        nulls = features_df.isna().sum()
        offenders = nulls[nulls > 0].to_dict()
        assert not offenders, f"Null feature values: {offenders}"

    def test_all_tiers_present(self, features_df):
        tiers = set(features_df["loyalty_tier"].unique())
        assert tiers == VALID_TIERS, f"Degenerate tier distribution: {sorted(tiers)}"

    def test_churn_label_binary(self, features_df):
        vals = set(int(v) for v in features_df["churn_label"].unique())
        assert vals <= {0, 1}, f"churn_label must be 0/1, found {vals}"

    def test_churn_rate_plausible(self, features_df):
        rate = features_df["churn_label"].mean()
        assert 0.15 <= rate <= 0.30, (
            f"Churn rate {rate:.1%} is outside the expected 15-30% band. "
            f"Check the holdout window in your feature job."
        )

    def test_bounded_features_in_range(self, features_df):
        for col in ["category_diversity_score", "online_to_store_ratio"]:
            assert features_df[col].between(0, 1).all(), f"{col} outside [0, 1]"
        if "churn_risk_score" in features_df.columns:
            assert features_df["churn_risk_score"].between(0, 1).all(), (
                "churn_risk_score outside [0, 1]"
            )

    def test_non_negative_counts(self, features_df):
        for col in ["purchase_frequency_30d", "purchase_frequency_90d",
                    "purchase_frequency_180d", "total_spend_90d",
                    "total_lifetime_value", "avg_basket_size_6m"]:
            assert (features_df[col] >= 0).all(), f"{col} has negative values"

    def test_frequency_windows_are_monotonic(self, features_df):
        """A 180-day window cannot contain fewer purchases than a 90-day one."""
        assert (features_df["purchase_frequency_180d"]
                >= features_df["purchase_frequency_90d"]).all(), \
            "purchase_frequency_180d < purchase_frequency_90d for some customers"
        assert (features_df["purchase_frequency_90d"]
                >= features_df["purchase_frequency_30d"]).all(), \
            "purchase_frequency_90d < purchase_frequency_30d for some customers"

    def test_label_not_trivially_separable(self, features_df):
        """Leakage smoke test.

        If every churner has worse recency than every retained customer, the
        label was computed over the same window as the features. A model trained
        on that posts a near-perfect AUC and collapses in production.
        """
        churned = features_df[features_df["churn_label"] == 1]["days_since_last_purchase"]
        active = features_df[features_df["churn_label"] == 0]["days_since_last_purchase"]
        assert churned.min() < active.max(), (
            "churn_label is perfectly separable by recency alone - the temporal "
            "split was not applied. Features must come from the observation "
            "window and the label from the holdout."
        )

    def test_recency_within_observation_window(self, features_df):
        """Recency is measured to the cutoff, so it cannot exceed the window."""
        max_window = (FEATURE_CUTOFF - WINDOW_START).days
        assert features_df["days_since_last_purchase"].max() <= max_window, (
            "days_since_last_purchase exceeds the observation window - check that "
            "you anchored to FEATURE_CUTOFF and not to today()"
        )

    def test_no_feature_used_post_cutoff_data(self, processed_df, features_df):
        """Recompute the time-sensitive features from pre-cutoff rows only and
        require them to equal what the Glue job wrote.

        processed/ CONTAINS the holdout purchases (after FEATURE_CUTOFF), because
        the label is built from them, so "every customer's max purchase_date is
        <= FEATURE_CUTOFF" is false on correct data. The test is on the FEATURES:
        if the Glue job let a holdout purchase into recency, tenure or a
        frequency window, the pandas recomputation (which drops everything after
        the cutoff) would disagree for that customer.
        """
        from data.feature_engineering import build_churn_feature_set

        recomputed = build_churn_feature_set(
            features_df[["customer_id"]], processed_df, FEATURE_CUTOFF)
        time_sensitive = ["days_since_last_purchase", "customer_tenure_days",
                          "purchase_frequency_30d", "purchase_frequency_90d",
                          "purchase_frequency_180d"]
        merged = features_df[["customer_id"] + time_sensitive].merge(
            recomputed[["customer_id"] + time_sensitive], on="customer_id",
            suffixes=("_glue", "_pandas"))
        assert len(merged) == len(features_df), "customers lost in the join"
        for col in time_sensitive:
            bad = merged[merged[f"{col}_glue"] != merged[f"{col}_pandas"]]
            assert bad.empty, (
                f"{col} differs from the pre-cutoff recomputation for {len(bad)} "
                f"customers (first: {bad.iloc[0]['customer_id']}). The feature job "
                f"may be reading purchases after FEATURE_CUTOFF.")

    def test_features_are_fresh(self, features_df):
        """The pipeline must refuse to train on a feature set older than 48 hours.

        event_time is Fractional epoch seconds: the wall-clock time the Lab 2
        feature job ran, NOT the cutoff T. Consequence: re-run the
        northstar-dev-feature-engineer Glue job (~90 s) before any session in
        which you will push, or this test fails the build.
        """
        age = feature_age_hours(features_df)
        assert age < MAX_FEATURE_AGE_HOURS, (
            f"Newest feature record is {age:.1f} hours old (limit "
            f"{MAX_FEATURE_AGE_HOURS}). Re-run the northstar-dev-feature-engineer "
            f"Glue job.")

    def test_staleness_check_rejects_old_features(self):
        """The freshness check can fail: a set whose newest record is 49 hours
        old is over the limit, and one from an hour ago is not."""
        now = 1_800_000_000.0
        stale = pd.DataFrame({"event_time": [now - 49 * 3600, now - 60 * 3600]})
        fresh = pd.DataFrame({"event_time": [now - 3600, now - 60 * 3600]})
        assert feature_age_hours(stale, now=now) >= MAX_FEATURE_AGE_HOURS
        assert feature_age_hours(fresh, now=now) < MAX_FEATURE_AGE_HOURS


# ── Cross-dataset consistency ─────────────────────────────────────────────────

class TestConsistency:

    def test_feature_customers_are_subset_of_processed(self, processed_df, features_df):
        proc = set(processed_df["customer_id"].unique())
        feat = set(features_df["customer_id"].unique())
        orphans = feat - proc
        assert not orphans, f"{len(orphans)} customers in features but not in processed"

    def test_row_survival_rate(self, raw_df, processed_df):
        """Only nulls and duplicates should be removed. Losing much more than
        that means the transform is dropping valid data."""
        survival = len(processed_df) / len(raw_df)
        assert 0.85 <= survival <= 1.0, (
            f"Only {survival:.1%} of raw rows survived to processed. Expected "
            f"85-100%; the transform should remove only null-key and duplicate rows."
        )
