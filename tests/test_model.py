"""
NorthStar Retail — Model Evaluation Tests
CS 401R Lab 4 Starter Kit

These tests verify that the churn model meets production quality gates before
it can be promoted through the Model Registry. They run in the CodeBuild
BUILD phase, after the SageMaker Pipeline has trained the model, against the
artifact it produced and the evaluation_metrics.json it emitted.

The key design decision here: we test the MODEL CONTRACT, not the model's
internal implementation. These tests would catch:
  - A model that always predicts "not churned" (high accuracy, useless)
  - A model that performs well on aggregate but fails on a key customer segment
  - A model artifact that can't load or can't produce valid probability scores
  - Prediction drift between model versions

Run locally (after the local training run in the lab handout):
    cp /tmp/out/model_metadata.json /tmp/model/
    pytest tests/test_model.py -v --model-path /tmp/model/xgboost-model \
        --eval-metrics-path /tmp/out/evaluation_metrics.json
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── CLI Option ─────────────────────────────────────────────────────────────────
#
# --model-path is registered in tests/conftest.py, NOT here. pytest_addoption is
# an initialisation hook and is only collected from conftest.py or an installed
# plugin; defining it in a test module silently does nothing, and the run then
# dies with "unrecognized arguments: --model-path" (defect 50).


@pytest.fixture(scope="session")
def model_path(request):
    path = request.config.getoption("--model-path")
    if path is None:
        path = os.environ.get("MODEL_PATH", "./model/model.xgb")
    return path


# ── Fixtures ──────────────────────────────────────────────────────────────────

# These are the model's input features. They must match the FEATURE_COLUMNS in
# churn_training_skeleton.py and every one must exist in the Lab 2 Feature Group.
# churn_label is the target and churn_risk_score is the baseline - neither is an input.
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


@pytest.fixture(scope="module")
def loaded_model(model_path):
    """Load the trained model. Skip if file doesn't exist."""
    if not XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    if not Path(model_path).exists():
        pytest.skip(f"Model file not found at {model_path}. Train a model first.")
    model = xgb.Booster()
    model.load_model(model_path)
    return model


@pytest.fixture(scope="module")
def model_metadata(model_path):
    """Load model metadata if it exists alongside the model."""
    meta_path = Path(model_path).parent / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


@pytest.fixture
def synthetic_churn_batch():
    """
    Synthetic batch of 200 customers with known expected behavior.
    Group A (100): Clear churners — high recency, low frequency, low spend
    Group B (100): Clear retainers — low recency, high frequency, high spend
    """
    np.random.seed(42)
    n = 100

    # Group A: high churn risk
    churners = pd.DataFrame({
        "days_since_last_purchase":     np.random.randint(90, 200, n),
        "purchase_frequency_90d":       np.random.randint(0, 2, n),
        "purchase_frequency_180d":      np.random.randint(0, 3, n),
        "avg_basket_size_6m":           np.random.uniform(20, 60, n),
        "total_spend_90d":              np.random.uniform(0, 50, n),
        "category_diversity_score":     np.random.uniform(0, 0.3, n),
        "online_to_store_ratio":        np.random.uniform(0, 0.2, n),
        "customer_tenure_days":         np.random.randint(30, 365, n),
        "purchase_frequency_30d":       np.zeros(n),
        "avg_order_value":              np.random.uniform(20, 60, n),
        "total_lifetime_value":         np.random.uniform(50, 400, n),
        "true_label":                   np.ones(n),  # Expected: high churn score
    })

    # Group B: low churn risk
    retainers = pd.DataFrame({
        "days_since_last_purchase":     np.random.randint(1, 15, n),
        "purchase_frequency_90d":       np.random.randint(6, 20, n),
        "purchase_frequency_180d":      np.random.randint(10, 35, n),
        "avg_basket_size_6m":           np.random.uniform(120, 400, n),
        "total_spend_90d":              np.random.uniform(400, 2000, n),
        "category_diversity_score":     np.random.uniform(0.5, 1.0, n),
        "online_to_store_ratio":        np.random.uniform(0.4, 0.8, n),
        "customer_tenure_days":         np.random.randint(300, 420, n),
        "purchase_frequency_30d":       np.random.randint(2, 8, n),
        "avg_order_value":              np.random.uniform(120, 400, n),
        "total_lifetime_value":         np.random.uniform(1500, 9000, n),
        "true_label":                   np.zeros(n),  # Expected: low churn score
    })

    return pd.concat([churners, retainers], ignore_index=True)


# ── Model Loading Tests ────────────────────────────────────────────────────────

class TestModelLoading:

    def test_model_loads_successfully(self, loaded_model):
        assert loaded_model is not None

    def test_model_has_expected_feature_count(self, loaded_model):
        """Model should have been trained on exactly 11 features.

        Not 12. Twelve is the *data capture* column count in Lab 6, which adds
        endpointOutput to the 11 model inputs -- a different number for a
        different reason. Conflating them is how the Model Monitor baseline
        ends up on the wrong column count.
        """
        n_features = loaded_model.num_features()
        assert n_features == len(FEATURE_COLUMNS), (
            f"Model has {n_features} features, expected {len(FEATURE_COLUMNS)}. "
            f"Feature columns may have changed."
        )

    def test_model_metadata_exists(self, model_metadata):
        """model_metadata.json should be saved alongside the model."""
        assert model_metadata, (
            "model_metadata.json not found. The training script must save metadata alongside the model artifact."
        )

    def test_model_metadata_has_evaluation_metrics(self, model_metadata):
        if not model_metadata:
            pytest.skip("No metadata available")
        assert "evaluation_metrics" in model_metadata, (
            "Metadata must include evaluation_metrics from training"
        )

    def test_metadata_records_feature_list(self, model_metadata):
        if not model_metadata:
            pytest.skip("No metadata available")
        recorded_features = model_metadata.get("feature_columns", [])
        assert recorded_features == FEATURE_COLUMNS, (
            f"Metadata feature list doesn't match expected. "
            f"Got {recorded_features}, expected {FEATURE_COLUMNS}"
        )


# ── Prediction Output Tests ────────────────────────────────────────────────────

class TestPredictionOutput:

    def test_predictions_are_probabilities(self, loaded_model, synthetic_churn_batch):
        """Output must be probabilities in [0, 1] — not class labels or log-odds."""
        features = synthetic_churn_batch[FEATURE_COLUMNS]
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        predictions = loaded_model.predict(dmatrix)

        assert predictions.min() >= 0.0, f"Predictions below 0: {predictions.min():.4f}"
        assert predictions.max() <= 1.0, f"Predictions above 1: {predictions.max():.4f}"

    def test_predictions_have_no_nans(self, loaded_model, synthetic_churn_batch):
        features = synthetic_churn_batch[FEATURE_COLUMNS]
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        predictions = loaded_model.predict(dmatrix)
        nan_count = np.isnan(predictions).sum()
        assert nan_count == 0, f"{nan_count} NaN predictions — possible missing value issue"

    def test_predictions_use_full_probability_range(self, loaded_model, synthetic_churn_batch):
        """The model should output scores spread across [0.05, 0.95] — not all clustered near 0.5."""
        features = synthetic_churn_batch[FEATURE_COLUMNS]
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        predictions = loaded_model.predict(dmatrix)
        spread = predictions.max() - predictions.min()
        assert spread > 0.4, (
            f"Predictions are too tightly clustered (range: {spread:.2f}). "
            f"Model may be underfitting or always predicting the mean."
        )

    def test_single_sample_inference(self, loaded_model):
        """Model must handle a single-row input without errors."""
        single_sample = pd.DataFrame([{f: 0.0 for f in FEATURE_COLUMNS}])
        # Set some realistic values
        single_sample["days_since_last_purchase"] = 45
        single_sample["purchase_frequency_90d"] = 2
        single_sample["customer_tenure_days"] = 730

        dmatrix = xgb.DMatrix(single_sample, feature_names=FEATURE_COLUMNS)
        pred = loaded_model.predict(dmatrix)
        assert len(pred) == 1
        assert 0 <= pred[0] <= 1


# ── Discrimination Quality Tests ───────────────────────────────────────────────

class TestModelDiscrimination:

    def test_churners_score_higher_than_retainers(self, loaded_model, synthetic_churn_batch):
        """
        The model should assign higher churn scores to the synthetic churner group
        than to the retainer group. This is the most basic sanity check.
        """
        features = synthetic_churn_batch[FEATURE_COLUMNS]
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        predictions = loaded_model.predict(dmatrix)

        churner_mask = synthetic_churn_batch["true_label"] == 1
        retainer_mask = synthetic_churn_batch["true_label"] == 0

        avg_churn_score = predictions[churner_mask].mean()
        avg_retain_score = predictions[retainer_mask].mean()

        assert avg_churn_score > avg_retain_score + 0.15, (
            f"Churner avg score ({avg_churn_score:.3f}) should be significantly higher than "
            f"retainer avg score ({avg_retain_score:.3f}). "
            f"Gap: {avg_churn_score - avg_retain_score:.3f} — expected > 0.15"
        )

    def test_auc_roc_on_synthetic_data(self, loaded_model, synthetic_churn_batch):
        """AUC-ROC on synthetic data with clear signal should be > 0.80."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            pytest.skip("scikit-learn not installed")

        features = synthetic_churn_batch[FEATURE_COLUMNS]
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        predictions = loaded_model.predict(dmatrix)
        labels = synthetic_churn_batch["true_label"].values

        auc = roc_auc_score(labels, predictions)
        assert auc > 0.80, (
            f"AUC-ROC on synthetic data is {auc:.4f} — expected > 0.80 on clearly labeled data. "
            f"Check that feature engineering matches what the model was trained on."
        )


# ── Sensitivity / Robustness Tests ─────────────────────────────────────────────

class TestModelRobustness:

    def test_model_handles_missing_values(self, loaded_model):
        """
        XGBoost handles NaN natively. Verify the model doesn't crash on null features.
        In production, Feature Store may return NaN for customers with no clickstream.
        """
        features_with_nulls = pd.DataFrame([{f: np.nan for f in FEATURE_COLUMNS}])
        features_with_nulls["days_since_last_purchase"] = 30  # at least some real values
        dmatrix = xgb.DMatrix(features_with_nulls, feature_names=FEATURE_COLUMNS)
        pred = loaded_model.predict(dmatrix)
        assert len(pred) == 1
        assert not np.isnan(pred[0]), "Model predicted NaN on null-feature input"

    def test_model_monotone_on_recency(self, loaded_model):
        """
        Increasing days_since_last_purchase (more dormant) should increase churn probability.
        This tests that the model learned the correct direction for the recency feature.

        A monotonicity test: model behaviour testing beyond aggregate metrics.

        A tree ensemble is piecewise constant, so neighbouring recency values
        that fall in the same leaf interval score IDENTICALLY, and small dips
        between neighbours are split noise. The test therefore allows ties and
        dips up to MONO_TOLERANCE, and requires a substantial overall rise.
        (An earlier version counted strictly increasing pairs and failed a
        model scoring 0.230, 0.317, 0.307, 0.307, 0.633, 0.765 -- 2026-10-19.)
        """
        recency_values = [5, 30, 60, 90, 150, 200]
        scores = []

        for days in recency_values:
            features = pd.DataFrame([{
                "days_since_last_purchase": days,
                "purchase_frequency_90d": 3,
                "purchase_frequency_180d": 6,
                "avg_basket_size_6m": 85.0,
                "total_spend_90d": 200.0,
                "category_diversity_score": 0.5,
                "online_to_store_ratio": 0.4,
                "avg_order_value": 120.0,
                "total_lifetime_value": 2400.0,
                "customer_tenure_days": 1000,
                "purchase_frequency_30d": 2,
            }])
            dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
            scores.append(float(loaded_model.predict(dmatrix)[0]))

        MONO_TOLERANCE = 0.02   # largest allowed dip between neighbours
        MIN_RISE = 0.20         # 5 -> 200 days dormant must raise the score this much
        shown = [f"{s:.3f}" for s in scores]
        dips = [(recency_values[i], recency_values[i + 1], round(scores[i] - scores[i + 1], 3))
                for i in range(len(scores) - 1) if scores[i + 1] < scores[i] - MONO_TOLERANCE]
        assert not dips, (
            f"Churn score falls by more than {MONO_TOLERANCE} as recency grows: {dips}. "
            f"Scores {shown}."
        )
        assert scores[-1] - scores[0] >= MIN_RISE, (
            f"Score rises only {scores[-1] - scores[0]:.3f} from {recency_values[0]} to "
            f"{recency_values[-1]} days dormant (need >= {MIN_RISE}). Scores {shown}."
        )

    def test_prediction_stability_across_calls(self, loaded_model):
        """Same input must produce identical output on repeated calls (deterministic)."""
        features = pd.DataFrame([{
            "days_since_last_purchase": 45, "purchase_frequency_90d": 3,
            "purchase_frequency_180d": 6, "avg_basket_size_6m": 85.0,
            "total_spend_90d": 200.0, "category_diversity_score": 0.5,
            "online_to_store_ratio": 0.4, "avg_order_value": 120.0,
            "total_lifetime_value": 2400.0, "customer_tenure_days": 365,
            "purchase_frequency_30d": 2,
        }])
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        pred1 = loaded_model.predict(dmatrix)[0]
        pred2 = loaded_model.predict(dmatrix)[0]
        assert pred1 == pred2, f"Model is non-deterministic: {pred1} != {pred2}"


# ── Stored Metrics Quality Gate ────────────────────────────────────────────────

class TestStoredMetricsGate:
    """
    Reads the evaluation_metrics.json saved by the training script and
    verifies the model passed all required thresholds.
    These tests are the definitive quality gate in the CI pipeline.
    """

    # Aligned with Lab 3's gates as of 2026-08-02.
    #
    # There is deliberately NO auc_roc entry. A 0.72 AUC gate was removed after
    # a 200-split sweep measured the reference model failing it on 58% of
    # splits: it gated the random seed, not the model. AUC is reported, not
    # thresholded.
    #
    # The old precision 0.40 / recall 0.35 pair also did not match Lab 3, which
    # has always asked for 0.50 / 0.25. That mismatch meant a model could pass
    # Lab 3 and fail Lab 4's CI, or the reverse.
    REQUIRED_METRICS = {
        "precision_top10": 0.50,
        "recall_top10": 0.25,
    }

    @pytest.fixture(scope="class")
    def eval_metrics(self):
        metrics_path = os.environ.get("EVAL_METRICS_PATH", "/tmp/eval_metrics.json")
        if not Path(metrics_path).exists():
            pytest.skip(f"Evaluation metrics not found at {metrics_path}.")
        with open(metrics_path) as f:
            return json.load(f)

    def test_precision_top10_meets_threshold(self, eval_metrics):
        p = eval_metrics.get("precision_top10", 0)
        assert p >= self.REQUIRED_METRICS["precision_top10"], (
            f"Precision@Top10 {p:.4f} < required {self.REQUIRED_METRICS['precision_top10']}"
        )

    def test_recall_top10_meets_threshold(self, eval_metrics):
        r = eval_metrics.get("recall_top10", 0)
        assert r >= self.REQUIRED_METRICS["recall_top10"], (
            f"Recall@Top10 {r:.4f} < required {self.REQUIRED_METRICS['recall_top10']}"
        )

    def test_model_beats_baseline(self, eval_metrics):
        """The promotion gate: the lift CI must exclude zero.

        Not a threshold on the point estimate. A lift of +0.05 with a CI of
        [-0.01, 0.11] is not evidence of anything; a lift of +0.02 with a CI
        of [0.005, 0.035] is. The baseline is the recency-only model, not a
        constant predictor - beating a coin flip proves nothing.
        """
        lo = eval_metrics.get("lift_ci_low")
        hi = eval_metrics.get("lift_ci_high")
        assert lo is not None and hi is not None, (
            "lift_ci_low/lift_ci_high missing from evaluation metrics. The "
            "training script must emit a bootstrap CI on the lift over the "
            "recency-only baseline."
        )
        assert lo > 0, (
            f"Lift 95% CI [{lo:.4f}, {hi:.4f}] includes zero. No evidence the "
            f"model beats the recency-only baseline; it will not be promoted."
        )

    def test_positive_rate_is_realistic(self, eval_metrics):
        """The validation set churn rate should be near the measured 22%."""
        rate = eval_metrics.get("positive_rate_val", -1)
        if rate < 0:
            pytest.skip("positive_rate_val not in metrics")
        assert 0.15 <= rate <= 0.30, (
            f"Validation churn rate {rate:.1%} outside expected range [15%, 30%]. "
            f"The reference dataset measures 22.0%. Check that your validation "
            f"split is stratified."
        )


# ── Regression Against the Champion ────────────────────────────────────────────

MODEL_PACKAGE_GROUP = os.environ.get("MODEL_PACKAGE_GROUP", "northstar-churn-models")

# How far the candidate's AUC may fall below the champion's before the build
# fails. AUC wobbles by about this much from split to split, so a zero tolerance
# would fail good models on noise.
REGRESSION_TOLERANCE = 0.02


def find_champion(sm, s3, group=MODEL_PACKAGE_GROUP):
    """Return (version, auc_roc) of the newest Approved package, or None.

    The champion is the newest *Approved* package, never simply the newest. By
    the time this test runs, the pipeline has already registered the candidate
    under test as PendingManualApproval, so "newest" would compare the model
    with itself and always pass.

    Takes the boto3 clients as arguments so the lookup can be tested with fakes.
    Any AWS error propagates: a lookup that cannot run must fail the build, not
    pass it. Only "nothing is Approved yet" returns None.
    """
    resp = sm.list_model_packages(
        ModelPackageGroupName=group, ModelApprovalStatus="Approved",
        SortBy="CreationTime", SortOrder="Descending", MaxResults=1,
    )
    packages = resp["ModelPackageSummaryList"]
    if not packages:
        return None
    summary = packages[0]
    detail = sm.describe_model_package(ModelPackageName=summary["ModelPackageArn"])
    # The pipeline attaches evaluation_metrics.json as the package's ModelMetrics.
    uri = detail["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
    bucket, _, key = uri[len("s3://"):].partition("/")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return summary["ModelPackageVersion"], float(json.loads(body)["auc_roc"])


class _FakeBody:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()


class _FakeSageMaker:
    """Stands in for boto3's SageMaker client: records what it was asked."""

    def __init__(self, approved):
        self.approved = approved   # list of (version, auc), newest first
        self.list_kwargs = None

    def list_model_packages(self, **kwargs):
        self.list_kwargs = kwargs
        newest = self.approved[:1]
        return {"ModelPackageSummaryList": [
            {"ModelPackageArn": f"arn:pkg/{v}", "ModelPackageVersion": v} for v, _ in newest]}

    def describe_model_package(self, ModelPackageName):
        return {"ModelMetrics": {"ModelQuality": {"Statistics": {
            "S3Uri": f"s3://bucket/{ModelPackageName.rsplit('/', 1)[1]}/evaluation_metrics.json"}}}}


class _FakeS3:
    def __init__(self, aucs_by_version):
        self.aucs = aucs_by_version

    def get_object(self, Bucket, Key):
        version = int(Key.split("/")[0])
        return {"Body": _FakeBody({"auc_roc": self.aucs[version]})}


class TestChampionLookup:
    """The lookup logic, with no AWS: runs everywhere, including your laptop."""

    def test_asks_only_for_approved_packages(self):
        sm = _FakeSageMaker([(3, 0.80)])
        find_champion(sm, _FakeS3({3: 0.80}))
        assert sm.list_kwargs["ModelApprovalStatus"] == "Approved"
        assert sm.list_kwargs["SortOrder"] == "Descending"

    def test_returns_version_and_auc_of_newest_approved(self):
        sm = _FakeSageMaker([(3, 0.80), (2, 0.70)])
        assert find_champion(sm, _FakeS3({3: 0.80, 2: 0.70})) == (3, 0.80)

    def test_no_approved_package_returns_none(self):
        assert find_champion(_FakeSageMaker([]), _FakeS3({})) is None


class TestRegressionAgainstChampion:
    """The candidate must not be meaningfully worse than the model in production."""

    @pytest.fixture
    def candidate_metrics(self):
        metrics_path = os.environ.get("EVAL_METRICS_PATH", "/tmp/eval_metrics.json")
        if not Path(metrics_path).exists():
            pytest.skip(f"Evaluation metrics not found at {metrics_path}.")
        with open(metrics_path) as f:
            return json.load(f)

    def test_new_model_does_not_regress_against_champion(self, candidate_metrics):
        """new AUC >= champion AUC - 0.02, with the champion read from the registry.

        Skips (does not fail) in exactly two cases: no AWS credentials, which is
        a laptop run, and no Approved package yet, which is the very first
        build. Approve the first green run's package to arm this test.
        """
        boto3 = pytest.importorskip("boto3")
        from botocore.exceptions import NoCredentialsError

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        try:
            champion = find_champion(boto3.client("sagemaker", region_name=region),
                                     boto3.client("s3", region_name=region))
        except NoCredentialsError:
            pytest.skip("No AWS credentials: cannot read the Model Registry from here.")
        if champion is None:
            pytest.skip(f"No Approved package in {MODEL_PACKAGE_GROUP} yet. Approve a "
                        f"version from an earlier green run to enable this test.")

        version, champ_auc = champion
        new_auc = candidate_metrics["auc_roc"]
        # Printed before asserting, so the build log records which champion this
        # run was compared against. -rw puts it in the warnings summary.
        warnings.warn(f"champion v{version} auc={champ_auc:.4f}; candidate auc={new_auc:.4f}")
        assert new_auc >= champ_auc - REGRESSION_TOLERANCE, (
            f"Candidate AUC {new_auc:.4f} is more than {REGRESSION_TOLERANCE} below "
            f"champion v{version} ({champ_auc:.4f})."
        )
