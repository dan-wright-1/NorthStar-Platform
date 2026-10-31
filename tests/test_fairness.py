"""
NorthStar Retail -- Fairness Check (slice evaluation by loyalty tier)
CS 401R Lab 4, Task 1

Measures whether the churn model finds churners equally well in every loyalty
tier. It FLAGS a gap, it does not block the build: the point is that the person
approving the model sees the gap, not that the build stays quiet. Expect the
flag to fire on this data.

Where the numbers come from: train() in models/churn/churn_model.py puts a
per-tier breakdown in metrics["slices"]["tiers"] (via evaluate_slices). This
file reads that from the same evaluation_metrics.json as test_model.py
(EVAL_METRICS_PATH), so no model or data has to be loaded here.

    recall_top10_global   recall at the SAME top-10% score cutoff the business
                          uses, so it is what each tier actually receives.

Run locally:
    pytest tests/test_fairness.py -v -rw --eval-metrics-path /tmp/out/evaluation_metrics.json
"""

import json
import os
import warnings
from pathlib import Path

import pytest

TIERS = ("Bronze", "Silver", "Gold", "Platinum")

# Flag when the best-served tier's recall exceeds the worst-served tier's by
# more than this many probability points (0.10 = 10 percentage points).
MAX_RECALL_GAP = 0.10


@pytest.fixture
def tiers():
    """Per-tier slice results from the training run's metrics file."""
    metrics_path = os.environ.get("EVAL_METRICS_PATH", "/tmp/eval_metrics.json")
    if not Path(metrics_path).exists():
        pytest.skip(f"Evaluation metrics not found at {metrics_path}.")
    with open(metrics_path) as f:
        metrics = json.load(f)
    # Missing slices is a FAILURE, not a skip: it means train() stopped emitting
    # them, and a fairness check that silently checks nothing is worse than none.
    assert "slices" in metrics and metrics["slices"].get("tiers"), (
        "evaluation metrics have no slices.tiers. train() must put the output of "
        "evaluate_slices() under metrics['slices'].")
    return metrics["slices"]["tiers"]


def recall_gap(tiers):
    """(gap, best_tier, worst_tier) over tiers that have a recall to compare."""
    recalls = {t: r["recall_top10_global"] for t, r in tiers.items()
               if r["recall_top10_global"] is not None}
    best = max(recalls, key=recalls.get)
    worst = min(recalls, key=recalls.get)
    return recalls[best] - recalls[worst], best, worst


class TestSliceCoverage:

    def test_every_loyalty_tier_is_evaluated(self, tiers):
        missing = [t for t in TIERS if t not in tiers]
        assert not missing, f"No slice results for tiers: {missing}"

    def test_slices_account_for_every_validation_row(self, tiers):
        """Tier sizes sum to the validation set: no rows dropped by a null tier."""
        metrics_path = os.environ.get("EVAL_METRICS_PATH", "/tmp/eval_metrics.json")
        with open(metrics_path) as f:
            n_val = json.load(f)["n_val_samples"]
        assert sum(r["n"] for r in tiers.values()) == n_val

    def test_tiny_tiers_are_marked_not_hidden(self, tiers):
        """A tier with too few churners to support a claim must say so."""
        for name, r in tiers.items():
            assert r["too_small"] == (r["n_pos"] < 30), (
                f"{name}: too_small={r['too_small']} but n_pos={r['n_pos']}")


class TestRecallParity:

    def test_recall_gap_between_tiers_is_flagged_not_blocking(self, tiers):
        """Warn when recall@top10% differs by more than 10 points across tiers.

        This test never fails on the size of the gap: a failing fairness check
        would stop the model reaching an approver who can decide what to do
        about it. The warning lands in the pytest warnings summary (-rw), which
        is where the build log shows it.
        """
        gap, best, worst = recall_gap(tiers)
        assert 0.0 <= gap <= 1.0

        if gap > MAX_RECALL_GAP:
            small = [t for t in (best, worst) if tiers[t]["too_small"]]
            note = (f" (note: {', '.join(small)} has under 30 churners, so treat this "
                    f"as indicative)") if small else ""
            warnings.warn(
                f"FAIRNESS FLAG: recall@top10% gap is {gap * 100:.1f} points "
                f"(> {MAX_RECALL_GAP * 100:.0f}): {best} {tiers[best]['recall_top10_global']:.3f} "
                f"vs {worst} {tiers[worst]['recall_top10_global']:.3f}{note}")
