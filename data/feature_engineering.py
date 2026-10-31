"""Pandas port of the Lab 2 feature logic (glue-scripts/feature_engineer.py).

The Glue job computes every feature in one Spark groupBy. This module splits
the same definitions into small pandas functions so tests can feed them tiny,
hand-built DataFrames. Definitions must stay identical to the Glue job:

  * every window is measured backwards from the snapshot date T, never today()
  * a window of N days is "days_before < N" (T itself is day 0)
  * history is purchase_date <= T; anything later is the label holdout
  * category_diversity_score excludes the imputed 'unknown' category
"""

from __future__ import annotations

import pandas as pd

N_CATEGORIES = 8.0          # denominator for category_diversity_score
NO_PURCHASE_SENTINEL = 999  # days_since_last_purchase for a customer with no orders


def _with_days_before(transactions: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Copy of `transactions` with a whole-number `_days_before` column.

    Glue works on DATE columns, so time-of-day is dropped on both sides before
    subtracting. Each feature function calls this instead of repeating the
    date arithmetic.
    """
    df = transactions.copy()
    purchased = pd.to_datetime(df["purchase_date"]).dt.normalize()
    df["_days_before"] = (pd.Timestamp(snapshot_date).normalize() - purchased).dt.days
    return df


def compute_recency_features(transactions: pd.DataFrame, snapshot_date,
                             customer_ids: pd.DataFrame | None = None) -> pd.DataFrame:
    """days_since_last_purchase and customer_tenure_days, one row per customer.

    If `customer_ids` is given, every id in it gets a row; customers with no
    transactions get the 999 sentinel (tenure 0) instead of silently vanishing.
    """
    df = _with_days_before(transactions, snapshot_date)
    grouped = df.groupby("customer_id")["_days_before"]
    out = pd.DataFrame({
        "days_since_last_purchase": grouped.min(),   # smallest gap = latest order
        "customer_tenure_days": grouped.max(),       # largest gap = first order
    }).reset_index()

    if customer_ids is not None:
        out = customer_ids[["customer_id"]].merge(out, on="customer_id", how="left")
        out["days_since_last_purchase"] = out["days_since_last_purchase"].fillna(NO_PURCHASE_SENTINEL)
        out["customer_tenure_days"] = out["customer_tenure_days"].fillna(0)
    return out


def compute_frequency_features(transactions: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Order counts in the 30, 90 and 180 days before T, one row per customer.

    Summing a boolean column counts the True rows, so a customer with orders
    only outside a window still gets a row with 0 for that window.
    """
    df = _with_days_before(transactions, snapshot_date)
    for days in (30, 90, 180):
        df[f"purchase_frequency_{days}d"] = df["_days_before"] < days
    cols = [f"purchase_frequency_{d}d" for d in (30, 90, 180)]
    return df.groupby("customer_id")[cols].sum().astype(int).reset_index()


def compute_monetary_features(transactions: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """avg_order_value, total_spend_90d, total_lifetime_value, avg_basket_size_6m.

    avg_basket_size_6m is ITEMS per order over 180 days, not dollars. The
    denominator is floored at 1 so a customer with no orders in the window
    gets 0.0 instead of a divide-by-zero.
    """
    df = _with_days_before(transactions, snapshot_date)
    in_90 = df["_days_before"] < 90
    in_180 = df["_days_before"] < 180
    df["_spend_90d"] = df["order_value"].where(in_90, 0.0)
    df["_items_180d"] = df["num_items"].where(in_180, 0)
    df["_orders_180d"] = in_180.astype(int)

    g = df.groupby("customer_id")
    out = pd.DataFrame({
        "avg_order_value": g["order_value"].mean(),
        "total_spend_90d": g["_spend_90d"].sum(),
        "total_lifetime_value": g["order_value"].sum(),
        "_items_180d": g["_items_180d"].sum(),
        "_orders_180d": g["_orders_180d"].sum(),
    })
    out["avg_basket_size_6m"] = out["_items_180d"] / out["_orders_180d"].clip(lower=1)
    return out.drop(columns=["_items_180d", "_orders_180d"]).reset_index()


def compute_behavioral_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """category_diversity_score and online_to_store_ratio, over all history.

    No snapshot date: neither feature is windowed in the Glue job. 'unknown'
    is dropped before counting categories (nunique ignores the resulting
    NaNs), otherwise the score can exceed 1.0. Despite its name,
    online_to_store_ratio is the fraction of orders that were online.
    """
    df = transactions.copy()
    df["_category"] = df["product_category"].where(df["product_category"] != "unknown")
    df["_is_online"] = (df["channel"] == "online").astype(float)
    g = df.groupby("customer_id")
    out = pd.DataFrame({
        "category_diversity_score": g["_category"].nunique() / N_CATEGORIES,
        "online_to_store_ratio": g["_is_online"].mean(),
    })
    return out.reset_index()


def build_churn_feature_set(customers: pd.DataFrame, transactions: pd.DataFrame,
                            snapshot_date) -> pd.DataFrame:
    """One row per customer in `customers`, with all 11 model features.

    Steps: keep only history (purchase_date <= T), compute the four feature
    groups, left-join them onto the roster, then fill the gaps for customers
    with no history. Null strategy: no orders means recency is the 999
    sentinel and every count, spend, ratio and score is 0.
    """
    T = pd.Timestamp(snapshot_date).normalize()
    purchased = pd.to_datetime(transactions["purchase_date"]).dt.normalize()
    history = transactions[purchased <= T]

    features = customers.copy()
    for part in (
        compute_recency_features(history, snapshot_date),
        compute_frequency_features(history, snapshot_date),
        compute_monetary_features(history, snapshot_date),
        compute_behavioral_features(history),
    ):
        features = features.merge(part, on="customer_id", how="left")

    features["days_since_last_purchase"] = features["days_since_last_purchase"].fillna(NO_PURCHASE_SENTINEL)
    fill_zero = [c for c in features.columns
                 if c not in customers.columns and c != "days_since_last_purchase"]
    features[fill_zero] = features[fill_zero].fillna(0)
    return features
