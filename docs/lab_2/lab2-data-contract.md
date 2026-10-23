# Data Contract: processed/customers

| | |
|---|---|
| **Dataset** | `s3://northstar-dev-data-{account-id}/processed/customers/` |
| **Format** | Apache Parquet (Snappy), written with `mode("overwrite")` — each run replaces the full dataset |
| **Contract version** | v1 (2026-10-02) |
| **Owner** | NorthStar data engineering (Lab 2) |

## Producer

Team / process: Glue ETL job `northstar-dev-transform` (Glue 4.0, `glue-scripts/transform.py`),
running as IAM role `northstar-dev-DataEngineer` inside the private subnet.

Input: Glue catalog table `northstar_dev.customers`, registered by crawler
`northstar-dev-raw-crawler` over `raw/customers/` (CSV). The job trims whitespace, converts
blank strings to null, parses both date formats, casts every column, drops rows with no
`customer_id`, imputes remaining nulls, and removes duplicate `transaction_id` rows.

## Consumers

- Feature engineering job `northstar-dev-feature-engineer` — aggregates this dataset to one
  row per customer, writes `features/customers/` and the `northstar-dev-customer-features`
  Feature Group
- (Future) Direct model training in Lab 3

## Grain

One row per transaction. A customer appears on many rows.

`transaction_id` is the natural key. `customer_id` repeating across rows is expected purchase
history, not duplication — the feature engineering job depends on it.

## Schema

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `transaction_id` | string | No | Natural key, unique per row. Format `TXN-{12 uppercase alphanumeric}`. |
| `customer_id` | string | No | Customer key, repeats across rows. Format `CUST-{8 digits}`. Join key for all downstream features. |
| `purchase_date` | date | No | Order date. Source rows arrive as ISO 8601 (`yyyy-MM-dd`) or `MM/dd/yyyy`; both are parsed to a Parquet `DATE`. |
| `order_value` | double | No | Gross order value in USD, 2 decimal places. Source nulls (~4%) are imputed with the column median. |
| `num_items` | int | No | Number of line items in the order. Source nulls are imputed with the rounded column median. |
| `payment_method` | string | No | One of `credit_card`, `debit_card`, `gift_card`, `cash`; `unknown` if missing at source. |
| `channel` | string | No | One of `online`, `store`; `unknown` if missing at source. |
| `store_id` | string | No | `STORE-{3 digits}` for in-store orders, `ONLINE` for e-commerce; `unknown` if missing at source. |
| `product_category` | string | No | Primary category: `Apparel`, `Beauty`, `Electronics`, `Footwear`, `Grocery`, `Home`, `Outdoor`, `Toys`; `unknown` if missing at source (~2% of rows). |

"Nullable: No" is a guarantee of the contract, enforced by the producer. The Parquet file
schema itself may still mark string columns as nullable; consumers should rely on this
contract, not on the Parquet nullability flag.

## Quality Guarantees

Each guarantee is a measurable assertion. The ones marked **(gate)** are asserted inside
`transform.py` before it writes, so a violating run fails instead of publishing bad data.
All of them are checked against the published output by `scripts/verify-lab2.sh`.

| # | Guarantee | How to measure | Measured on sample run |
|---|-----------|----------------|------------------------|
| 1 | `customer_id` is never null **(gate)** | `count(customer_id IS NULL) = 0` | 0 |
| 2 | No duplicate `transaction_id` rows **(gate)** — a `customer_id` repeating across rows is expected, not a defect | `count(*) = count(DISTINCT transaction_id)` | 157,627 = 157,627 |
| 3 | `purchase_date` is a valid ISO 8601 date **(gate)**, within the source extract window | not null; `2025-04-01 <= purchase_date <= 2026-06-30` | 0 nulls; min 2025-04-01, max 2026-06-30 |
| 4 | No nulls in any column | `count(<col> IS NULL) = 0` for all 9 columns | 0 in every column |
| 5 | `order_value` is within expected range | `0 < order_value <= 10,000` (USD) | min 15.00, median 141.75, max 620.00 |
| 6 | `num_items` is within expected range | `1 <= num_items <= 50` | min 1, median 5, max 9 |
| 7 | Transaction grain is preserved | `count(*) > count(DISTINCT customer_id)` | 157,627 rows / 9,999 customers (~15.8 per customer) |
| 8 | Key formats are valid | `transaction_id ~ ^TXN-[A-Z0-9]{12}$`, `customer_id ~ ^CUST-\d{8}$` | 100% / 100% |
| 9 | Categorical columns contain only allowed values | values ⊆ the lists in the schema table above (plus `unknown`) | 100% |
| 10 | `channel` and `store_id` agree | `channel = 'online'` ⇔ `store_id = 'ONLINE'` (rows with `unknown` excluded) | 100% |
| 11 | Row loss from cleaning is bounded | output rows ≥ 95% of raw rows | 157,627 / 163,255 = 96.6% (3,265 null `customer_id` dropped, 2,363 duplicates removed) |

The ranges in 5 and 6 are deliberately wider than the observed data, so they flag corruption
(negative values, unit errors, cents-as-dollars) without failing on normal growth.

Imputed values are not flagged in the output. Consumers that need to tell imputed values from
real ones should not treat `order_value = median` or category `unknown` as observed. The
feature job already excludes `unknown` from `category_diversity_score`.

## SLA

- Data is available in `processed/customers/` within 2 hours of landing in `raw/customers/`.
  The pipeline is crawler then transform job, both on demand; a typical end-to-end run is a
  few minutes, most of it Glue provisioning workers and their ENIs in the private subnet, which
  leaves ample headroom inside the 2-hour window.
- Each run fully replaces the dataset (`overwrite`). Consumers must not read while the transform
  job is `RUNNING`; read once `aws glue get-job-runs --job-name northstar-dev-transform` reports
  `SUCCEEDED`.
- A failed quality gate fails the job run (`JobRunState: FAILED`) and leaves no partial
  replacement for consumers to read as valid; the producer investigates the same business day.

## Versioning

- Schema changes require a new S3 prefix (e.g., `processed/customers/v2/`).
- Breaking changes require consumer notification 5 business days in advance.
- **Breaking** (new prefix + notice): removing or renaming a column, changing a column's type,
  changing the grain, relaxing a quality guarantee (e.g. allowing nulls), or changing the meaning
  of a value (e.g. `order_value` from gross to net).
- **Non-breaking** (same prefix, changelog entry): adding a nullable column at the end of the
  schema, adding a new allowed value to a categorical column, or tightening a guarantee.
- The previous version's prefix stays readable for at least one full consumer release cycle
  after the new version ships, so the feature job can be migrated and verified before cut-over.
- The contract version at the top of this file is bumped together with any change to it.
