# Probe 01 results: resource monitor guardrail

**Date:** 2026-10-04 · **Account:** Snowflake trial, AWS us-east-2 · **Script:** [01-resource-monitor.sql](01-resource-monitor.sql)

**Verdict: PASS.** The account guardrail is attached and the trigger mechanism works.
Monitor reaction lags real usage by roughly 15 minutes.

## Part A: real guardrail

| Check | Result |
|---|---|
| `NORTHSTAR_TRIAL_GUARD` exists | yes |
| `level` | `ACCOUNT` |

## Part B: smoke test (`RM_SMOKE_TEST`, quota 1 credit, X-Small warehouse)

| Check | Expected | Observed |
|---|---|---|
| NOTIFY email (trigger 5%) | email arrives | arrived, reported at **22.5%** |
| SUSPEND (trigger 10%) | warehouse suspends | suspended, email reported at **22.5%** |
| Resume after suspend | refused | `Warehouse 'WH_SMOKE_TEST' cannot be resumed because resource monitor 'RM_SMOKE_TEST' has exceeded its quota.` |
| Total credits burned | ~0.1 | **0.34** |
| Cleanup | both dropped | both dropped |

Emails received (verbatim):

> Resource Monitor RM_SMOKE_TEST has reached 22.5 % of its NEVER quota of 1 credits which has triggered a NOTIFY action.

> Resource Monitor RM_SMOKE_TEST has reached 22.5 % of its NEVER quota of 1 credits which has triggered a SUSPEND action.

## Findings

**1. Monitors are not real-time.** Both triggers fired in the same evaluation at 22.5%,
and both thresholds had been crossed before the monitor first looked. Final usage was
0.34 credits against a 0.10 suspend point:

| Point | Credits | X-Small runtime |
|---|---|---|
| Suspend threshold | 0.10 | ~6 min |
| First evaluation | 0.225 | ~13.5 min |
| Final usage | 0.34 | ~20 min |

Observed reaction lag is **~15 minutes** of runtime past the threshold (one sample).

**2. Measure overshoot in time, not percent.** Size the buffer below the hard cap as
*(largest warehouse credit rate) × (lag)*:

| Warehouse running at trigger | Approximate overshoot |
|---|---|
| X-Small (1 credit/hr) | ~0.3 credits |
| Medium (4/hr) | ~1.3 credits |
| Large (8/hr) | ~2.7 credits |

`NORTHSTAR_TRIAL_GUARD` suspends at 90% of 80, leaving an **8-credit buffer**. That is
adequate for the X-Small/Small warehouses this project uses. Revisit it if a lab needs
Large or above.

**3. Suspend locks out everyone, including ACCOUNTADMIN.** Resuming requires raising
`CREDIT_QUOTA` via `ALTER RESOURCE MONITOR`. That is intended, but expect it if it hits
mid-lab.

**4. Coverage gap (unchanged).** This verifies warehouse spend only. Cortex AI, serverless
tasks / dynamic table refresh, and SPCS compute pools are outside resource monitors and
need the `METERING_DAILY_HISTORY` check in [PLAN.md](../PLAN.md) §3.

## AWS comparison note

AWS Budgets *alert* (with hours of billing-data delay) and stop nothing without extra
automation. A Snowflake resource monitor *stops* warehouse compute, also with a lag
(~15 min here). Neither is real-time, so both need a buffer. Snowflake's guarantee is
stronger, but only for warehouses.
