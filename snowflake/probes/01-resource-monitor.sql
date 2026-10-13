-- Probe 01: resource monitor guardrail
--
-- Proves that the account's credit guardrail exists, is attached at account
-- level, and actually fires: NOTIFY emails arrive, SUSPEND stops the
-- warehouse, and a suspended warehouse cannot be resumed.
--
-- Run in a Snowsight worksheet one step at a time (Ctrl+Enter runs only the
-- statement under the cursor). The Run button does NOT run the whole sheet;
-- use Run All (Ctrl+Shift+Enter) for multi-statement blocks.
--
-- Cost: ~0.35 credits (~$1). See 01-resource-monitor-results.md.

USE ROLE ACCOUNTADMIN;

------------------------------------------------------------------------------
-- Part A: the real guardrail (one-time setup, kept permanently)
------------------------------------------------------------------------------

CREATE RESOURCE MONITOR IF NOT EXISTS northstar_trial_guard
  WITH CREDIT_QUOTA = 80
       FREQUENCY = NEVER          -- one total cap for the trial, not monthly
       START_TIMESTAMP = IMMEDIATELY
  TRIGGERS ON 50 PERCENT DO NOTIFY
           ON 75 PERCENT DO NOTIFY
           ON 90 PERCENT DO SUSPEND            -- let running queries finish
           ON 100 PERCENT DO SUSPEND_IMMEDIATE;

ALTER ACCOUNT SET RESOURCE_MONITOR = northstar_trial_guard;

-- Expect: level = ACCOUNT, credit_quota = 80, frequency = NEVER,
-- notify_at = 50%,75%, suspend_at = 90%, suspend_immediately_at = 100%.
-- A blank level means the ALTER ACCOUNT above never ran.
SHOW RESOURCE MONITORS LIKE 'NORTHSTAR_TRIAL_GUARD';

------------------------------------------------------------------------------
-- Part B: smoke test on a throwaway monitor + warehouse
------------------------------------------------------------------------------

-- Prerequisite: Snowsight > Settings > Notifications has resource monitor
-- alerts enabled and a verified email. Otherwise NOTIFY fires silently.

CREATE OR REPLACE RESOURCE MONITOR rm_smoke_test
  WITH CREDIT_QUOTA = 1 FREQUENCY = NEVER START_TIMESTAMP = IMMEDIATELY
  TRIGGERS ON 5 PERCENT DO NOTIFY          -- ~3 min of X-Small
           ON 10 PERCENT DO SUSPEND;       -- ~6 min of X-Small

CREATE OR REPLACE WAREHOUSE wh_smoke_test
  WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 900 INITIALLY_SUSPENDED = TRUE
  RESOURCE_MONITOR = rm_smoke_test;

-- A started warehouse bills per second even while idle: a controlled burn.
ALTER WAREHOUSE wh_smoke_test RESUME;

-- Wait 15-20 minutes. Monitor evaluation lags (see results), so allow more
-- time than the trigger math suggests.

SHOW RESOURCE MONITORS LIKE 'RM_SMOKE_TEST';   -- used_credits >= 0.1
SHOW WAREHOUSES LIKE 'WH_SMOKE_TEST';          -- state = SUSPENDED

-- Expect FAILURE: "cannot be resumed because resource monitor
-- 'RM_SMOKE_TEST' has exceeded its quota". The error is the pass condition.
ALTER WAREHOUSE wh_smoke_test RESUME;

------------------------------------------------------------------------------
-- Cleanup: drop BOTH. A leftover warehouse without its monitor is guarded
-- only by the account-level monitor.
------------------------------------------------------------------------------

DROP WAREHOUSE IF EXISTS wh_smoke_test;
DROP RESOURCE MONITOR IF EXISTS rm_smoke_test;
