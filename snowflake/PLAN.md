# NorthStar on Snowflake — Parallel Track Plan

A second implementation of the CS 401R NorthStar platform, rebuilt on Snowflake one lab
at a time. This snowflake folder is out of scope if we are working on a lab.


The goal is to compare the two stacks honestly, not to replace the graded AWS
work. Nothing here is built yet. This document sets up the design so the build goes well.

> **Why this track makes sense for NorthStar:** the case overview says NorthStar *already
> runs Snowflake* as its warehouse, fed by nightly ETL. The course puts "the Snowflake
> integration" out of scope and picks AWS. A real architect would be asked on day one:
> *"Why not build the AI platform where the data already lives?"* This track tests that
> question with working code.

> **Guiding principle: compare best to best.** Don't mirror the AWS architecture. For
> each lab, take the *outcome* the lab requires (e.g. "access boundaries between data
> tiers", "versioned, recoverable data") and build it the way Snowflake would. A
> faithful copy of an AWS design running on Snowflake makes Snowflake look worse than it
> is, and tells us nothing. Where the native approach differs from AWS, that difference
> is the comparison.

---

## 0. Assumptions and the decision case

### Stated assumptions

The rebuild's conclusions only hold under these. Every comparison write-up should say
which of them a result depends on.

| # | Assumption | Source | If it were false |
|---|---|---|---|
| **A1** | **The data already lives in Snowflake.** NorthStar's POS, e-commerce, CRM and loyalty data land in a Snowflake warehouse through nightly ETL. | Case overview, "Data Architecture (Current State)" | The AWS build has to copy that data into S3 first: a second copy, a second pipeline, and two places to govern. If the data were in S3, Snowflake would have the copying problem instead. |
| **A2** | **The team already knows Snowflake**: SQL, roles, warehouses, the existing ETL. | Implied by A1: someone runs that warehouse today | The learning curve, hiring, and on-call burden move to whichever platform the team doesn't know. |
| **A3** | **The AI platform is new.** No AWS ML estate (SageMaker pipelines, Glue jobs, IAM conventions) exists to reuse. | Case overview: "the AI infrastructure that does not yet exist" | Existing AWS investment becomes a switching cost Snowflake has to overcome. |
| **A4** | **This isn't AWS *or* Snowflake.** Snowflake runs on AWS (this account is in us-east-2). The real choice is AWS-native services vs Snowflake-managed services, often with AWS still underneath. | This account | n/a: a framing correction, not a bet |

The course states "NorthStar has approved AWS as the cloud provider." Under A4 that
approval doesn't rule Snowflake out, since Snowflake on AWS is still AWS.

### The decision depends on where you start

A strong case says *when* the answer changes. A case that says "Snowflake always wins"
reads like a vendor pitch. Three starting points to argue, each with its own default:

| Scenario | Starting point | Default answer | Why | What would change it |
|---|---|---|---|---|
| **S1: NorthStar as written** | A1–A3 true | **Build on Snowflake** | No data copying, one governance model, skills already exist, and there's no network layer to run (Labs 1–2 evidence) | A hard requirement Snowflake can't meet (e.g. real-time serving latency, a model that won't run in its runtimes) |
| **S2: Greenfield** | No data platform yet | **Depends on the workload** | Snowflake: SQL-first analytics and ML on tabular data, low ops. AWS: custom or deep-learning ML, real-time systems, broad service choice | Team skills; mix of tabular vs unstructured or real-time workloads |
| **S3: AWS-mature org** | Data in S3, team skilled in AWS | **Stay on AWS** unless the case clears a high bar | Switching costs (migration, retraining, rewriting IAM and pipelines) are real and come first; the benefits arrive later | Ops or governance burden on AWS measured as larger than the switching cost; a hybrid (Iceberg tables shared between both) may beat a full switch |

### Building the case, lab by lab

Each lab's comparison (§5) adds evidence to this table. By the end it becomes the
recommendation to NorthStar's CDO and CFO.

| Factor | Question it answers | Evidence so far |
|---|---|---|
| Data gravity | How much data movement does each platform need? | A1: Snowflake needs none; AWS needs an S3 copy + ETL |
| Ops burden | How much infrastructure do you run? | Lab 1–2: network layer, crawler, offline-store sync all disappear |
| Governance | One policy model, or several to keep in sync? | Lab 2: Snowflake uses grants + masking + DMFs; AWS uses IAM + CloudWatch, and would need Lake Formation for column-level masking (not built in the labs) |
| Cost | Total cost at NorthStar scale, not trial scale | Pending: needs the weekly metering data |
| Skills | Who can build and run it on day one? | A2 |
| Capability gaps | What can't one platform do? | Trial blockers so far (online features, DMFs); real-time serving TBD (Lab 5+) |
| Lock-in | How hard is it to leave? | Open: Iceberg as the mitigation to evaluate |

---

## 1. The mental-model shift (read this first)

| | AWS (what you built) | Snowflake |
|---|---|---|
| **Shape** | You put a platform together from separate services (S3, Glue, IAM, VPC, SageMaker) and connect them | You get one platform; you create *objects* inside it (databases, warehouses, roles, tasks, models) |
| **Storage vs compute** | Separate services, separate bills | Separate inside one product: storage is always there, compute (a **warehouse**) starts and stops on demand |
| **Language of the platform** | API calls + IAM JSON | SQL. Almost every object can be created with `CREATE ...` and secured with `GRANT ...` |
| **Security boundary** | Network (VPC, subnets, SGs) + IAM policies | Identity + RBAC. There is no VPC to design. Access is controlled through roles and network *policies*, not network topology |
| **Billing unit** | Per-service (instance-hours, DPU-hours, GB-months, tokens) | **Credits**. Warehouses burn credits per second while running; serverless features (tasks, dynamic tables, Cortex) burn credits on their own meters |

**The key idea:** a warehouse is *just compute*. It doesn't hold your data. An X-Small
warehouse uses 1 credit/hour, is billed per second with a 60-second minimum, and can
auto-suspend after 60 seconds idle. Each size up doubles the rate. Most of your cost
control comes from keeping warehouses small and setting `AUTO_SUSPEND` short.

---

## 2. Account facts to confirm before designing anything

Check each of these in Snowsight and write the answers into §9. Several design choices
depend on them.

| Check | How | Why it matters |
|---|---|---|
| **Edition** | Snowsight → Admin → Accounts (or the trial welcome email) | **Enterprise or higher is required** for Feature Store, data metric functions (the data-contract equivalent), masking/row-access policies, object lineage, and Time Travel beyond 1 day. If you chose Standard, consider re-signing up as Enterprise *before* you put work in. |
| **Cloud + region** | `SELECT CURRENT_REGION();` | Ideally AWS `us-east-1`/`us-west-2`, so Cortex model availability is good and an AWS↔Snowflake comparison is like-for-like. |
| **Trial end date** | Snowsight → Admin → Billing | **Confirmed: 120 days, ending 2027-02-01** (longer than the standard 30-day trial). Plan for whichever ends first: the date or the $400. When a trial ends, the account is **suspended**: you can log in but can't run anything. Keep everything in code in this repo so you can rebuild in a new account. |
| **AI features enabled?** | `SELECT AI_COMPLETE('claude-haiku-4-5', 'hi');` (us-east-2 reaches Claude only via cross-region inference) | **Cortex AI features are disabled on self-service trials until you add a credit card.** Without one, the Lab 3 equivalent (Bedrock → Cortex) can't run. Decision in §9. |
| **Cross-region inference** | `SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;` | Controls which LLMs you can call. `DISABLED` limits you to models hosted in your region. |

**Things trial accounts cannot do at all** (per Snowflake docs): external network access,
hybrid tables, outbound private connectivity, Openflow, and Snowpark Container Services
*except for notebooks*. That last one matters a lot. See the Lab 5+ gap in §4.

---

## 3. Guardrails first: the equivalent of your AWS budget alarm

Do this **before** anything else, just like `aws-account-setup.md` set up budgets first.
It's the only part of this plan worth running now:

```sql
USE ROLE ACCOUNTADMIN;

-- $400 ≈ ~130 credits at Enterprise list price (~$3/credit, varies by region/edition).
-- Cap warehouse spend well below that and leave room for serverless + storage.
CREATE RESOURCE MONITOR northstar_trial_guard
  WITH CREDIT_QUOTA = 80
       FREQUENCY = NEVER          -- one total cap for the whole trial, not monthly
       START_TIMESTAMP = IMMEDIATELY
  TRIGGERS ON 50 PERCENT DO NOTIFY
           ON 75 PERCENT DO NOTIFY
           ON 90 PERCENT DO SUSPEND            -- let running queries finish
           ON 100 PERCENT DO SUSPEND_IMMEDIATE;

ALTER ACCOUNT SET RESOURCE_MONITOR = northstar_trial_guard;
```

> **Verified 2026-10-04:** attached at `ACCOUNT` level, triggers fire, suspended
> warehouses can't resume. The monitor reacted **~15 min late**, so keep the 90% suspend
> buffer. See [probes/01-resource-monitor-results.md](probes/01-resource-monitor-results.md).

**Lesson: what this does *not* protect.** Resource monitors only limit
**warehouses**. Serverless compute (tasks without a warehouse, dynamic table refresh,
Snowpipe, search optimization), **Cortex AI calls**, and storage don't count toward
it. That's the same lesson as AWS budgets (they alert, they don't stop spending), with a
different blind spot. Check total spend weekly:

```sql
SELECT usage_date, service_type, ROUND(SUM(credits_used), 2) AS credits
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
GROUP BY 1, 2 ORDER BY 1 DESC, 3 DESC;
```

Also set up now:
- **MFA on your login user.** Use passkey/authenticator, not Duo (Duo isn't available on trials).
- **A working user that isn't ACCOUNTADMIN.** ACCOUNTADMIN is like the AWS root user:
  use it for billing and monitors only.
- **Key-pair authentication** for the CLI and Terraform. Snowflake is phasing out
  password-only sign-in, so programmatic access should use keys from the start. Keep the
  private key outside the repo (e.g. `~/.snowflake/keys/`) and add `*.p8` to
  `.gitignore`. Your existing `scripts/check-secrets.sh` should catch it too.

---

## 4. AWS → Snowflake mapping

### Labs 1–2 (built on AWS; concrete)

| Lab item | AWS implementation | Snowflake equivalent | What changes |
|---|---|---|---|
| **Network** | VPC, public subnet, IGW, route table, SG | *Nothing to build.* Optional: a network policy that limits login IPs | Lab 1 A2 and Lab 2 Task 1 (private subnet + NAT ≈ $32+/month) mostly disappear. **The biggest trade-off of the track.** You give up control and gain zero ops work. |
| **Storage zones** | 1 bucket, prefixes `raw/ processed/ features/ artifacts/` | Database `NORTHSTAR_DEV`, one schema per zone: `RAW` (`@LANDING` stage + as-is `TRANSACTIONS_RAW` table), `PROCESSED` (tables), `FEATURES` (Feature Store), `ML` (`@ARTIFACTS` stage + Model Registry) | S3 prefixes do two jobs: storing files and marking zones. Snowflake separates them: **stages hold files, schemas mark zones**. Test: *will anything query it with SQL?* Yes → table. A file something else reads → stage. Keeping `processed`/`features` as files on stages would give up pruning, Time Travel, masking/DMFs, lineage and Feature Store. Optional exercise: **Iceberg tables** if open, multi-engine files are the goal. |
| **Versioning** | S3 versioning | **Time Travel** (1 day Standard, up to 90 Enterprise) + 7-day Fail-safe; `UNDROP` | Built in. Versioning works on tables, not files. |
| **Encryption / public block** | SSE-S3, public access block | Always on; tables can't be made public | Nothing to configure, and nothing to show in a screenshot. |
| **Lifecycle rules** | S3 expiry by prefix | No TTL on tables. Use `DATA_RETENTION_TIME_IN_DAYS` for Time Travel, a scheduled **Task** that deletes rows older than 24 months, `PURGE = TRUE` on `COPY` for staged files | You write retention as code. Good discussion point for the 24-month retention policy. |
| **IAM roles** | MLEngineer / DataEngineer / ModelMonitor | Account roles with the same names, granted to *functional* roles, using schema-level `GRANT`s. Database roles if you want them scoped | Least privilege is written as `GRANT SELECT ON ALL TABLES IN SCHEMA ...` plus **future grants**, instead of IAM JSON. Usually easier to read and check. |
| **SageMaker Studio domain** | Domain + user profile + idle shutdown | **Snowflake Notebooks** (on Container Runtime) or Snowsight Workspaces | No domain to provision or tear down. |
| **IaC** | Terraform `hashicorp/aws`, S3 remote state + DynamoDB lock | Candidates: **DCM Projects** (Snowflake-native declarative, GA 2026-08-07; no state file; plan → deploy) and/or the **`snowflakedb/snowflake` Terraform provider** (stable resources only) | Picking Terraform just because AWS uses it would copy AWS. Terraform is strong for account objects (roles, grants, warehouses, monitors) but risky for tables (some changes drop and recreate). DCM is designed for objects inside a database. Some DCM features are still preview: masking/RAP, tags, GitHub Actions, some grants. Decide through a probe (D6). If Terraform is used, keep its state local, not in the AWS backend. |
| **LocalStack** | Local emulator of AWS | **No Snowflake emulator.** Use (a) Snowpark Python's *local testing mode* for unit tests, (b) a `NORTHSTAR_TEST` database built by **zero-copy clone** of dev | Different testing story; worth an ADR. |
| **Glue crawler + catalog** | Infers schema from CSV | `INFER_SCHEMA` on the stage, or just a declared table; the catalog is built in | No crawler to run or wait for. |
| **`transform.py`** | Glue 4.0 PySpark job | Option A: **SQL**: `TRY_TO_DATE` for both formats, `MEDIAN()`, `QUALIFY ROW_NUMBER() ...`. Option B: **Snowpark Python**, whose DataFrame API is close to PySpark, so the port is mostly a find-replace exercise | Both run on a warehouse; nothing to provision or start up. Compare to Glue cold-start minutes. |
| **Pipeline orchestration** | Glue job triggers | **Dynamic Tables** (declare the result + a target lag; Snowflake handles refresh) or **Tasks** (cron-style DAG) | Declarative vs imperative. Good topic for an ADR. |
| **`feature_engineer.py` + Feature Store** | Glue job + SageMaker Feature Group (offline S3 + online) | **Snowflake Feature Store** (Enterprise): an `Entity` `CUSTOMER`, a `FeatureView` backed by a dynamic table; point-in-time retrieval via `ASOF JOIN` | Your `FEATURE_CUTOFF` / leakage logic carries over unchanged. That logic is the same on any platform. |
| **Data contract** | Markdown doc + `assert` gates in Glue + `verify-lab2.sh` after the fact | No native "contract" object (AWS has none either). Enforce by combining: **table definition** (types, `NOT NULL` enforced at write), **check staging table, then `ALTER TABLE ... SWAP WITH`** gate (Snowflake doesn't enforce `UNIQUE`/`PK`), **DMFs** + expectations for continuous monitoring, `FRESHNESS` + anomaly detection + **Alerts** for the SLA, masking/tags for PII. Consider writing the contract as **ODCS YAML** (machine-readable) | ⚠️ **DMFs: "Trial accounts don't support this feature"** (data quality docs). Transactional writes remove the "don't read during the run" rule entirely. dbt tests aren't a gate by default (they run after the table is rebuilt), so the AWS pre-write gate is stronger than either tool's default. dbt = the cross-platform option, run once as a comparison. |
| **Lineage diagram** | Hand-drawn PNG | Snowsight shows lineage between objects automatically (Enterprise) | Compare the drawn version with the generated one. |
| **Teardown** | `teardown-lab2.sh` for resources Terraform misses | `DROP DATABASE`; warehouses cost nothing while suspended | Teardown is close to trivial. |

### Labs 3–7 (not released yet; provisional, revise each week)

| Course capability | AWS (expected) | Snowflake candidate | Risk on a trial |
|---|---|---|---|
| LLM access (Pre-Lab 3) | Bedrock | **Cortex AI functions** (`AI_COMPLETE` etc.) | Needs a card on file (§2). Trial Cortex usage is also capped per day. |
| Model training (Pre-Lab 4) | SageMaker training jobs | `snowflake-ml-python` in a Notebook on Container Runtime, or ML Jobs | Notebooks are allowed on trials; check that compute pools are too. |
| Model registry | SageMaker Model Registry | **Snowflake Model Registry** | Low |
| Weekly batch scoring (Mon 6 AM ET) | Batch transform + scheduler | `model.run()` from SQL in a **Task** on a CRON schedule | Low. This is Snowflake's strongest area. |
| RAG offers (<2 s) | Bedrock + vector store / KB | **Cortex Search** over a product/offer table + `AI_COMPLETE` | Medium. Measure the latency instead of assuming it. |
| Real-time model endpoint | SageMaker endpoint | Model serving on **Snowpark Container Services** | **High: SPCS isn't available on trials** beyond notebooks. Plan a documented gap. |
| Customer service agent | Bedrock Agents / custom | **Cortex Agents** (+ Snowflake Intelligence) | Medium (needs Cortex) |
| CI/CD | GitHub Actions + Terraform | GitHub Actions + `snow` CLI + Terraform; Snowflake Git integration | Low |
| Monitoring / drift | SageMaker Model Monitor, CloudWatch | **ML Observability** model monitors, DMFs, Alerts | Medium (Enterprise) |

---

## 5. How to compare the two stacks each week

Use the same headings for every lab so the comparisons add up into a final write-up
(and a good portfolio piece). After each lab, add `snowflake/docs/labN-comparison.md`:

0. **Assumptions in play**: which of A1–A4 this lab's result depends on, and whether it would flip under S2 or S3 (§0)
1. **What I built**: objects created, lines of code, time spent
2. **Outcome vs implementation**: the outcome the lab required, and how each platform's *native* approach delivers it
3. **What disappeared**: AWS work that had no Snowflake equivalent (and whether that's good)
4. **What got harder**: places Snowflake fought you or lacked a feature
5. **Cost**: AWS $ from the cost estimate vs Snowflake credits from `METERING_DAILY_HISTORY`, *both scaled to NorthStar's real size* (2.1M customers, $85k/month budget)
6. **Governance**: how the PII / GDPR / CCPA / FCRA requirements were met
7. **Lock-in and portability**: how much of the code would carry over to another platform
8. **Verdict for NorthStar**: which stack you'd recommend to Maya Chen for this layer, and why

Rule: **compare at NorthStar scale, not trial scale.** Your 163k-row sample costs almost
nothing on either platform. The real question is 2.1M customers, 90k events/hour, and
2.3M reviews.

---

## 6. Credit budget (draft)

Assumes Enterprise at about $3/credit, so about 130 credits total. Revisit once §2 is confirmed.

| Phase | Credits | Notes |
|---|---|---|
| Labs 1–2 rebuild | 10 | 163k rows on an X-Small is seconds of compute |
| Lab 3 (LLM) | 25 | Cortex token pricing; cache responses during development |
| Lab 4 (training) | 25 | Notebook / container compute is the wildcard |
| Labs 5–7 | 40 | Serving, agents, monitoring |
| Reserve | 30 | Mistakes happen, like a warehouse left running |

---

## 7. Proposed repo layout (when building starts)

```
snowflake/
  PLAN.md                  this file
  probes/                  feasibility checks (NN-name.sql + NN-name-results.md)
  infrastructure/          Terraform (snowflakedb/snowflake): warehouses, DB, schemas,
                           roles, grants, resource monitor — mirrors ../infrastructure
  sql/                     ordered DDL/DML: stages, tables, dynamic tables, tasks, DMFs
  snowpark/                Python ports of glue-scripts/ (transform, feature_engineer)
  notebooks/               training / exploration (Lab 3+)
  scripts/                 verify-labN.sh equivalents using `snow sql`
  docs/                    ADRs and labN-comparison.md
```

Keeping it in this repo puts the two implementations side by side, but it's also your
graded repo. See the decision in §9.

---

## 8. Tooling to install (when ready)

- **Snowflake CLI** (`snow`): runs SQL, uploads to stages, deploys Snowpark code. Configure a
  named connection with key-pair auth in `~/.snowflake/config.toml` (outside the repo).
- **`snowflake-ml-python`**: Feature Store, Model Registry, modeling APIs. It requires
  specific Python versions, so give it its own venv.
- **Terraform `snowflakedb/snowflake` provider**: same Terraform skills, different
  provider. Run it as a dedicated service user with key-pair auth.
- **VS Code Snowflake extension** (optional): runs SQL from the editor.

---

## 9. Decisions log

| # | Decision | Options | Status |
|---|---|---|---|
| D1 | Edition | Standard / **Enterprise** / Business Critical | **Enterprise**: Feature Store, DMFs, masking, lineage all in scope |
| D2 | Cloud + region | **AWS us-east-2 (Ohio)**: an "extended" Cortex region. Only AI_EMBED/AI_EXTRACT run natively; AI_COMPLETE (Claude) needs cross-region inference. Recommend `CORTEX_ENABLED_CROSS_REGION = 'AWS_US'` (keeps prompts inside the US; relevant to the GDPR/CCPA constraints) | **decided** |
| D3 | Real trial end date | 30 days vs 120 days | **120 days: started 2026-10-04, ends 2027-02-01** (or when the $400 runs out) |
| D4 | Add a credit card to unlock Cortex? | Yes (unlocks Lab 3+ AI; credits still apply) / No (AI labs become paper designs) | _open_ |
| D5 | Where the code lives | `snowflake/` in this repo / separate repo / separate branch | **separate repo**, with the course repo as an upstream remote |
| D6 | IaC tooling | DCM Projects only / Terraform (account level) + DCM (inside the database) / Terraform only. ML objects (Feature Store, Registry) are created in Python either way | _open: probe whether DCM can define the Lab 1 objects (DB, schemas, stages, roles, grants, warehouse, resource monitor). The existing monitor was created by hand and must be adopted by whichever tool wins_ |
| D8 | When to upgrade off the trial | Before the Lab 2 rebuild (DMFs need it) / later, with DMFs as a documented gap until then | _open: probe whether calling a system DMF directly (unscheduled) works on the trial, e.g. `SELECT SNOWFLAKE.CORE.NULL_COUNT(SELECT customer_id FROM t);`_ |
| D7 | Transform style | SQL + Dynamic Tables / Snowpark Python port / both, to compare | _proposed: both for Lab 2_ |

---

## 10. Concepts to learn before building Lab 1–2

In order. Each one maps to a piece of Lab 1–2:

1. **Warehouses and credits**: sizing, auto-suspend, per-second billing
2. **System roles**: `ACCOUNTADMIN` / `SECURITYADMIN` / `USERADMIN` / `SYSADMIN`, and why custom roles should roll up to `SYSADMIN`
3. **Object hierarchy**: account → database → schema → table/view/stage; fully qualified names
4. **Stages and `COPY INTO`**: how files become tables (this replaces S3 + crawler)
5. **Time Travel, Fail-safe, zero-copy clone**: replaces versioning and drives the testing strategy
6. **Dynamic tables vs tasks**: declarative vs scheduled pipelines
7. **Snowpark DataFrames**: compare with your PySpark in `glue-scripts/`
8. **Feature Store entities and feature views**

Sources: Snowflake docs on [trial accounts](https://docs.snowflake.com/en/user-guide/admin-trial-account),
[Feature Store](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview),
[cross-region inference](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cross-region-inference).
Checked 2026-10-04; trial rules change, so re-check before relying on them.
