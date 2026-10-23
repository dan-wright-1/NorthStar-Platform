# Lab 3 Design Justification

**Track C: customer service agent on Amazon Bedrock AgentCore** (Harness + Gateway + Cedar policy engine, deployed with Terraform).

## Churn model: why gradient boosting

NorthStar has about 10,000 customers, 11 tabular features (recency, frequency windows, spend) and a 22% churn rate; XGBoost trains on 5,949 customers, about 1,300 of them churners. That is too little data for a neural net to earn its keep, and tabular aggregates are where trees do well with little tuning. I measured this on the same split:

| Model | Test AUC | Precision @ top 10% |
|---|---|---|
| XGBoost (depth 4) | 0.7831 | 0.740 |
| Logistic regression | 0.7599 | 0.687 |
| MLP (32, 16) | 0.7646 | 0.700 |

XGBoost leads logistic regression by +0.023 AUC (paired bootstrap 95% CI [0.010, 0.036]) and the MLP by +0.018 ([0.007, 0.030]). The edge is small, only XGBoost was tuned, and this is one split, so the honest claim is "modestly better", resting on that edge and, plausibly, on interactions (recency mattering differently by lifetime value) that a linear model needs hand-built terms for.

On interpretability, logistic regression is less clear than usual: the frequency features are collinear, so 30-day frequency gets a negative coefficient (-0.33) and 90-day a positive one (+0.52). Gain importance tells the retention team which behaviours rank highest, though not why. I would switch to logistic regression if per-customer reason codes had to satisfy a regulator or the gap vanished on a time-based holdout, and to a neural net only with far more rows or raw sequences (clickstream, order history).

## What the features bought

The recency-only baseline scores AUC 0.7242 and the full model 0.7831: a lift of +0.059 (95% CI [0.040, 0.080]). Real but modest; recency carries most of the signal, plausibly because churn means no purchase in the 90 days after the snapshot. At the top-10% cutoff, precision is 0.74 and recall 0.336 (222 of 300 flagged customers churn, out of 660 churners).

By gain, `days_since_last_purchase` (66.7) and `purchase_frequency_180d` (59.6) carry the signal; `customer_tenure_days` and `avg_basket_size_6m` (both 9.2) are close to decorative, though I did not ablate them. `churn_risk_score` bought nothing consistent (AUC 0.7858 with it, 0.7831 without, but 215 instead of 222 churners in the top 10%), so it is excluded. Silver is the weakest tier (AUC 0.718); Platinum looks strongest (0.873) but has only 21 positives, too few to conclude anything.

## LLM system: why an agent

The alternative is a fixed pipeline: retrieve policy, then answer. But inquiries branch: order status needs `lookup_order`, a return needs the order, the policy and possibly `initiate_return`, and an out-of-policy request needs `escalate_to_human`. A pipeline runs every step every time or needs hand-written routing per case. An agent chooses the tools, and the AgentCore Harness runs that loop.

The primary production risk is the price of that choice: the model, not code, decides what happens and what it says. At first TC-005 never called `escalate_to_human` and hinted a human "may have options", yet the course harness passed it because the reply said "escalate". A decision rule made escalation a tool call (6/6); removing the hints took two more prompt versions, one of which backfired. The final prompt dictates the exact sequence and held in 3 of 4 TC-005 samples, a development-set result.

**Why the $50 cap lives in three places.** Each fails differently. The prompt makes the model escalate instead of trying, but customers can talk a model out of advice. `tools.py` refuses over-cap credits even if the Lambda is called directly, but code changes, and its hardcoded 50 can drift from Terraform's `credit_cap_dollars`. The gateway's Cedar `forbid` blocks the call before the Lambda runs. Alone, I would trust the gateway policy: it sits outside both the model and the tool code, applies to every call the agent can make (the gateway is its only path to the tools; direct Lambda invocation is gated by IAM, not Cedar), is declarative, and `policy_tests.py` proved it with no model in the loop ($51 and $75 denied). None of them reliably stops two $40 credits: the tool check and Cedar rule are per-call, and the prompt is untested advice. That needs session state, either a temporal policy (the InvokeHarness API lists no policy-session parameter, so as far as the documentation shows a harness cannot use one today; untested) or a running total in the tool's datastore; see `docs/lab_3/lab3-evidence/track-c-failures.md`.

## With 10x the time or data

For churn, I would replace the random split with a rolling-origin backtest (train on earlier snapshots, test on later ones): the model scores weekly, and one random split is optimistic about how it ages. Ten times the data would give Platinum about 200 positives, and would outgrow a loader that pages Athena results 1,000 rows at a time. For the agent, I would build a held-out set of a few hundred conversations, including paraphrased pressure and customers who lie about their tier, sample each several times, report a pass rate with an interval, and close the split-credit hole before real customers reach `apply_loyalty_credit`.
