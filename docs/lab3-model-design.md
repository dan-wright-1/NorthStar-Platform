# Lab 3 Design Justification

**Track C: customer service ReAct agent** (client-side loop over Bedrock Converse).

## Churn model: why gradient boosting

NorthStar has about 10,000 customers, 11 tabular features (recency, frequency windows, spend), and a 22% churn rate. XGBoost trains on 5,949 customers (about 1,300 churners) and is scored on 3,000. That is too little data for a neural net to earn its keep, and tabular aggregates are where trees are strong with little tuning. I tested this on the same split instead of assuming it:

| Model | Test AUC | Precision @ top 10% |
|---|---|---|
| XGBoost (depth 4) | 0.7831 | 0.740 |
| Logistic regression | 0.7599 | 0.687 |
| MLP (32, 16) | 0.7646 | 0.700 |

XGBoost leads logistic regression by +0.023 AUC (paired bootstrap 95% CI [0.010, 0.036]) and the MLP by +0.018 ([0.007, 0.030]). The intervals exclude zero, but the edge is small, and the precision gaps are 12 to 16 customers out of 300, which I would not call significant. Two caveats favor XGBoost: I picked its depth from four runs, while the other two models were not tuned, and this is one split of one snapshot. So the defensible claim is "no worse than the simpler models, and modestly better," not "much better." The feature table has no missing cells, so boosting's native handling of missing values is a convenience here, not evidence; the case for it rests on the measured edge and on interactions, such as recency mattering differently by tier.

Interpretability favors logistic regression less than usual here. Its coefficients are unreliable because the features are collinear: 30-day frequency is negative (-0.33) while 90-day is positive (+0.52). I did not try dropping a collinear feature or adding regularization, which would likely stabilize the signs, and a linear model is still more transparent about mechanism. Gain importance from the boosted model shows the retention team which behaviors rank highest, but it is a ranking, not a causal explanation, and I did not compute SHAP values.

I would switch to logistic regression if per-customer reason codes had to be defensible to a regulator, or if the AUC gap vanished on a time-based holdout. I would move to a neural net only with roughly 100x more rows or raw sequences (clickstream, order history) that trees cannot consume. The 100x figure is a rule of thumb, not something I measured.

## What the features bought

A recency-only baseline scores AUC 0.7242; the full model scores 0.7831, a lift of +0.059 (95% CI [0.040, 0.080]). That is real but modest: recency alone captures most of the signal, plausibly because churn means no purchase in the 90 days after the snapshot, so recent inactivity predicts continued inactivity almost by construction. At the top-10% cutoff the model has precision 0.74 and recall 0.336 (222 of 300 flagged customers churn; 660 churn in total).

By gain, `days_since_last_purchase` (51.4) and `purchase_frequency_180d` (45.2) carry the signal; `customer_tenure_days` (8.5) and `avg_basket_size_6m` (8.8) rank last. I did not ablate features, so that means low gain, not proven useless. The precomputed `churn_risk_score` added nothing (AUC 0.7784 with it, 0.7787 without, at depth 6), so I left it out. Silver is the weakest tier (AUC 0.718). Platinum looks strongest (0.873) but has only 21 positives, too few to trust.

## LLM system: why an agent

The alternative is a fixed pipeline: retrieve policy, then answer. That suits pure policy questions, but inquiries branch. Order status needs an order lookup; a return needs the order record, the policy, and possibly an action; an out-of-policy request needs a human handoff. A fixed pipeline either runs every step every time or needs hand-written routing per case. The agent is partly a pipeline anyway: the loop forces policy retrieval on the first step of every turn, because prompt fixes only partly worked. In the first run the model skipped retrieval in all three policy scenarios (TC-002, 003, 005) and answered from memory or stalled. Splitting read tools from action tools got TC-003 and TC-005 retrieving, but TC-002 still skipped it and escalation stayed conditional, and a further rewording changed nothing.

The main production risk is acting on claims it has not verified. In the Platinum scenario the customer asserts a tier and a final-sale item with no order number, and the agent accepts both. Code, not the prompt, limits the damage: `initiate_return` refuses final-sale and out-of-window orders whatever the model says, and credits are capped. The agent passed all six scenarios in the last two runs, but I tuned the prompt and loop while looking at those same scenarios, so 6/6 is a development-set result, and it is one sample per scenario at temperature 0, not a reliability estimate.

## With 10x the time or data

For churn, I would replace the random split with a rolling-origin backtest (train on earlier snapshots, test on later ones), since the model scores weekly and a single random split is optimistic about how it will age. Ten times the data would also lift Platinum from 21 positives to about 200, enough to evaluate that tier. For the agent, I would build a labeled set of a few hundred conversations with paraphrased pressure tactics, sample each several times at nonzero temperature, and report a pass rate with an interval. Six scenarios cannot show whether a 95% pass rate is real.
