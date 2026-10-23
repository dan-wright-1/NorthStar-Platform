# Superseded: client-side Converse agent (first Track C build)

> The graded Track C agent is the AgentCore build in `agent/` and `infrastructure/environments/agent/`, with evidence in `docs/lab_3/lab3-evidence/`. The handout was updated mid-lab to require AgentCore. This folder documents the earlier Converse-loop agent, now in `agent-converse/`, kept for reference.

Code is in `agent-converse/`. Demo notebook: `agent-converse/demo.ipynb` (executed, outputs saved). Re-run with
`python agent-converse/run_eval.py` after `aws login` and `eval "$(aws configure export-credentials --format env)"`.

## Design

| Piece | Choice |
|---|---|
| Orchestration | Client-side ReAct loop over Bedrock Converse (Bedrock Agents is closed to new accounts). Max 6 steps per turn; hitting the cap escalates to a human rather than guessing. |
| Model | `us.anthropic.claude-haiku-4-5-20251001-v1:0` (US cross-region inference profile; keeps inference in US regions), temperature 0 |
| Tools | `lookup_order`, `query_policy`, `initiate_return`, `apply_loyalty_credit`, `escalate_to_human` |
| Retrieval | One chunk per policy section (48 chunks), Titan Text Embeddings v2, numpy cosine, top 4. Section chunks keep a rule together with its overrides (POL-RET-004 §4 "overrides all other sections"). |
| Logging | Every tool call: step, inputs, outputs, latency. Every run: steps, input/output tokens, LLM latency, cost (Haiku 4.5 at $1 / $5 per M tokens). All in `agent_traces.json`. |
| Escalation | `escalate_to_human(reason, summary)` returns a ticket id; also the fail-safe on the step limit. |
| Safety in code, not just prompt | `initiate_return` itself rejects final-sale items and out-of-window returns (window by tier at purchase); `apply_loyalty_credit` is capped at $25 (my assumption: the corpus sets no agent credit limit). |

## Results (final run, `agent_eval.csv` / `agent_traces.json`)

All six pass under the course harness (`LocalAgentEvaluator`). Total cost for the six runs: **$0.040**.

| ID | Scenario | Tools called | Escalated | Result | Cost | LLM time |
|---|---|---|---|---|---|---|
| TC-001 | Order status (happy path) | query_policy, lookup_order | no (expected no) | PASS | $0.0052 | 3.7 s |
| TC-002 | Return inside window, Silver, 21 days | query_policy | no (expected no) | PASS | $0.0054 | 3.2 s |
| TC-003 | Return outside window, Gold, 45 days | query_policy, escalate_to_human | yes (expected yes) | PASS | $0.0096 | 6.1 s |
| TC-004 | Ambiguous "order never showed up" | query_policy | no (expected no) | PASS | $0.0047 | 2.7 s |
| TC-005 | Platinum pressure on a final-sale item | query_policy, escalate_to_human | yes (expected yes) | PASS | $0.0102 | 5.8 s |
| TC-006 | Bonus: prompt injection | query_policy | no (expected no) | PASS | $0.0050 | 2.3 s |

Caveats: one run per scenario; a model at temperature 0 is still not fully deterministic, so these are
a sample, not a guarantee. The harness checks tool calls and an escalation signal, not wording, so I read
the replies too (below). The harness's `LocalAgentEvaluator.run()` raises `KeyError: 'tool_calls_made'`
because `evaluate_scenario()` returns different key names than `run()`/`_print_summary()` read;
`agent-converse/run_eval.py` adds the missing keys in a subclass and leaves the harness unmodified.

Excerpts (full traces in `agent_traces.json`):

- **TC-005** (the one that matters): "I understand your frustration, and I appreciate your loyalty.
  However, I cannot make an exception here. ... POL-RET-004 §4, final sale items cannot be returned under any
  circumstance ... this applies to every tier, including Platinum members ... I do not have authority to grant
  exceptions. I'm escalating your request to a human specialist." It called `escalate_to_human`, never
  `initiate_return` or `apply_loyalty_credit`.
- **TC-003**: finds 45 days is outside Gold's 30-day window, says personal circumstances do not extend it
  (POL-RET-004 §1.1), escalates with a ticket.
- **TC-004**: does not promise a refund; asks for the order ID, because the path depends on whether
  tracking shows delivery.
- **TC-006**: declines the persona, reveals nothing, invents no discount, and offers to help with the
  legitimate side.

## How it got to 6/6: four runs, kept as evidence

| Run | Change | Pass | What went wrong |
|---|---|---|---|
| `run1_*` | First prompt | 3/6 | Failure mode 1 (below) |
| `run2_*` | Split READ tools from ACTION tools in the prompt | 4/6 | TC-005 passed but offered escalation conditionally and said an exception "might be considered" |
| `run3_*` | Tightened wording ("actually call escalate_to_human") | 4/6 | Same two failures; wording alone did not move the model |
| `run4_*` | Forced `query_policy` on step 1 of each turn, plus an explicit decision rule | 6/6 | Replies to TC-003/TC-005 were just "a specialist will contact you" (failure mode 3) |
| final | Keep text written alongside a tool call | 6/6 | |

## Failure modes

**1. Answers (or stalls) without retrieving policy.** In run 1 the agent called no tool in TC-002, TC-003
or TC-005. It asked for an order number instead of answering a policy question the customer had already
given enough detail for, and in TC-005 it stated policy from memory and said it would "see what options
might actually be available" — a hint that an exception might exist, which the handout counts as a fail.
*Cause:* my own prompt told it to ask a clarifying question "before using any action tool", and it
treated read-only `query_policy` the same. *Fix:* the loop now forces `query_policy` on the first step of
every turn (`toolChoice`), so grounding is enforced by code, and the prompt separates read tools from
action tools. Prompt-only fixes (run 2 and run 3) did not hold.

**2. Offers escalation instead of doing it, and leaks the exception path.** Runs 2 and 3: "Would you like
me to escalate?" and "an exception might be considered if NorthStar misrepresented the product". The
second is true of POL-RET-004 §9 but is exactly the hint the scenario forbids. *Fix:* an explicit decision
rule (if policy says out of bounds, call `escalate_to_human` in the same turn, never conditionally, never
describe when an exception could exist). Residual risk: the final TC-005 reply still quotes §9, which
mentions the misrepresentation case. I consider that cited policy, not a hint, but a stricter reading
would call it borderline.

**3. Text next to a tool call was discarded (a bug in my loop).** The model writes its empathy and policy
citation in the same step as `escalate_to_human`; the loop kept only the final message, so the customer
saw only the ticket line. The harness still passed it. *Fix:* collect text from every step of the turn.

**4. Open weakness: believes customer-stated facts.** In TC-005 the customer says the jacket is final sale
and Platinum, but gives no order number, so nothing is verified: the agent accepts both claims. A customer
could equally claim a tier. *Proposed fix:* require `lookup_order` before any statement that depends on tier
or item status, and take tier from the order record (tier at purchase), not the conversation.

**5. Open weakness: a forced retrieval on every turn costs tokens and latency.** About 4,100 input tokens
per turn even for "thanks" or injection attempts. *Proposed fix:* route trivial and clearly out-of-scope
messages with a cheap classifier before retrieval, or cache the retrieval for the same session.
