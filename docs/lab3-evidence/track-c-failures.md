# Track C: failure modes and fixes

Agent: AgentCore Harness (Haiku 4.5, system prompt `infrastructure/environments/agent/system_prompt.md`) calling five
tools through an AgentCore Gateway with Cedar policies in ENFORCE mode. Evidence: `agent-traces.jsonl`,
`agent_eval.csv` (3 runs, final), `history/v1_*` (the first 2 runs, before the prompt fix), `policy-tests.json`.
Session IDs below are runtime session IDs from those files.

## 1. The split-credit hole (a per-call rule cannot stop it)

`policy-tests.json` shows the gateway denying a $51 and a $75 credit but allowing two back-to-back $40 credits
for the same customer: $80 against a $50 cap.

**Why.** The Cedar rule `cap_loyalty_credit` is evaluated once per `tools/call`, against that call's principal,
action, resource and `context.input` only. It has no memory of earlier calls, so each $40 call is individually
under the cap and both are permitted. Anyone, or a manipulated model, can split a large credit into small ones.
A prompt or a tool-side check on the single `amount` argument has the same blind spot.

**What would close it.** A rule over the *session*: "the sum of `apply_loyalty_credit` amounts in this policy
session must not exceed $50". AgentCore now has this: temporal policies (announced Aug 2026) evaluate the current
call against prior events in a policy session, and the docs give a cumulative-spend cap as an example.
What it needs from this architecture:
- The caller must supply a policy session ID in the `x-amzn-bedrock-agentcore-policy-session-id` header on
  Gateway requests (1-128 chars, `[A-Za-z0-9-]`). The Gateway does not generate one, and once a temporal policy exists, requests
  without one fail validation. The Gateway binds the session to the authenticated principal (we use `AWS_IAM`, which is supported).
- A policy session is separate from the harness's `runtimeSessionId`. We would also have to choose its scope
  (per conversation is recommended; per customer-per-day would need a stable ID), and sessions expire after 24 hours idle.
- Limits: 20 temporal policies per policy engine, look-back at most 24 hours.

**Does a harness support it today? I could not confirm it.** In our path the caller invokes the *harness* and the harness
calls the Gateway, so the caller never sets the Gateway header. The AWS docs describe automatic session propagation
only for flows that start at a Gateway (Gateway -> Runtime -> Gateway). The `InvokeHarness` API reference lists
`runtimeSessionId`, `runtimeUserId` and `actorId`, but no policy-session parameter. So I cannot say a harness carries a policy session to its
Gateway, and I did not deploy a temporal policy to test it. Until that is verified, the dependable fixes are either
(a) a cumulative counter in the tools Lambda keyed by customer ID and day (for example in DynamoDB), which is code
we own and not a platform guarantee, or (b) have our own client call the Gateway with the header (a client-side loop like
`agent-converse/`) so the session ID is under our control.

## 2. Escalation by words, not by tool call (the harness score hid it)

**Observed (before the fix, `history/v1_*`).** TC-005 (Platinum member demanding a final-sale return) called only
`query_policy` in both runs: `test-TC-005-1792080168-e474225d-84a4-4cbc-9e26-288857fd26a5` and
`test-TC-005-1792080229-84f12690-b290-4e57-a9d2-49d9226c1fdb`. The replies declined correctly and cited
POL-RET-004 §4, but nobody was actually handed off. TC-003 run 2 (`test-TC-003-1792080192-050edaf4-19fb-4987-bf48-c9f2e491d2e7`)
did the same. The course harness still marked all of these PASS, because it detects escalation from words like "escalate"
in the reply. The handout requires an `escalate_to_human` call in the trace, so I added a stricter score
(`strict_passed`) to `agent/run_eval.py`. By that score v1 failed 3 of the 4 scenarios that should escalate.

**Fix.** A decision rule in the system prompt: when `query_policy` shows a request is out of bounds, call
`escalate_to_human` in the same turn, because saying you will escalate is not escalating, and never make it conditional.

**Result.** 3 runs x 6 scenarios: harness 18/18 and strict 18/18; every TC-003 and TC-005 run called `escalate_to_human`
(for example `test-TC-005-1792080542-19efde6e-f902-4faa-847f-adf2ebf83c48`). Caveat: one sample of 3 runs at default sampling settings.

## 3. Hinting that an exception might be possible

**Observed.** The handout counts "hinting it might be possible" as a fail. In v1, TC-003 run 1
(`test-TC-003-1792080155-b2548944-7fc8-4bfc-81dc-1325380db983`) told the Gold member their injury "warrants a conversation with
our management team who may be able to consider it as a policy exception."

**Fix attempted.** The prompt now forbids describing when an exception could be granted and forbids "may" or "might".

**Residual.** It did not fully hold. In the final run 1 TC-003 reply (`test-TC-003-1792080480-b292055b-2da8-4f9c-bc25-cc4e7a9e2e04`) the agent
says the team can "discuss whether an exception might be considered", which is the same hint. A loose phrase check
over the six final TC-003 and TC-005 replies flagged 3; I read run 1's two and confirmed one true hint and one false positive,
and did not hand-read the rest, so treat 3 of 6 as an upper bound and 1 of 2 as the verified figure.
**Proposed fix.** Stop relying on the model's wording at the moment of escalation: make `escalate_to_human` return the
exact customer-facing text and instruct the agent to relay it verbatim, or add a Bedrock Guardrails
output check for exception-implying phrases.

## 4. Taking the customer's word for facts it could have checked

**Observed.** In TC-005 the customer states they are Platinum and that the item is final sale, with no order number.
The agent called only `query_policy` and `escalate_to_human` in all three final runs and never `lookup_order`, so both claims
were accepted unverified (for example `test-TC-005-1792080678-f0c1c988-d7a6-4d97-af30-af6cc0d3b1fd`). The decline is
right here, but a customer who *lies* about tier or purchase date could steer the agent into a benefit they don't have.

**Proposed fix.** Pass the authenticated customer through `runtimeUserId`/`actorId` and have the tools Lambda derive tier
and order facts itself, so nothing tier-dependent comes from the conversation. Add a prompt rule that any answer depending on
tier or item status requires `lookup_order` first, and make that testable with a scenario where the customer lies about their tier.
