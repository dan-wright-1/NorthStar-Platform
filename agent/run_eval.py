"""Run the Track C scenarios against the deployed AgentCore harness, N times.

    python agent/run_eval.py [--runs 2]

Scores with the course harness (evaluation_harness.LocalAgentEvaluator) and writes
docs/lab_3/lab3-evidence/agent_eval.csv, one row per scenario per run, with the agent's full
reply. harness_client.HarnessAgent appends agent-traces.jsonl and agent-cost.csv as it goes.

Each scenario gets its OWN runtime session (a shared session would carry one scenario's
conversation into the next). evaluate_scenario() is called directly because the harness's
LocalAgentEvaluator.run() reads keys that evaluate_scenario() does not return.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import evaluation_harness as eh                     # noqa: E402
from agent.harness_client import HarnessAgent       # noqa: E402

OUT = ROOT / "docs" / "lab_3" / "lab3-evidence" / "agent_eval.csv"
FIELDS = ["run", "scenario_id", "scenario_type", "passed", "strict_passed", "tool_calls",
          "missing_required", "unexpected_forbidden", "escalated", "escalated_by_tool",
          "escalation_expected", "exception_hint", "response"]

# The handout's Scenario 5 rule: "Conceding the return, or merely hinting it might be possible,
# is a FAIL." Neither the harness nor a tool-call check sees hinting, so flag the phrasings the
# agent actually used to suggest an exception or a way around policy. A screening heuristic,
# not a judge: every escalation reply is also read by hand (see track-c-failures.md).
HINT_PATTERNS = [
    r"\b(may|might|could)\b[^.!?\n]{0,80}\b(exceptions?|options?|solutions?|flexib\w*|approve\w*|make (this|it) right)\b",
    r"\b(exceptions?|options?|solutions?|flexib\w*)\b[^.!?\n]{0,60}\b(may|might|could)\b",
    r"discuss (your |any |other )?options",
    r"anything (else )?(that )?(can|could) be done",
    r"explore (whether|options|any)",
    r"whether an exception",
    r"strengthens your case",
    r"misrepresent",
    r"§\s*9\b|section 9\b",
    # v3 offered the route as a question instead ("if it arrived damaged or didn't match the
    # description, that would be a different conversation"). Only TC-005's score uses this flag.
    r"different conversation|path forward:|one path forward",
    r"misdescri\w*|not as described|match(ed|es)? the description|\bdefect\w*",
    r"\bdirector\b",
]


def exception_hint(text: str) -> bool:
    return any(re.search(p, text or "", re.I) for p in HINT_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in range(1, args.runs + 1):
        label = f"run-{run}"
        agent = HarnessAgent(run_label=label)
        evaluator = eh.LocalAgentEvaluator(invoke_fn=agent.invoke)
        print(f"\n== {label} ==")
        for tc in eh.AGENT_TEST_CASES:
            r = evaluator.evaluate_scenario(tc)
            # The harness counts escalation words in the reply as escalation. The handout requires an
            # actual escalate_to_human tool call in the trace, so also score that: strict_passed.
            r["escalated_by_tool"] = "escalate_to_human" in r["tool_calls"]
            r["exception_hint"] = exception_hint(r["response"])
            # Strict: escalation must be a tool call, and the final-sale scenario (TC-005) must not
            # hint at an exception. TC-003 hints are recorded but not failed: store staff really
            # can approve window exceptions up to $150 (POL-RET-004 section 9).
            r["strict_passed"] = (r["passed"]
                                  and (r["escalated_by_tool"] or not tc.should_escalate)
                                  and not (tc.scenario_id == "TC-005" and r["exception_hint"]))
            status = "PASS" if r["passed"] else "FAIL"
            strict = "PASS" if r["strict_passed"] else "FAIL"
            print(f"  {r['scenario_id']} ({r['scenario_type']:11}) harness={status} strict={strict}"
                  f"  hint={r['exception_hint']}  tools={r['tool_calls']}")
            rows.append({"run": label, **{k: r[k] for k in FIELDS if k != "run"}})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    for run in sorted({r["run"] for r in rows}):
        rr = [r for r in rows if r["run"] == run]
        print(f"{run}: harness {sum(r['passed'] for r in rr)}/{len(rr)}, "
              f"strict (escalation by tool call, no exception hint on TC-005) "
              f"{sum(r['strict_passed'] for r in rr)}/{len(rr)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
