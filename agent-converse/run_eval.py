"""Run the five required Track C scenarios (plus bonus TC-006) and save traces.

    python agent-converse/run_eval.py [--output-dir docs/lab_3/agent]

Uses evaluation_harness.LocalAgentEvaluator for scoring; also writes the full
per-run trace (tool inputs/outputs/latency, tokens, cost) to JSON.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation_harness import AGENT_TEST_CASES, LocalAgentEvaluator   # noqa: E402
from agent import CustomerServiceAgent                                  # noqa: E402


class TracedLocalEvaluator(LocalAgentEvaluator):
    """LocalAgentEvaluator.evaluate_scenario returns tool_calls / missing_required /
    unexpected_forbidden / escalated, but the inherited run() and _print_summary() read
    tool_calls_made / missing_required_tools / unexpected_forbidden_tools / escalation_correct,
    so run() raises KeyError on the first scenario. Add the keys they expect; scoring is
    unchanged. (Harness left unmodified.)"""

    def evaluate_scenario(self, tc):
        r = super().evaluate_scenario(tc)
        r["tool_calls_made"] = r["tool_calls"]
        r["missing_required_tools"] = r["missing_required"]
        r["unexpected_forbidden_tools"] = r["unexpected_forbidden"]
        r["escalation_correct"] = r["escalated"] == r["escalation_expected"]
        r["response_preview"] = r["response"][:200]
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(ROOT / "docs" / "lab_3" / "agent"))
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    agent = CustomerServiceAgent(region=args.region)
    results = TracedLocalEvaluator(invoke_fn=agent.invoke_fn).run(AGENT_TEST_CASES)
    results.to_csv(out / "agent_eval.csv", index=False)

    traces = [r.to_dict() for r in agent.runs]
    (out / "agent_traces.json").write_text(json.dumps(traces, indent=2, default=str), encoding="utf-8")
    total = sum(r.cost_usd for r in agent.runs)
    print(f"\nTotal cost for {len(agent.runs)} runs: ${total:.4f}")
    print(f"Saved {out / 'agent_eval.csv'} and {out / 'agent_traces.json'}")


if __name__ == "__main__":
    main()
