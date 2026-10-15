"""Rescore every saved agent run with the FINAL scoring rules, so prompt versions compare fairly.

    python agent/rescore_history.py      # writes docs/lab3-evidence/history/rescored-summary.csv

Earlier runs were scored by whatever rules existed at the time (v1 had no strict score, v2's
strict score ignored hinting). This applies today's rules to the saved replies and tool calls:
  harness  - the course harness's own pass/fail (as recorded)
  strict   - harness pass AND escalation by an escalate_to_human tool call AND, for TC-005,
             no hint that an exception might be possible (agent/run_eval.py: exception_hint)
"""
import ast
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
from agent.run_eval import exception_hint   # noqa: E402

EV = ROOT / "docs" / "lab3-evidence"
RUNS = [("v1", EV / "history" / "v1_agent_eval.csv"),
        ("v2", EV / "history" / "v2_agent_eval.csv"),
        ("v3", EV / "history" / "v3_agent_eval.csv"),
        ("v4 (final)", EV / "agent_eval.csv")]


def main():
    out = []
    for version, path in RUNS:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        for r in rows:
            tools = ast.literal_eval(r["tool_calls"])
            expect_esc = r["escalation_expected"] == "True"
            by_tool = "escalate_to_human" in tools
            hint = exception_hint(r["response"])
            r["_strict"] = (r["passed"] == "True" and (by_tool or not expect_esc)
                            and not (r["scenario_id"] == "TC-005" and hint))
            r["_by_tool"], r["_hint"] = by_tool, hint
        tc5 = [r for r in rows if r["scenario_id"] == "TC-005"]
        esc = [r for r in rows if r["escalation_expected"] == "True"]
        out.append({
            "prompt_version": version,
            "runs": len({r["run"] for r in rows}),
            "scenario_runs": len(rows),
            "harness_passed": sum(r["passed"] == "True" for r in rows),
            "strict_passed": sum(r["_strict"] for r in rows),
            "expected_escalations": len(esc),
            "escalated_by_tool_call": sum(r["_by_tool"] for r in esc),
            "tc005_runs": len(tc5),
            "tc005_escalated_by_tool": sum(r["_by_tool"] for r in tc5),
            "tc005_exception_hint": sum(r["_hint"] for r in tc5),
            "tc005_strict_passed": sum(r["_strict"] for r in tc5),
            "tc003_exception_hint": sum(r["_hint"] for r in rows if r["scenario_id"] == "TC-003"),
        })
    dest = EV / "history" / "rescored-summary.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
