"""Run the golden-question evalset through the agent and score with the LLM judge."""

import json

from semantic_layer.agent.controller import answer_stream
from semantic_layer.eval.evalset import load_evalset
from semantic_layer.eval.judge import judge_answer


def default_answer_fn(question: str) -> str:
    """Drive the deterministic controller path and return the final answer text."""
    final = ""
    for event in answer_stream(question):
        if event.get("type") == "answer":
            final = event.get("content", "")
    return final


def run_eval(evalset: list[dict], answer_fn=default_answer_fn, judge_fn=judge_answer) -> dict:
    """Answer each question and judge it; return per-item results plus aggregates."""
    results = []
    for item in evalset:
        answer = answer_fn(item["question"])
        verdict = judge_fn(item["question"], answer, item["expect"])
        results.append({"id": item["id"], "score": verdict["score"], "reason": verdict["reason"]})
    n = len(results) or 1
    mean = round(sum(r["score"] for r in results) / n, 2)
    pass_rate = round(sum(1 for r in results if r["score"] >= 3) / n, 2)
    return {"results": results, "mean_score": mean, "pass_rate": pass_rate}


def main() -> None:
    report = run_eval(load_evalset())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
