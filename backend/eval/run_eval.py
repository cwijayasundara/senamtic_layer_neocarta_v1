"""Run the golden set through the real agent and emit a routing/answer/latency
scorecard. Routing and answering are injected so the aggregation is unit-testable
without an LLM or a live graph."""

import argparse
import json
import time
from pathlib import Path

from eval.golden import GoldenQuestion, load_golden
from eval.scorer import check_answer, routing_scores


def evaluate(route_fn, answer_fn, questions: list[GoldenQuestion], timer=time.perf_counter) -> dict:
    """route_fn(q: GoldenQuestion) -> list[str] table ids;
    answer_fn(q: GoldenQuestion) -> str answer. Both are injected so the runner is
    testable without an LLM or live graph (main() wires the real agent)."""
    records, latencies = [], []
    hits = answers_ok = 0
    prec_sum = rec_sum = 0.0
    for q in questions:
        t0 = timer()
        routed = route_fn(q)
        t1 = timer()
        answer = answer_fn(q)
        t2 = timer()
        r = routing_scores(routed, q.expected_tables)
        ok = check_answer(answer, q.answer_check)
        hits += int(r["hit"])
        answers_ok += int(ok)
        prec_sum += r["precision"]
        rec_sum += r["recall"]
        lat = {"route": round((t1 - t0) * 1000, 1),
               "answer": round((t2 - t1) * 1000, 1),
               "total": round((t2 - t0) * 1000, 1)}
        latencies.append(lat["total"])
        records.append({"id": q.id, "category": q.category, "routing": r,
                        "routed": routed, "answer_ok": ok, "latency_ms": lat})
    n = len(questions) or 1
    latencies.sort()
    summary = {
        "n": len(questions),
        "routing_hit_rate": round(hits / n, 3),
        "routing_precision": round(prec_sum / n, 3),
        "routing_recall": round(rec_sum / n, 3),
        "answer_accuracy": round(answers_ok / n, 3),
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0,
    }
    return {"questions": records, "summary": summary}


def _print_summary(card: dict) -> None:
    s = card["summary"]
    print("\n=== Scale Agent Scorecard ===")
    print(f"questions:          {s['n']}")
    print(f"routing hit-rate:   {s['routing_hit_rate']:.1%}")
    print(f"routing precision:  {s['routing_precision']:.3f}")
    print(f"routing recall:     {s['routing_recall']:.3f}")
    print(f"answer accuracy:    {s['answer_accuracy']:.1%}")
    print(f"latency p50/p95 ms: {s['latency_p50_ms']} / {s['latency_p95_ms']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Score the agent over the golden set.")
    ap.add_argument("--out", default="scorecard.json")
    ap.add_argument("--routing-only", action="store_true",
                    help="skip full answer generation (cheaper; routing metrics only)")
    args = ap.parse_args(argv)

    from semantic_layer.agent.routing import route_tables
    from semantic_layer.config import settings

    def route_fn(q: GoldenQuestion):
        return route_tables(q.question,
                            k_ret=settings.schema_routing_k_ret,
                            k_rank=settings.schema_routing_k_rank)

    if args.routing_only:
        def answer_fn(q: GoldenQuestion):
            return ""
    else:
        from semantic_layer.agent.build import ask

        def answer_fn(q: GoldenQuestion):
            return ask(q.question)

    questions = load_golden()
    card = evaluate(route_fn, answer_fn, questions)
    card["summary"]["embedding_mode"] = "fake" if settings.fake_embeddings else "openai"
    card["summary"]["routing_only"] = args.routing_only
    Path(args.out).write_text(json.dumps(card, indent=2))
    _print_summary(card)


if __name__ == "__main__":
    main()
