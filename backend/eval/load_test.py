"""Concurrency harness: fire golden questions at an injected send_fn and report
throughput + tail latency. send_fn is wired to POST /chat in production."""

import time
from concurrent.futures import ThreadPoolExecutor

from eval.golden import GoldenQuestion, load_golden


def run_load(send_fn, questions: list[GoldenQuestion], concurrency: int = 8,
             rounds: int = 1, timer=time.perf_counter) -> dict:
    jobs = [q.question for _ in range(rounds) for q in questions]
    latencies, errors = [], 0

    def one(text: str):
        t0 = timer()
        try:
            send_fn(text)
            return (timer() - t0) * 1000, None
        except Exception as exc:  # noqa: BLE001 — load harness records, never raises
            return (timer() - t0) * 1000, exc

    start = timer()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for ms, err in pool.map(one, jobs):
            latencies.append(ms)
            if err is not None:
                errors += 1
    elapsed = max(timer() - start, 1e-9)
    latencies.sort()
    p = lambda q: latencies[min(len(latencies) - 1, int(len(latencies) * q))] if latencies else 0.0
    return {
        "n": len(jobs),
        "throughput_qps": round(len(jobs) / elapsed, 2),
        "latency_p50_ms": round(p(0.50), 1),
        "latency_p95_ms": round(p(0.95), 1),
        "errors": errors,
    }


def main() -> None:
    import httpx

    def send_fn(text: str) -> None:
        httpx.post("http://localhost:8000/chat", json={"question": text}, timeout=120).raise_for_status()

    res = run_load(send_fn, load_golden(), concurrency=8)
    print(res)


if __name__ == "__main__":
    main()
