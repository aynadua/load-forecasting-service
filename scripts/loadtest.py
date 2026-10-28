"""Hammer a running service and report latency percentiles.

Usage:  python scripts/loadtest.py [--url URL] [--requests N] [--concurrency C]

Deliberately dependency-light: it builds one realistic request body and
replays it, which is enough to characterise serving latency. It is not a
substitute for a real load generator under production traffic shapes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from urllib import error, request

# The service needs max(lag) + max(rolling window) = 504 hours before it can
# build a complete feature row; anything shorter is correctly rejected with 422.
HOURS_OF_HISTORY = 600


def build_body(steps: int) -> bytes:
    """A synthetic but well-formed history: daily sine plus a weekly ripple."""
    import math

    start = datetime(2018, 6, 1)
    history = []
    for i in range(HOURS_OF_HISTORY):
        stamp = start + timedelta(hours=i)
        daily = 8000.0 * math.sin(2 * math.pi * (stamp.hour - 4) / 24)
        weekly = 2500.0 * math.sin(2 * math.pi * stamp.weekday() / 7)
        history.append(
            {"timestamp": stamp.isoformat(sep=" "), "load_mw": 32000.0 + daily + weekly}
        )
    return json.dumps({"history": history, "steps": steps}).encode()


def one_call(url: str, body: bytes) -> tuple[float, int]:
    req = request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=30) as response:
            response.read()
            status = response.status
    except error.HTTPError as exc:
        status = exc.code
    except OSError:
        status = 0
    return (time.perf_counter() - started) * 1e3, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/forecast")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()

    body = build_body(args.steps)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda _: one_call(args.url, body), range(args.requests)))
    elapsed = time.perf_counter() - started

    latencies = sorted(latency for latency, status in results if status == 200)
    failures = sum(1 for _, status in results if status != 200)

    if not latencies:
        print(f"all {args.requests} requests failed; is the service running at {args.url}?")
        return 1

    def percentile(p: float) -> float:
        return latencies[min(int(len(latencies) * p / 100), len(latencies) - 1)]

    print(f"requests      {args.requests} at concurrency {args.concurrency}")
    print(f"failures      {failures}")
    print(f"throughput    {args.requests / elapsed:.1f} req/s")
    print(f"latency p50   {statistics.median(latencies):.1f} ms")
    print(f"latency p95   {percentile(95):.1f} ms")
    print(f"latency p99   {percentile(99):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
