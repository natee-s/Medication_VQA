#!/usr/bin/env python3
"""
Lightweight production load test for Medication_VQA.

Default profile is intentionally read-only:
- GET /
- GET /liff/camera
- GET /liff/config
- GET /test-db/{drug}

It avoids /webhook, /liff/upload-label, and /cron/check-reminder by default
because those endpoints can call LINE, Gemini, YOLO, or reminder side effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://ginya.v89tech.com"
DEFAULT_DRUG = "AMITRIPTYLINE"
USER_AGENT = "Medication_VQA production load tester/1.0"


@dataclass(frozen=True)
class Target:
    name: str
    method: str
    url: str
    expected_status: int = 200


@dataclass
class Result:
    target: str
    status_code: int
    ok: bool
    latency_ms: float
    error: str = ""


def build_url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def build_profile_targets(profile: str, base_url: str, drug: str, pdpa_base_url: str) -> list[Target]:
    targets: list[Target] = []
    if profile in {"health", "production-read", "all-read"}:
        targets.append(Target("main_root", "GET", build_url(base_url, "/")))
    if profile in {"liff", "production-read", "all-read"}:
        targets.extend(
            [
                Target("liff_camera", "GET", build_url(base_url, "/liff/camera")),
                Target("liff_config", "GET", build_url(base_url, "/liff/config")),
            ]
        )
    if profile in {"db", "production-read", "all-read"}:
        targets.append(Target("test_db", "GET", build_url(base_url, f"/test-db/{drug}")))
    if profile in {"pdpa-health", "all-read"}:
        targets.append(Target("pdpa_health", "GET", build_url(pdpa_base_url, "/health")))
    return targets


def parse_custom_targets(values: list[str]) -> list[Target]:
    targets = []
    for index, value in enumerate(values, start=1):
        if "=" in value:
            name, url = value.split("=", 1)
            name = name.strip() or f"custom_{index}"
        else:
            name = f"custom_{index}"
            url = value
        targets.append(Target(name, "GET", url.strip()))
    return targets


def request_once(target: Target, timeout: float) -> Result:
    started = time.perf_counter()
    status_code = 0
    error = ""
    ok = False
    try:
        request = Request(target.url, method=target.method, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=timeout) as response:
            status_code = response.status
            response.read(512)
        ok = status_code == target.expected_status
    except HTTPError as exc:
        status_code = exc.code
        error = f"HTTPError: {exc.code}"
        ok = status_code == target.expected_status
    except URLError as exc:
        error = f"URLError: {exc.reason}"
    except TimeoutError:
        error = "Timeout"
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic script.
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000
    return Result(target=target.name, status_code=status_code, ok=ok, latency_ms=latency_ms, error=error)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((percent / 100) * (len(values) - 1)))))
    return values[index]


def summarize(results: list[Result], duration_seconds: float) -> dict:
    grouped: dict[str, list[Result]] = {}
    for result in results:
        grouped.setdefault(result.target, []).append(result)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(duration_seconds, 2),
        "total_requests": len(results),
        "requests_per_second": round(len(results) / duration_seconds, 2) if duration_seconds > 0 else 0,
        "targets": {},
    }

    for target_name, target_results in sorted(grouped.items()):
        latencies = [result.latency_ms for result in target_results]
        ok_count = sum(1 for result in target_results if result.ok)
        status_counts: dict[str, int] = {}
        error_counts: dict[str, int] = {}
        for result in target_results:
            status_counts[str(result.status_code)] = status_counts.get(str(result.status_code), 0) + 1
            if result.error:
                error_counts[result.error] = error_counts.get(result.error, 0) + 1

        count = len(target_results)
        summary["targets"][target_name] = {
            "requests": count,
            "ok": ok_count,
            "errors": count - ok_count,
            "error_rate_percent": round(((count - ok_count) / count) * 100, 2) if count else 0,
            "rps": round(count / duration_seconds, 2) if duration_seconds > 0 else 0,
            "latency_ms": {
                "min": round(min(latencies), 2) if latencies else 0,
                "avg": round(statistics.fmean(latencies), 2) if latencies else 0,
                "p50": round(percentile(latencies, 50), 2),
                "p95": round(percentile(latencies, 95), 2),
                "p99": round(percentile(latencies, 99), 2),
                "max": round(max(latencies), 2) if latencies else 0,
            },
            "status_counts": status_counts,
            "error_counts": error_counts,
        }

    return summary


def print_summary(summary: dict, expected_requests_per_user_per_minute: float) -> None:
    print("\n=== Medication_VQA Load Test Summary ===")
    print(f"Generated at: {summary['generated_at']}")
    print(f"Duration: {summary['duration_seconds']}s")
    print(f"Total requests: {summary['total_requests']}")
    print(f"Total RPS: {summary['requests_per_second']}")

    for target_name, data in summary["targets"].items():
        latency = data["latency_ms"]
        print(f"\n[{target_name}]")
        print(
            "requests={requests} ok={ok} errors={errors} error_rate={error_rate_percent}% rps={rps}".format(
                **data
            )
        )
        print(
            "latency_ms min={min} avg={avg} p50={p50} p95={p95} p99={p99} max={max}".format(
                **latency
            )
        )
        print(f"status_counts={data['status_counts']}")
        if data["error_counts"]:
            print(f"error_counts={data['error_counts']}")

    total_rps = float(summary["requests_per_second"])
    if expected_requests_per_user_per_minute > 0:
        requests_per_user_per_second = expected_requests_per_user_per_minute / 60
        estimated_active_users = total_rps / requests_per_user_per_second
        print("\n=== Rough Capacity Estimate ===")
        print(
            "If 1 active user generates about "
            f"{expected_requests_per_user_per_minute} requests/minute, "
            f"this test rate is roughly {estimated_active_users:.0f} active users."
        )
        print("Use this as a rough HTTP-read estimate, not a final LINE/Gemini/YOLO capacity number.")


def save_csv(results: list[Result], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["target", "status_code", "ok", "latency_ms", "error"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "target": result.target,
                    "status_code": result.status_code,
                    "ok": result.ok,
                    "latency_ms": round(result.latency_ms, 2),
                    "error": result.error,
                }
            )


def run_load_test(args: argparse.Namespace) -> tuple[list[Result], float]:
    targets = build_profile_targets(args.profile, args.base_url, args.drug, args.pdpa_base_url)
    targets.extend(parse_custom_targets(args.url))
    if not targets:
        raise SystemExit("No targets selected.")

    print("Targets:")
    for target in targets:
        print(f"- {target.name}: {target.method} {target.url}")

    results: list[Result] = []
    results_lock = threading.Lock()
    deadline = time.monotonic() + args.duration_seconds

    def worker(worker_index: int) -> None:
        if args.ramp_up_seconds > 0 and args.concurrency > 1:
            time.sleep((args.ramp_up_seconds / args.concurrency) * worker_index)
        request_index = worker_index
        while time.monotonic() < deadline:
            target = targets[request_index % len(targets)]
            result = request_once(target, args.timeout_seconds)
            with results_lock:
                results.append(result)
            request_index += args.concurrency
            if args.think_time_seconds > 0:
                time.sleep(args.think_time_seconds)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, index) for index in range(args.concurrency)]
        for future in futures:
            future.result()
    elapsed = time.monotonic() - started
    return results, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe read-only load test for Medication_VQA production.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--pdpa-base-url", default="http://127.0.0.1:17081")
    parser.add_argument(
        "--profile",
        choices=["health", "liff", "db", "pdpa-health", "production-read", "all-read"],
        default="production-read",
    )
    parser.add_argument("--drug", default=DEFAULT_DRUG)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--ramp-up-seconds", type=float, default=10)
    parser.add_argument("--think-time-seconds", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=20)
    parser.add_argument("--expected-requests-per-user-per-minute", type=float, default=2)
    parser.add_argument("--url", action="append", default=[], help="Custom GET target as name=https://... or https://...")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.duration_seconds < 1:
        raise SystemExit("--duration-seconds must be >= 1")

    results, elapsed = run_load_test(args)
    summary = summarize(results, elapsed)
    print_summary(summary, args.expected_requests_per_user_per_minute)

    if args.output_csv:
        save_csv(results, Path(args.output_csv))
        print(f"\nSaved CSV results to {args.output_csv}")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON summary to {args.output_json}")

    any_errors = any(target["errors"] > 0 for target in summary["targets"].values())
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
