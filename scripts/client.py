#!/usr/bin/env python3
"""Dependency-free client; works with the system Python on the AWS host."""

import argparse
import csv
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Client:
    def __init__(self, url, key):
        self.url, self.key = url.rstrip("/"), key

    def request(self, route, payload=None, method=None, binary=False, content_type=None):
        data = payload
        headers = {"Authorization": f"Bearer {self.key}"}
        if payload is not None and not isinstance(payload, bytes):
            data = json.dumps(payload, ensure_ascii=False).encode()
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self.url + route, data, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=65) as response:
                value = response.read()
                return value if binary else json.loads(value)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"HTTP {error.code}: {error.read().decode(errors='replace')[:2000]}"
            ) from error

    def upload(self, path):
        path = Path(path)
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"Reference image too large: {path}")
        boundary = uuid.uuid4().hex
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        header = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="reference"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        body = header + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        return self.request("/v1/uploads", body, content_type=f"multipart/form-data; boundary={boundary}")[
            "id"
        ]

    def generate(self, payload, output, timeout=7200):
        start = time.monotonic()
        payload = dict(payload)
        paths = payload.pop("image_paths", [])
        payload["images"] = [self.upload(path) for path in paths]
        job = self.request("/v1/jobs", payload)
        print(f"{payload['model']}: job {job['id']} submitted", file=sys.stderr, flush=True)
        deadline = time.monotonic() + timeout
        previous = None
        while True:
            job = self.request(f"/v1/jobs/{job['id']}")
            if job["status"] != previous:
                print(f"{job['id']}: {job['status']}", file=sys.stderr, flush=True)
                previous = job["status"]
            if job["status"] in TERMINAL:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Client wait exceeded {timeout}s; job continues. Poll /v1/jobs/{job['id']}"
                )
            time.sleep(1)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        job["client_seconds"] = time.monotonic() - start
        output.with_suffix(".json").write_text(json.dumps(job, ensure_ascii=False, indent=2))
        if job["status"] == "succeeded":
            output.write_bytes(self.request(job["image_url"], binary=True))
        return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default=os.getenv("IMAGE_API_URL", f"http://127.0.0.1:{os.getenv('PORT', '18080')}")
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models")
    get = sub.add_parser("job")
    get.add_argument("id")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("id")
    generate = sub.add_parser("generate")
    generate.add_argument("--model", required=True)
    prompt = generate.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 text or structured JSON caption file")
    generate.add_argument("--width", type=int, default=1024)
    generate.add_argument("--height", type=int, default=1024)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--image", action="append", default=[])
    generate.add_argument("--parameters", default="{}", help="JSON object, for example '{\"steps\": 8}'")
    generate.add_argument("--output", default="results/image.png")
    generate.add_argument("--profiling", choices=("off", "stages", "detailed"), default="stages")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--cases", default="config/benchmark.jsonl")
    bench.add_argument("--models", help="Optional comma-separated canonical model IDs to select")
    bench.add_argument("--repeat", type=int, default=3)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--output", default=None)
    bench.add_argument(
        "--profiling",
        choices=("off", "stages", "detailed"),
        help="Override profiling for every case (otherwise case value or stages)",
    )
    analyze_parser = sub.add_parser("analyze-timings", help="Compare warm 1K/2K stages from a benchmark")
    analyze_parser.add_argument("path", type=Path, help="Benchmark directory or results.jsonl")
    analyze_parser.add_argument("--output", type=Path, help="Optional JSON report path")
    latency_parser = sub.add_parser("analyze-latency", help="Compare warm latency against a P95 target")
    latency_parser.add_argument("path", type=Path)
    latency_parser.add_argument("--target-seconds", type=float, default=4.0)
    latency_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "analyze-latency":
        if __package__:
            from .analyze_latency import analyze, print_analysis
        else:
            from analyze_latency import analyze, print_analysis
        report = analyze(args.path, args.target_seconds)
        print_analysis(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["groups"] else 1
    if args.command == "analyze-timings":
        if __package__:
            from .analyze_timings import analyze, print_analysis
        else:
            from analyze_timings import analyze, print_analysis

        report = analyze(args.path)
        print_analysis(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["groups"] else 1
    key = os.environ.get("API_KEY")
    if not key:
        raise ValueError("Set API_KEY, or use ./lab on the deployment host")
    client = Client(args.url, key)
    if args.command == "models":
        print(json.dumps(client.request("/v1/models"), indent=2, ensure_ascii=False))
    elif args.command == "job":
        print(json.dumps(client.request(f"/v1/jobs/{args.id}"), indent=2, ensure_ascii=False))
    elif args.command == "cancel":
        print(json.dumps(client.request(f"/v1/jobs/{args.id}/cancel", {}, method="POST")))
    elif args.command == "generate":
        job = client.generate(
            {
                "model": args.model,
                "prompt": args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt,
                "width": args.width,
                "height": args.height,
                "seed": args.seed,
                "image_paths": args.image,
                "parameters": json.loads(args.parameters),
                "profiling": args.profiling,
            },
            args.output,
        )
        print(json.dumps(job, indent=2, ensure_ascii=False))
        return 0 if job["status"] == "succeeded" else 1
    elif args.command == "benchmark":
        if args.repeat < 1 or args.warmup < 0:
            raise ValueError("repeat must be >=1, warmup must be >=0")
        root = Path(args.output or f"results/benchmark-{time.strftime('%Y%m%d-%H%M%S')}")
        root.mkdir(parents=True, exist_ok=False)
        cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
        if args.models:
            cases = [case for case in cases if case["model"] in args.models.split(",")]
        if not cases:
            raise ValueError("No benchmark cases selected")
        # Keep each model resident across its resolution cases.
        cases.sort(key=lambda item: item["model"])
        failed = False
        fields = [
            "case",
            "model",
            "width",
            "height",
            "parameters",
            "profiling",
            "timings",
            "iteration",
            "warmup",
            "status",
            "cold_start",
            "load_seconds",
            "generation_seconds",
            "inference_seconds",
            "queue_seconds",
            "client_seconds",
            "error",
        ]
        with (
            (root / "summary.csv").open("w", newline="") as stream,
            (root / "results.jsonl").open("w") as raw,
        ):
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for index, case in enumerate(cases):
                for iteration in range(args.warmup + args.repeat):
                    payload = {**case, "seed": case.get("seed", 42) + iteration}
                    if args.profiling:
                        payload["profiling"] = args.profiling
                    row = {
                        "case": index,
                        "model": case["model"],
                        "width": case.get("width", 1024),
                        "height": case.get("height", 1024),
                        "iteration": iteration,
                        "warmup": iteration < args.warmup,
                        "parameters": json.dumps(case.get("parameters", {}), sort_keys=True),
                        "profiling": payload.get("profiling", "stages"),
                    }
                    try:
                        job = client.generate(payload, root / f"case-{index}-{iteration}.png")
                        row.update(
                            status=job["status"],
                            error=job.get("error", ""),
                            queue_seconds=job.get("queue_seconds"),
                            client_seconds=job["client_seconds"],
                            parameters=json.dumps(job["request"]["parameters"], sort_keys=True),
                            timings=json.dumps(job.get("result", {}).get("timings"), ensure_ascii=False),
                        )
                        row.update({k: job.get("load", {}).get(k) for k in ("cold_start", "load_seconds")})
                        row.update(
                            {
                                k: job.get("result", {}).get(k)
                                for k in ("generation_seconds", "inference_seconds")
                            }
                        )
                    except Exception as error:
                        job = {"request": payload, "status": "failed", "error": str(error)}
                        row.update(status="failed", error=str(error))
                    raw.write(json.dumps({"summary": row, "job": job}, ensure_ascii=False) + "\n")
                    raw.flush()
                    writer.writerow(row)
                    stream.flush()
                    if row["status"] != "succeeded":
                        failed = True
                        # Do not repeat an unavailable/OOM/unsupported case and obscure the original failure.
                        break
        print(f"Benchmark saved to {root}; warmups are explicitly marked in summary.csv")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, urllib.error.URLError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
