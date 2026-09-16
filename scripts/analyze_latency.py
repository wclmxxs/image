"""Compare warm latency profiles without confusing profiler time with serving latency."""

import hashlib
import json
import math
import statistics
from collections import Counter


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
        "max": ordered[-1],
    }


def analyze(path, target_seconds=4.0):
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("target_seconds must be finite and positive")
    path = path / "results.jsonl" if path.is_dir() else path
    groups, skipped = {}, Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        job, row = item["job"], item.get("summary", {})
        request = job["request"]
        result = job.get("result") or {}
        if (
            row.get("warmup")
            or job.get("load", {}).get("cold_start")
            or result.get("optimization", {}).get("first_shape_request")
        ):
            skipped["warmup_or_first_shape"] += 1
            continue
        identity = {
            "model": request["model"],
            "width": request.get("width", 1024),
            "height": request.get("height", 1024),
            "parameters": request.get("parameters", {}),
            "profiling": request.get("profiling", "stages"),
            "prompt_sha": hashlib.sha256(request["prompt"].encode()).hexdigest()[:8],
        }
        key = json.dumps(identity, sort_keys=True)
        group = groups.setdefault(key, {**identity, "generation": [], "service": [], "failed": 0})
        if job["status"] != "succeeded":
            group["failed"] += 1
            continue
        seconds = result.get("generation_seconds")
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            skipped["missing_latency"] += 1
            group["failed"] += 1
            continue
        group["generation"].append(seconds)
        service = result.get("timings", {}).get("request", {}).get("service_seconds")
        if isinstance(service, (int, float)) and math.isfinite(service) and service > 0:
            group["service"].append(service)
    output = []
    for group in groups.values():
        generation = distribution(group.pop("generation"))
        service = distribution(group.pop("service"))
        status = "miss"
        if group["failed"]:
            status = "failed_requests"
        elif group["profiling"] != "off":
            status = "diagnostic_only"
        elif not generation or generation["n"] < 10:
            status = "insufficient_samples"
        elif generation["p95"] < target_seconds:
            status = "pass"
        output.append(
            {
                **group,
                "generation_seconds": generation,
                "service_seconds": service,
                "generation_target_status": status,
            }
        )
    return {
        "target_seconds": target_seconds,
        "minimum_warm_samples": 10,
        "note": "Target applies to generation P95 only, not network/queue time or image quality. "
        "Nearest-rank P95 with 10 samples is preliminary; use more samples before production.",
        "groups": output,
        "skipped": dict(skipped),
    }


def print_analysis(report):
    for group in report["groups"]:
        dist = group["generation_seconds"]
        print(
            f"{group['model']} {group['width']}x{group['height']} {json.dumps(group['parameters'], sort_keys=True)}"
        )
        if dist:
            print(
                f"  warm n={dist['n']}: mean={dist['mean']:.4f}s p50={dist['p50']:.4f}s p95={dist['p95']:.4f}s max={dist['max']:.4f}s"
            )
        print(f"  generation P95 < {report['target_seconds']}s: {group['generation_target_status']}")
    print("Skipped:", json.dumps(report["skipped"]))
    print(report["note"])
