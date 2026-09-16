"""Summarize completed benchmark timings without contacting the server."""

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(path):
    path = Path(path)
    if path.is_dir():
        path = path / "results.jsonl"
    groups = defaultdict(list)
    skipped = defaultdict(int)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        job, summary = record["job"], record["summary"]
        if job.get("status") != "succeeded":
            skipped["failed"] += 1
            continue
        if summary.get("warmup") or job.get("load", {}).get("cold_start"):
            skipped["warmup_or_cold"] += 1
            continue
        timings = job.get("result", {}).get("timings") or {}
        if not timings.get("phases"):
            skipped["no_stage_timings"] += 1
            continue
        request = job["request"]
        key = json.dumps(
            {
                "model": request["model"],
                "parameters": request["parameters"],
                "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
                "images": request.get("images", []),
                "profiling": timings["mode"],
            },
            sort_keys=True,
        )
        groups[key].append(job)
    output = []
    for key, jobs in groups.items():
        sizes = defaultdict(list)
        for job in jobs:
            sizes[(job["request"]["width"], job["request"]["height"])].append(job)
        resolutions = {}
        for (width, height), cases in sorted(sizes.items()):
            phase_values = defaultdict(list)
            for job in cases:
                timings = job["result"]["timings"]
                for name, phase in timings["phases"].items():
                    phase_values[name].append(phase["self_seconds"])
                phase_values["unattributed"].append(timings["generation_unattributed_seconds"])
            resolutions[f"{width}x{height}"] = {
                "n": len(cases),
                "generation_seconds": statistics.mean(j["result"]["generation_seconds"] for j in cases),
                "phases": {
                    name: {"self_seconds": statistics.mean(values), "observed_n": len(values)}
                    for name, values in phase_values.items()
                },
            }
        one, two = resolutions.get("1024x1024"), resolutions.get("2048x2048")
        deltas = []
        if one and two:
            for name in one["phases"].keys() & two["phases"].keys():
                a, b = one["phases"][name], two["phases"][name]
                # Missing probes/calls are not zeros. Compare only phases present in every sample.
                if a["observed_n"] != one["n"] or b["observed_n"] != two["n"]:
                    continue
                deltas.append(
                    {
                        "phase": name,
                        "one_k_seconds": a["self_seconds"],
                        "two_k_seconds": b["self_seconds"],
                        "added_seconds": b["self_seconds"] - a["self_seconds"],
                        "ratio": b["self_seconds"] / a["self_seconds"] if a["self_seconds"] else None,
                    }
                )
            deltas.sort(key=lambda r: r["added_seconds"], reverse=True)
        output.append({**json.loads(key), "resolutions": resolutions, "one_k_to_two_k": deltas})
    return {
        "groups": output,
        "skipped": dict(skipped),
        "note": "Warmups, cold loads and failed jobs are excluded. Groups preserve model, prompt, "
        "parameters, reference IDs and profiling mode. Self times avoid double counting. "
        "API success is not an image quality assessment.",
    }


def print_analysis(report):
    for group in report["groups"]:
        print(f"\n{group['model']} / {group['profiling']} / prompt={group['prompt_sha256'][:8]}")
        for size, row in group["resolutions"].items():
            print(f"  {size}: {row['generation_seconds']:.4f}s, warm n={row['n']}")
        if group["one_k_to_two_k"]:
            print("  Phase                       1K(s)     2K(s)     Added(s)  Ratio")
            for row in group["one_k_to_two_k"]:
                ratio = f"{row['ratio']:.2f}x" if row["ratio"] is not None else "n/a"
                print(
                    f"  {row['phase']:<27} {row['one_k_seconds']:>8.4f} {row['two_k_seconds']:>9.4f} "
                    f"{row['added_seconds']:>10.4f} {ratio:>7}"
                )
    print("\nSkipped:", json.dumps(report["skipped"], ensure_ascii=False))
