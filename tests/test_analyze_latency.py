import json

import pytest

from scripts.analyze_latency import analyze


def record(seconds, *, warmup=False, cold=False, first_shape=False, profiling="off", failed=False):
    return {
        "summary": {"warmup": warmup},
        "job": {
            "request": {
                "model": "ideogram-instant-fast",
                "prompt": "fox",
                "width": 2048,
                "height": 2048,
                "profiling": profiling,
                "parameters": {"compile": True},
            },
            "status": "failed" if failed else "succeeded",
            "load": {"cold_start": cold},
            "result": {
                "generation_seconds": seconds,
                "optimization": {"first_shape_request": first_shape},
                "timings": {"request": {"service_seconds": seconds + 1}},
            },
        },
    }


def test_latency_uses_warm_p95_and_keeps_service_separate(tmp_path):
    records = [record(100, warmup=True), record(80, cold=True), record(60, first_shape=True)]
    records += [record(3)] * 9 + [record(4.5)]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    result = analyze(tmp_path)
    group = result["groups"][0]
    assert group["generation_seconds"]["p50"] == 3
    assert group["generation_seconds"]["p95"] == 4.5
    assert group["service_seconds"]["p95"] == 5.5
    assert group["generation_target_status"] == "miss"
    assert result["skipped"]["warmup_or_first_shape"] == 3


@pytest.mark.parametrize(
    ("count", "profiling", "failed", "status"),
    [
        (10, "off", False, "pass"),
        (2, "off", False, "insufficient_samples"),
        (10, "detailed", False, "diagnostic_only"),
        (10, "off", True, "failed_requests"),
    ],
)
def test_latency_cannot_claim_success_for_diagnostic_or_failed_runs(
    tmp_path, count, profiling, failed, status
):
    records = [record(3, profiling=profiling)] * count
    if failed:
        records.append(record(3, failed=True))
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    assert analyze(tmp_path)["groups"][0]["generation_target_status"] == status


def test_latency_cli_is_offline_and_needs_no_api_key(tmp_path, monkeypatch, capsys):
    from scripts.client import main

    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(record(3)) for _ in range(10)))
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["client", "analyze-latency", str(tmp_path)])
    assert main() == 0
    assert "p95=3.0000s" in capsys.readouterr().out
