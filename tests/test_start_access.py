"""Exercise the real shell entry point with CPU-only stand-ins for Docker and Linux."""

import json
import os
import shutil
import subprocess
import sys

import pytest
from conftest import KEY, ROOT


@pytest.mark.parametrize(
    ("configured", "exported", "expected"),
    [
        ('"hf_quoted_test_token"', None, "hf_quoted_test_token"),
        ("'hf_single_quoted_token'", None, "hf_single_quoted_token"),
        ("", "hf_exported_test_token", "hf_exported_test_token"),
        ("hf_config_test_token", "hf_override_test_token", "hf_override_test_token"),
    ],
)
def test_access_check_uses_resolved_token_without_gpu_operations(tmp_path, configured, exported, expected):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("env.sh", "preflight.sh"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    shutil.copy(ROOT / "start.sh", repo / "start.sh")
    shutil.copytree(ROOT / "config", repo / "config")
    data = tmp_path / "data"
    data.mkdir()
    (repo / ".env").write_text(f"API_KEY={KEY}\nDATA_ROOT={data}\nHF_TOKEN={configured}\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["TEST_DOCKER_LOG"]).open("a") as log:
    log.write(json.dumps({"args": args, "token": os.environ.get("HF_TOKEN")}) + "\\n")
if args[:2] == ["context", "inspect"]:
    print("unix:///var/run/docker.sock")
"""
    )
    docker.chmod(0o755)
    uname = bin_dir / "uname"
    uname.write_text('#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo x86_64 ;; *) exit 99 ;; esac\n')
    uname.chmod(0o755)
    gpu = bin_dir / "nvidia-smi"
    gpu.write_text('#!/bin/sh\ntouch "$TEST_GPU_TOUCHED"\nexit 99\n')
    gpu.chmod(0o755)
    log = tmp_path / "docker.jsonl"
    gpu_touched = tmp_path / "gpu-touched"
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "HF_TOKEN",
            "MAGE_LOCAL_PATH",
            "DOCKER_HOST",
        }
    }
    env.update(PATH=f"{bin_dir}:{env['PATH']}", TEST_DOCKER_LOG=str(log), TEST_GPU_TOUCHED=str(gpu_touched))
    if exported is not None:
        env["HF_TOKEN"] = exported
    result = subprocess.run(
        ["bash", "start.sh", "--check-access"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "GPU workloads were not inspected or stopped" in result.stdout
    assert expected not in result.stdout + result.stderr
    assert not gpu_touched.exists()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call["args"][0] for call in calls] == ["info", "compose", "context", "build", "run"]
    assert calls[1]["args"] == ["compose", "version"]
    run = calls[-1]
    assert run["token"] == expected
    assert run["args"][run["args"].index("HF_TOKEN") - 1] == "-e"
    assert expected not in " ".join(run["args"])
    assert run["args"][-1] == "--check-only"
    assert "--gpus" not in run["args"]
