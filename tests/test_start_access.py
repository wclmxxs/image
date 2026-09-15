"""Exercise the real shell entry point with CPU-only stand-ins for Docker and Linux."""

import json
import os
import pty
import select
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from conftest import KEY, ROOT


@pytest.fixture
def start_harness(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("env.sh", "preflight.sh"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    shutil.copy(ROOT / "start.sh", repo / "start.sh")
    shutil.copytree(ROOT / "config", repo / "config")
    data = tmp_path / "data"
    data.mkdir()
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
    return SimpleNamespace(repo=repo, data=data, env=env, log=log, gpu_touched=gpu_touched)


@pytest.mark.parametrize(
    ("configured", "exported", "legacy_token", "expected"),
    [
        ('"hf_quoted_test_token"', None, None, "hf_quoted_test_token"),
        ("'hf_single_quoted_token'", None, None, "hf_single_quoted_token"),
        ("", "hf_exported_test_token", None, "hf_exported_test_token"),
        ("hf_config_test_token", "hf_override_test_token", None, "hf_override_test_token"),
        ("", None, "hf_ignored_legacy_token", ""),
        ("hf_config_test_token", None, "hf_ignored_legacy_token", "hf_config_test_token"),
        ("", "hf_exported_test_token", "hf_ignored_legacy_token", "hf_exported_test_token"),
        ("", None, None, ""),
    ],
)
def test_access_check_uses_resolved_token_without_gpu_operations(
    start_harness, configured, exported, legacy_token, expected
):
    repo, env = start_harness.repo, start_harness.env
    (repo / ".env").write_text(f"API_KEY={KEY}\nDATA_ROOT={start_harness.data}\nHF_TOKEN={configured}\n")
    if legacy_token is not None:
        (repo / ".hf-token.env").write_text(f"HF_TOKEN={legacy_token}\n")
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
    if expected:
        assert expected not in result.stdout + result.stderr
    assert not start_harness.gpu_touched.exists()
    calls = [json.loads(line) for line in start_harness.log.read_text().splitlines()]
    assert [call["args"][0] for call in calls] == ["info", "compose", "context", "build", "run"]
    assert calls[1]["args"] == ["compose", "version"]
    run = calls[-1]
    assert run["token"] == expected
    assert run["args"][run["args"].index("HF_TOKEN") - 1] == "-e"
    if expected:
        assert expected not in " ".join(run["args"])
    assert run["args"][-1] == "--check-only"
    assert "--gpus" not in run["args"]


def run_with_hidden_input(harness, token):
    master, slave = pty.openpty()
    process = subprocess.Popen(
        ["bash", "start.sh", "--check-access", "--ask-hf-token"],
        cwd=harness.repo,
        env=harness.env,
        stdin=slave,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    output = b""
    sent = False
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break  # Linux returns EIO when the child closes the PTY.
                if not chunk:
                    break
                output += chunk
                if not sent and b"Hugging Face read token (hidden): " in output:
                    os.write(master, token.encode() + b"\n")
                    sent = True
            elif process.poll() is not None:
                break
        assert sent, "Token input prompt did not appear"
        process.wait(timeout=2)
        return process.returncode, output.decode()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


@pytest.mark.parametrize("token", ["hf_prompted_test_token", ""])
def test_prompt_hides_input_does_not_persist_and_overrides_old_values(start_harness, token):
    harness = start_harness
    contents = f"API_KEY={KEY}\nDATA_ROOT={harness.data}\nHF_TOKEN=hf_old_config_token\n"
    (harness.repo / ".env").write_text(contents)
    harness.env["HF_TOKEN"] = "hf_old_exported_token"
    code, output = run_with_hidden_input(harness, token)
    assert (harness.repo / ".env").read_text() == contents
    assert not harness.gpu_touched.exists()
    if not token:
        assert code == 2
        assert "Token must not be empty" in output
        assert not harness.log.exists()
        return
    assert code == 0, output
    assert token not in output
    calls = [json.loads(line) for line in harness.log.read_text().splitlines()]
    assert calls[-1]["token"] == token
    assert calls[-1]["args"][-1] == "--check-only"
    assert token not in " ".join(calls[-1]["args"])


def test_prompt_requires_terminal_before_any_docker_operation(start_harness):
    harness = start_harness
    (harness.repo / ".env").write_text(f"API_KEY={KEY}\nDATA_ROOT={harness.data}\nHF_TOKEN=\n")
    result = subprocess.run(
        ["bash", "start.sh", "--check-access", "--ask-hf-token"],
        cwd=harness.repo,
        env=harness.env,
        capture_output=True,
        text=True,
        input="hf_test_token\n",
        timeout=15,
    )
    assert result.returncode == 2
    assert "requires an interactive terminal" in result.stderr
    assert not harness.log.exists()


@pytest.mark.parametrize(
    "models", ["flux-turbo,ideogram-instant,cosmos-4step", "flux,flux-turbo,cosmos,cosmos-4step"]
)
def test_start_builds_registry_images_and_shared_backends_once(start_harness, models):
    harness = start_harness
    (harness.repo / ".env").write_text(f"API_KEY={KEY}\nDATA_ROOT={harness.data}\nHF_TOKEN=\n")
    (harness.repo / "scripts/preflight.sh").write_text("#!/bin/sh\nexit 0\n")
    (harness.repo / "scripts/gpu_cleanup.py").write_text("# No real GPUs in this shell contract test.\n")
    stop = harness.repo / "stop.sh"
    stop.write_text("#!/bin/sh\nexit 0\n")
    stop.chmod(0o755)
    flock = harness.log.parent / "bin/flock"
    flock.write_text("#!/bin/sh\nexit 0\n")
    flock.chmod(0o755)
    result = subprocess.run(
        ["bash", "start.sh", "--models", models, "--no-gpu-cleanup"],
        cwd=harness.repo,
        env=harness.env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line)["args"] for line in harness.log.read_text().splitlines()]
    builds = [call for call in calls if call[0] == "build"]
    tags = [call[call.index("-t") + 1] for call in builds]
    assert tags.count("image-lab/flux:0.1.0") == 1
    assert tags.count("image-lab/cosmos:0.1.0") == 1
    assert tags.count("image-lab/base-cu126:0.1.0") == 1
    assert not any(
        "docker/flux-turbo.Dockerfile" in call or "docker/cosmos-4step.Dockerfile" in call for call in builds
    )
    if "ideogram-instant" in models:
        assert tags.count("image-lab/ideogram-instant:0.1.0") == 1
    cuda_checks = [call for call in calls if "--gpus" in call]
    assert len(cuda_checks) == (3 if "ideogram-instant" in models else 2)
    assert calls[-1][:4] == ["compose", "up", "-d", "--force-recreate"]
