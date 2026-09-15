from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from image_lab.common import atomic_json, safe_error
from image_lab.config import Registry
from image_lab.runtime import Cancelled, DockerRuntime
from image_lab.store import JobStore


def test_durable_recovery_and_queue_limit(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    store = JobStore(path)
    first = store.add({"model": "flux"}, 2)
    second = store.add({"model": "flux"}, 2)
    with pytest.raises(OverflowError):
        store.add({}, 2)
    store.claim()
    store.close()
    store = JobStore(path)
    store.recover()
    assert store.get(first["id"])["status"] == "interrupted"
    assert store.get(second["id"])["status"] == "queued"
    assert store.claim()["id"] == second["id"]
    store.close()


def test_weight_revision_mismatch(settings):
    registry = Registry(settings)
    model = registry.resolve("flux")
    marker = settings.root / "prepared/flux.json"
    value = registry.prepared(model)
    value["revision"] = "changed"
    atomic_json(marker, value)
    assert registry.prepared(model) is None


def test_docker_boundary_and_ownership(settings, monkeypatch):
    client = Mock()
    client.containers.list.return_value = []
    container = Mock()
    container.image.id = "sha256:test"
    client.containers.run.return_value = container
    monkeypatch.setattr("docker.from_env", lambda: client)
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-workers")
    runtime = DockerRuntime(settings, Registry(settings))
    runtime.recover()
    assert client.containers.list.call_args.kwargs["filters"] == {"label": runtime.label}
    monkeypatch.setattr(runtime, "wait_json", lambda *args: {"load_seconds": 0.1})
    model = runtime.registry.resolve("flux")
    assert runtime.ensure(model, lambda: False)["cold_start"] is True
    options = client.containers.run.call_args.kwargs
    assert "HF_TOKEN" not in options["environment"]
    assert options["device_requests"][0]["DeviceIDs"] == ["0"]
    assert options["labels"] == {"image-lab.deployment": settings.deployment_id}
    assert options["volumes"][str(settings.root)]["bind"] == str(settings.root)
    assert options["entrypoint"] == []
    runtime.stop()
    container.remove.assert_called_once_with(force=True)


def test_runtime_timeout_cancel_and_dead_process(tmp_path):
    runtime = DockerRuntime.__new__(DockerRuntime)
    runtime.settings = SimpleNamespace(poll_interval=0.001)
    runtime.session = tmp_path
    runtime.alive = lambda: True
    with pytest.raises(TimeoutError):
        runtime.wait_json(tmp_path / "never.json", 0.005, lambda: False)
    with pytest.raises(Cancelled):
        runtime.wait_json(tmp_path / "never.json", 1, lambda: True)
    runtime.alive = lambda: False
    with pytest.raises(RuntimeError, match="Worker exited"):
        runtime.wait_json(tmp_path / "never.json", 1, lambda: False)


def test_secret_redaction(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_private_test_value")
    assert "hf_private_test_value" not in safe_error(RuntimeError("URL failed hf_private_test_value"))
