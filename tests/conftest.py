import copy
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_lab.api import create_app
from image_lab.common import atomic_json
from image_lab.config import Registry, Settings
from image_lab.runtime import Cancelled

ROOT = Path(__file__).resolve().parents[1]
KEY = "unit-test-key-01234567890123456789"


class FakeRuntime:
    """Test-only execution: never shipped as a selectable production backend."""

    def __init__(self, settings, registry):
        self.settings = settings
        self.active_model = None
        self.loads = []
        self.stops = 0
        self.recovered = False

    def recover(self):
        self.recovered = True

    def ensure(self, model, interrupted):
        cold = model["id"] != self.active_model
        if cold:
            self.stop()
            self.loads.append(model["id"])
        self.active_model = model["id"]
        return {"cold_start": cold, "load_seconds": 0.01 if cold else 0}

    def generate(self, job, interrupted):
        if job["request"]["prompt"] == "test-failure":
            raise RuntimeError("simulated backend failure")
        if job["request"]["prompt"] == "test-block":
            deadline = time.monotonic() + 3
            while not interrupted() and time.monotonic() < deadline:
                time.sleep(0.005)
            raise Cancelled("simulated cancellation")
        if interrupted():
            raise Cancelled("cancelled")
        path = self.settings.root / "jobs" / job["id"] / "image.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (job["request"]["width"], job["request"]["height"])).save(path)
        return {"generation_seconds": 0.01, "parameters": job["request"]["parameters"]}

    def stop(self):
        self.stops += 1
        self.active_model = None

    def close(self):
        self.stop()


@pytest.fixture
def settings(tmp_path):
    config = json.loads((ROOT / "config/models.json").read_text())
    config = copy.deepcopy(config)
    registry_path = tmp_path / "models.json"
    registry_path.write_text(json.dumps(config))
    settings = Settings(tmp_path / "data", registry_path, KEY, poll_interval=0.005)
    registry = Registry(settings)
    for model in registry.preparation_plan(registry.models):
        if model["id"] == "mage":
            continue
        path = settings.root / "models" / model["id"]
        path.mkdir(parents=True)
        auxiliary_paths = {}
        for auxiliary in model.get("auxiliary", []):
            snapshot = (
                settings.root
                / "cache/huggingface/hub"
                / ("models--" + auxiliary["repo"].replace("/", "--"))
                / "snapshots"
                / auxiliary["revision"]
            )
            probe = snapshot / auxiliary["probe"]
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("CPU test fixture")
            if auxiliary.get("role"):
                auxiliary_paths[auxiliary["role"]] = str(snapshot)
        atomic_json(
            settings.root / "prepared" / f"{model['id']}.json",
            {
                "repo": model["repo"],
                "revision": model["revision"],
                "path": str(path),
                "auxiliary": model.get("auxiliary", []),
                "auxiliary_paths": auxiliary_paths,
                "allow_patterns": model.get("allow_patterns"),
                "base": registry.base_snapshot(model),
            },
        )
    return settings


@pytest.fixture
def client(settings):
    app = create_app(settings, runtime_factory=FakeRuntime)
    with TestClient(app, headers={"Authorization": f"Bearer {KEY}"}) as client:
        yield client


def wait_job(client, job_id):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.005)
    raise AssertionError(f"Job {job_id} did not finish")
