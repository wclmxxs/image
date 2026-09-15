import os
import time
import uuid
from pathlib import Path

from image_lab.common import atomic_json, read_json


class Cancelled(Exception):
    pass


class DockerRuntime:
    """Only the scheduler thread owns and mutates worker containers."""

    def __init__(self, settings, registry):
        import docker

        self.client = docker.from_env()
        self.settings = settings
        self.registry = registry
        self.container = None
        self.active_model = None
        self.session = None
        self.image_id = None

    @property
    def label(self):
        return f"image-lab.deployment={self.settings.deployment_id}"

    def recover(self):
        for container in self.client.containers.list(all=True, filters={"label": self.label}):
            container.remove(force=True)

    def stop(self):
        if self.container is not None:
            self.container.remove(force=True)
        self.container = None
        self.active_model = None

    def alive(self):
        if self.container is None:
            return False
        self.container.reload()
        return self.container.status == "running"

    def wait_json(self, path, timeout, interrupted):
        deadline = time.monotonic() + timeout
        while True:
            if interrupted():
                raise Cancelled("Job cancelled or service stopping")
            if path.exists():
                return read_json(path)
            if not self.alive():
                raise RuntimeError(f"Worker exited; inspect logs in {self.session}/worker.log")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Worker timeout after {timeout}s; inspect {self.session}/worker.log")
            time.sleep(self.settings.poll_interval)

    def ensure(self, model, interrupted):
        if self.active_model == model["id"] and self.alive():
            return {"cold_start": False, "load_seconds": 0, "image_id": self.image_id}
        began = time.monotonic()
        self.stop()
        prepared = self.registry.prepared(model)
        if prepared is None:
            raise RuntimeError("Model weights are not prepared")
        root = self.settings.root
        self.session = root / "sessions" / uuid.uuid4().hex
        self.session.mkdir(parents=True)
        specification = {**model, "weights": prepared, "session": str(self.session)}
        atomic_json(self.session / "model.json", specification)
        volumes = {str(root): {"bind": str(root), "mode": "rw"}}
        weight_path = Path(prepared["path"]).resolve()
        if not weight_path.is_relative_to(root):
            volumes[str(weight_path)] = {"bind": str(weight_path), "mode": "ro"}
        env = {
            "DATA_ROOT": str(root),
            "HF_HOME": str(root / "cache/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(root / "cache/inductor" / model["id"]),
            "TRITON_CACHE_DIR": str(root / "cache/triton" / model["id"]),
            "CUDA_CACHE_PATH": str(root / "cache/cuda" / model["id"]),
            "FLASHINFER_WORKSPACE_BASE": str(root / "cache/flashinfer"),
        }
        # Remote prompt expansion is opt-in. Model downloads never need credentials at runtime.
        if model["backend"] == "ideogram":
            for key in ("IDEOGRAM_API_KEY", "HIVE_TEXT_MODERATION_KEY", "HIVE_VISUAL_MODERATION_KEY"):
                if os.getenv(key):
                    env[key] = os.environ[key]
        import docker

        self.container = self.client.containers.run(
            model["image"],
            ["python3", "-m", "image_lab.worker", str(self.session / "model.json")],
            entrypoint=[],
            detach=True,
            init=True,
            shm_size="32g",
            volumes=volumes,
            environment=env,
            labels={"image-lab.deployment": self.settings.deployment_id},
            name=f"image-lab-{self.settings.deployment_id}-{model['id']}",
            device_requests=[
                docker.types.DeviceRequest(device_ids=[str(i) for i in model["gpus"]], capabilities=[["gpu"]])
            ],
            log_config=docker.types.LogConfig(type="json-file", config={"max-size": "20m", "max-file": "3"}),
        )
        self.image_id = self.container.image.id
        ready = self.wait_json(self.session / "ready.json", self.settings.load_timeout, interrupted)
        if "error" in ready:
            raise RuntimeError(ready["error"])
        self.active_model = model["id"]
        return {
            "cold_start": True,
            "load_seconds": time.monotonic() - began,
            "image_id": self.image_id,
            "worker_load_seconds": ready.get("load_seconds"),
        }

    def generate(self, job, interrupted):
        root = self.settings.root / "jobs" / job["id"]
        root.mkdir(parents=True, exist_ok=True)
        request = {**job["request"], "id": job["id"], "output_dir": str(root)}
        request["image_paths"] = [str(self.settings.root / "uploads" / f"{i}.png") for i in request["images"]]
        atomic_json(self.session / "inbox" / f"{job['id']}.json", request)
        result = self.wait_json(root / "result.json", self.settings.job_timeout, interrupted)
        if "error" in result:
            raise RuntimeError(result["error"])
        if not (root / "image.png").is_file():
            raise RuntimeError("Worker returned success without an image")
        return result

    def close(self):
        self.stop()
        self.client.close()
