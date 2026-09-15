import io
import time

from conftest import wait_job
from PIL import Image


def submit(client, model="flux", **fields):
    return client.post("/v1/jobs", json={"model": model, "prompt": "fox", **fields})


def test_auth_and_health(client):
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/v1/models").json()
    assert len(response["data"]) == 6
    assert next(m for m in response["data"] if m["id"] == "mage")["status"] == "unavailable"


def test_full_job_and_image_roundtrip(client):
    response = submit(client, model="FLUX.2 dev", seed=123)
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "succeeded"
    assert job["request"]["seed"] == 123
    assert job["request"]["model"] == "flux"
    assert job["load"]["cold_start"] is True
    assert job["queue_seconds"] >= 0
    image = client.get(job["image_url"])
    assert Image.open(io.BytesIO(image.content)).size == (1024, 1024)
    assert client.get(job["image_url"], headers={"Authorization": ""}).status_code == 401


def test_upload_and_edit(client):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32)).save(buffer, format="JPEG")
    upload = client.post(
        "/v1/uploads", files={"file": ("../not-a-path.jpg", buffer.getvalue(), "image/jpeg")}
    )
    assert upload.status_code == 201, upload.text
    upload_id = upload.json()["id"]
    result = submit(client, model="distil", images=[upload_id])
    assert result.status_code == 202
    job = wait_job(client, result.json()["id"])
    assert job["request"]["task"] == "image-edit"
    assert job["request"]["parameters"]["steps"] == 8
    assert client.post("/v1/uploads", files={"file": ("x.png", b"not an image")}).status_code == 422


def test_capabilities_parameters_and_path_validation(client):
    assert submit(client, "Mage-Flow-Edit").status_code == 422
    assert submit(client, "Ideogram 4", images=["a" * 32]).status_code == 422
    assert submit(client, "missing").status_code == 422
    assert submit(client, width=1025).status_code == 422
    assert submit(client, width="1024").status_code == 422
    assert submit(client, parameters={"steps": True}).status_code == 422
    assert submit(client, parameters={"arbitrary_shell": "echo bad"}).status_code == 422
    assert submit(client, images=["../../secret"]).status_code == 422
    assert submit(client, prompt=" ").status_code == 422
    assert client.get("/v1/jobs/missing").status_code == 404


def test_warm_reuse_switch_and_failure_recovery(client):
    for name in ("flux", "flux", "hunyuan", "flux"):
        job = wait_job(client, submit(client, name).json()["id"])
        assert job["status"] == "succeeded"
    runtime = client.app.state.scheduler.runtime
    assert runtime.loads == ["flux", "hunyuan", "flux"]
    failed = wait_job(client, submit(client, prompt="test-failure").json()["id"])
    assert failed["status"] == "failed"
    assert runtime.active_model is None
    again = wait_job(client, submit(client).json()["id"])
    assert again["status"] == "succeeded"
    assert again["load"]["cold_start"] is True


def test_running_and_queued_cancellation(client):
    active = submit(client, prompt="test-block").json()["id"]
    deadline = time.monotonic() + 2
    while client.get(f"/v1/jobs/{active}").json()["status"] != "running":
        assert time.monotonic() < deadline
        time.sleep(0.005)
    queued = submit(client).json()["id"]
    assert client.post(f"/v1/jobs/{queued}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/v1/jobs/{active}/cancel").status_code == 200
    assert wait_job(client, active)["status"] == "cancelled"
    assert wait_job(client, submit(client).json()["id"])["status"] == "succeeded"


def test_compatibility_endpoint(client):
    result = client.post("/v1/images/generations", json={"model": "flux", "prompt": "fox"})
    assert result.status_code == 200, result.text
    assert result.json()["data"][0]["b64_json"]
    assert (
        client.post("/v1/images/generations", json={"model": "flux", "prompt": "fox", "n": 2}).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/images/generations", json={"model": "flux", "prompt": "fox", "size": "2K"}
        ).status_code
        == 422
    )
