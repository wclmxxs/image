"""CPU contract tests for distilled weights, dependencies and sampler plumbing."""

import base64
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import wait_job
from huggingface_hub.errors import GatedRepoError
from PIL import Image

from image_lab import prepare
from image_lab.backends import TURBO_SIGMAS, CosmosBackend, FluxBackend, IdeogramInstantBackend
from image_lab.config import Registry


@pytest.mark.parametrize(
    ("name", "model_id", "steps", "guidance"),
    [
        ("FLUX.2-dev-Turbo", "flux-turbo", 8, 2.5),
        ("Ideogram 4 Instant", "ideogram-instant", 8, 1.0),
        ("Cosmos3-Super-Text2Image-4Step", "cosmos-4step", 4, 1.0),
    ],
)
def test_variant_alias_fixed_parameters_and_switching(client, name, model_id, steps, guidance):
    payload = {"model": name, "prompt": "fox"}
    for params in ({"steps": 12}, {"guidance": 7}):
        result = client.post("/v1/jobs", json={**payload, "parameters": params})
        assert result.status_code == 422
        assert "requires" in result.text
    response = client.post("/v1/jobs", json=payload)
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "succeeded"
    assert job["request"]["model"] == model_id
    assert job["request"]["parameters"]["steps"] == steps
    assert job["request"]["parameters"]["guidance"] == guidance


def run_prepare(settings, monkeypatch, names, info):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["prepare", "--models", names])
    monkeypatch.setattr(prepare.Settings, "from_env", lambda: settings)
    api = Mock()
    api.model_info.side_effect = info
    monkeypatch.setattr(prepare, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(prepare, "get_hf_file_metadata", Mock())
    return api


def file_info(*names):
    return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=100) for n in names])


@pytest.mark.parametrize("base_cached", [True, False])
def test_turbo_only_prepares_shared_base_once(settings, monkeypatch, base_cached):
    (settings.root / "prepared/flux-turbo.json").unlink()
    if not base_cached:
        (settings.root / "prepared/flux.json").unlink()
    api = run_prepare(
        settings,
        monkeypatch,
        "flux-turbo,FLUX.2-dev-Turbo",
        lambda repo, **kwargs: (
            file_info("flux.2-turbo-lora.safetensors", "comfy/duplicate.safetensors")
            if repo.startswith("fal/")
            else file_info("transformer/model.safetensors")
        ),
    )
    download = Mock()
    monkeypatch.setattr(prepare, "snapshot_download", download)
    prepare.main()
    repos = [call.args[0] for call in api.model_info.call_args_list]
    assert repos == (
        ["fal/FLUX.2-dev-Turbo"] if base_cached else ["black-forest-labs/FLUX.2-dev", "fal/FLUX.2-dev-Turbo"]
    )
    calls = [call.kwargs for call in download.call_args_list]
    assert len(calls) == (1 if base_cached else 2)
    assert "comfy/*" not in calls[-1]["allow_patterns"]
    assert "flux.2-turbo-lora.safetensors" in calls[-1]["allow_patterns"]
    registry = Registry(settings)
    marker = registry.prepared(registry.resolve("flux-turbo"))
    assert marker["base"]["path"] == str(settings.root / "models/flux")
    assert marker["path"] != marker["base"]["path"]
    assert marker["base"]["revision"] == registry.resolve("flux")["revision"]
    (settings.root / "prepared/flux.json").unlink()
    assert registry.prepared(registry.resolve("flux-turbo")) is None


def test_missing_base_gate_blocks_adapter_download(settings, monkeypatch):
    for name in ("flux", "flux-turbo"):
        (settings.root / f"prepared/{name}.json").unlink()
    run_prepare(
        settings,
        monkeypatch,
        "flux-turbo",
        lambda *a, **k: file_info("flux.2-turbo-lora.safetensors", "transformer/model.safetensors"),
    )

    def gate(url, **kwargs):
        if "black-forest-labs" in url:
            raise GatedRepoError("denied")

    monkeypatch.setattr(prepare, "get_hf_file_metadata", gate)
    download = Mock()
    monkeypatch.setattr(prepare, "snapshot_download", download)
    with pytest.raises(RuntimeError, match="https://huggingface.co/black-forest-labs/FLUX.2-dev"):
        prepare.main()
    download.assert_not_called()


@pytest.mark.parametrize("authorized", [False, True])
def test_instant_shared_components_gate_filter_and_offline_marker(settings, monkeypatch, authorized):
    (settings.root / "prepared/ideogram-instant.json").unlink()
    run_prepare(
        settings,
        monkeypatch,
        "ideogram-instant",
        lambda *a, **k: file_info("transformer/diffusion_pytorch_model-00001-of-00004.safetensors"),
    )
    checks, downloads = [], []

    def gate(url, **kwargs):
        checks.append(url)
        if "ideogram-ai" in url and not authorized:
            raise GatedRepoError("components denied")

    def download(**kwargs):
        assert len(checks) == 2  # No download before both repositories' access checks.
        downloads.append(kwargs)
        if "cache_dir" in kwargs:
            snapshot = (
                Path(kwargs["cache_dir"])
                / ("models--" + kwargs["repo_id"].replace("/", "--"))
                / "snapshots"
                / kwargs["revision"]
            )
            return str(snapshot)  # Fixture already contains the probe.

    monkeypatch.setattr(prepare, "get_hf_file_metadata", gate)
    monkeypatch.setattr(prepare, "snapshot_download", download)
    if not authorized:
        with pytest.raises(RuntimeError, match="https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers"):
            prepare.main()
        assert downloads == []
        return
    prepare.main()
    assert len(downloads) == 2
    assert "transformer/*" not in downloads[1]["allow_patterns"]
    assert "unconditional_transformer/*" not in downloads[1]["allow_patterns"]
    registry = Registry(settings)
    model = registry.resolve("ideogram-instant")
    marker = registry.prepared(model)
    assert marker is not None
    assert marker["auxiliary_paths"]["components"].endswith(model["auxiliary"][0]["revision"])
    (Path(marker["auxiliary_paths"]["components"]) / "text_encoder/model.safetensors").unlink()
    assert registry.prepared(model) is None


def torch_standin(monkeypatch):
    generator = Mock()
    generator.manual_seed.return_value = generator

    class Module:
        def register_buffer(self, name, value, **kwargs):
            setattr(self, name, value)

    torch = SimpleNamespace(
        bfloat16="bf16",
        Generator=Mock(return_value=generator),
        nn=SimpleNamespace(Module=Module),
        empty=lambda *a, **kwargs: SimpleNamespace(dtype=kwargs["dtype"]),
        zeros_like=Mock(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def sample_job(**fields):
    return {
        "prompt": "fox",
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "image_paths": [],
        "parameters": {"steps": 8, "guidance": 2.5},
        **fields,
    }


def test_turbo_loads_base_and_local_adapter_with_published_sigmas(settings, monkeypatch):
    torch_standin(monkeypatch)
    pipe = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (1024, 1024))]))
    loader = Mock(return_value=pipe)
    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(Flux2Pipeline=SimpleNamespace(from_pretrained=loader))
    )
    registry = Registry(settings)
    model = registry.resolve("flux-turbo")
    model["weights"] = registry.prepared(model)
    backend = FluxBackend(model)
    assert loader.call_args.args == (str(settings.root / "models/flux"),)
    assert loader.call_args.kwargs["local_files_only"] is True
    pipe.load_lora_weights.assert_called_once_with(
        str(settings.root / "models/flux-turbo"),
        weight_name="flux.2-turbo-lora.safetensors",
        local_files_only=True,
    )
    backend.generate(sample_job())
    assert pipe.call_args.kwargs["sigmas"] == list(TURBO_SIGMAS)
    assert pipe.call_args.kwargs["num_inference_steps"] == 8
    assert pipe.call_args.kwargs["guidance_scale"] == 2.5


def test_instant_uses_local_components_no_negative_weights_and_distilled_settings(settings, monkeypatch):
    torch = torch_standin(monkeypatch)
    pipe = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (1024, 1024))]))
    loader, transformer = Mock(return_value=pipe), Mock()
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            Ideogram4Pipeline=SimpleNamespace(from_pretrained=loader),
            Ideogram4Transformer2DModel=SimpleNamespace(from_pretrained=transformer),
        ),
    )
    registry = Registry(settings)
    model = registry.resolve("ideogram-instant")
    model["weights"] = registry.prepared(model)
    backend = IdeogramInstantBackend(model)
    assert transformer.call_args.kwargs["local_files_only"] is True
    assert transformer.call_args.kwargs["subfolder"] == "transformer"
    assert loader.call_args.args == (model["weights"]["auxiliary_paths"]["components"],)
    assert loader.call_args.kwargs["unconditional_transformer"] is None
    assert loader.call_args.kwargs["local_files_only"] is True
    stub = pipe.register_modules.call_args.kwargs["unconditional_transformer"]
    assert stub.dtype == torch.bfloat16
    assert stub.forward(hidden_states="latents") == (torch.zeros_like.return_value,)
    backend.sample(sample_job(), '{"high_level_description":"fox"}')
    kwargs = pipe.call_args.kwargs
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["guidance_scale"] == 1.0 and kwargs["guidance_schedule"] is None
    assert kwargs["mu"] == 0.0 and kwargs["std"] == 1.75


def test_cosmos_distilled_launch_and_request_leave_schedule_to_checkpoint(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr("subprocess.Popen", launch)
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 1024)).save(buffer, format="PNG")
    body = json.dumps({"data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}]}).encode()
    requests = []

    def urlopen(request, **kwargs):
        if isinstance(request, str):
            health = io.BytesIO()
            health.status = 200
            return health
        requests.append(json.loads(request.data))
        return io.BytesIO(body)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    backend = CosmosBackend({"weights": {"path": "/checkpoints/cosmos-4step"}, "distilled": True})
    command = launch.call_args.args[0]
    assert command[2] == "/checkpoints/cosmos-4step"
    assert command[command.index("--hsdp-shard-size") + 1] == "4"
    assert "--use-hsdp" in command
    backend.generate(sample_job(parameters={"steps": 4, "guidance": 1.0, "prompt_mode": "text"}))
    assert "num_inference_steps" not in requests[0] and "flow_shift" not in requests[0]
    assert requests[0]["guidance_scale"] == 1.0
    assert requests[0]["extra_args"]["guardrails"] is True


def test_cosmos_http_error_includes_bounded_redacted_detail(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "private-test-value")
    body = io.BytesIO(b"resolution unsupported private-test-value " + b"x" * 10000)
    error = urllib.error.HTTPError("http://localhost", 500, "server error", {}, body)
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=error))
    backend = CosmosBackend.__new__(CosmosBackend)
    backend.url, backend.distilled = "http://localhost", True
    with pytest.raises(RuntimeError) as caught:
        backend.generate(sample_job(parameters={"steps": 4, "guidance": 1.0, "prompt_mode": "text"}))
    assert "resolution unsupported" in str(caught.value)
    assert "private-test-value" not in str(caught.value)
    assert len(str(caught.value)) < 2100
    assert body.tell() == 8192
