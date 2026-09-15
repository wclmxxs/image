import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from image_lab.backends import (
    CosmosBackend,
    FluxBackend,
    HunyuanBackend,
    MageBackend,
    json_caption,
    local_hub_file,
)
from image_lab.worker import verify_image


def test_exact_resolution_never_upscales():
    with pytest.raises(ValueError, match="No resize"):
        verify_image(Image.new("RGB", (1024, 1024)), {"width": 2048, "height": 2048})


def test_local_ideogram_loader_missing_and_traversal(tmp_path):
    from huggingface_hub.errors import EntryNotFoundError

    (tmp_path / "config.json").write_text("{}")
    assert local_hub_file(str(tmp_path), "config.json") == str(tmp_path / "config.json")
    for name in ("missing.json", "../outside.json", "/etc/passwd"):
        with pytest.raises(EntryNotFoundError):
            local_hub_file(str(tmp_path), name)
    with pytest.raises(ValueError):
        json_caption("[]")


def test_hunyuan_official_tuple_contract_and_resolution_precheck():
    backend = HunyuanBackend.__new__(HunyuanBackend)
    backend.distil = True
    group = SimpleNamespace(get_target_size=lambda w, h: (w, h))
    backend.model = SimpleNamespace(
        image_processor=SimpleNamespace(vae_reso_group=group),
        generate_image=Mock(return_value=(["caption"], [Image.new("RGB", (1024, 768))])),
    )
    job = {
        "prompt": "edit",
        "width": 1024,
        "height": 768,
        "seed": 42,
        "image_paths": ["reference.png"],
        "parameters": {"bot_task": "think_recaption", "steps": 8},
    }
    image, metadata = backend.generate(job)
    kwargs = backend.model.generate_image.call_args.kwargs
    assert kwargs["image_size"] == (768, 1024)
    assert kwargs["diff_infer_steps"] == 8
    assert kwargs["image"] == "reference.png"
    assert kwargs["infer_align_image_size"] is False
    assert metadata["generated_caption"] == ["caption"]
    assert image.size == (1024, 768)
    group.get_target_size = lambda w, h: (1024, 1024)
    with pytest.raises(ValueError, match="resolution preset"):
        backend.generate({**job, "width": 2048, "height": 2048})
    assert backend.model.generate_image.call_count == 1


def test_mage_passes_references_as_nested_batch():
    backend = MageBackend.__new__(MageBackend)
    backend.pipe = SimpleNamespace(edit=Mock(return_value=[Image.new("RGB", (1024, 1024))]))
    job = {
        "prompt": "edit",
        "image_paths": ["a.png", "b.png"],
        "width": 1024,
        "height": 1024,
        "seed": 7,
        "parameters": {"steps": 30, "guidance": 5},
    }
    backend.generate(job)
    assert backend.pipe.edit.call_args.args == (["edit"], [["a.png", "b.png"]])
    assert backend.pipe.edit.call_args.kwargs["seeds"] == [7]


def test_flux_reference_and_seed_contract(monkeypatch, tmp_path):
    import sys

    generator = Mock()
    generator.manual_seed.return_value = generator
    torch = SimpleNamespace(Generator=Mock(return_value=generator))
    monkeypatch.setitem(sys.modules, "torch", torch)
    backend = FluxBackend.__new__(FluxBackend)
    backend.turbo = False
    backend.pipe = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (1024, 1024))]))
    reference = tmp_path / "ref.png"
    Image.new("RGB", (32, 32)).save(reference)
    backend.generate(
        {
            "prompt": "edit",
            "image_paths": [str(reference)],
            "width": 1024,
            "height": 1024,
            "seed": 9,
            "parameters": {"steps": 50, "guidance": 4},
        }
    )
    kwargs = backend.pipe.call_args.kwargs
    assert kwargs["image"][0].size == (32, 32)
    assert kwargs["num_inference_steps"] == 50
    assert kwargs["generator"] is generator
    torch.Generator.assert_called_once_with("cpu")
    generator.manual_seed.assert_called_once_with(9)


def test_cosmos_request_matches_official_vllm_schema(monkeypatch):
    backend = CosmosBackend.__new__(CosmosBackend)
    backend.distilled = False
    backend.url = "http://127.0.0.1:8001"
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 1024)).save(buffer, format="PNG")
    body = json.dumps({"data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}]}).encode()
    captured = []

    def urlopen(request, **kwargs):
        captured.append(json.loads(request.data))
        return io.BytesIO(body)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    image, _ = backend.generate(
        {
            "prompt": "fox",
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "parameters": {"steps": 50, "guidance": 4, "prompt_mode": "text"},
        }
    )
    assert image.size == (1024, 1024)
    assert captured[0]["extra_args"] == {"guardrails": True, "use_resolution_template": False}
    assert captured[0]["num_inference_steps"] == 50
    assert captured[0]["flow_shift"] == 3
