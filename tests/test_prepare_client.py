import io
import json
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit

import pytest
from conftest import KEY
from PIL import Image

from image_lab import prepare
from image_lab.config import Registry
from scripts.client import Client


def configure_prepare(settings, monkeypatch):
    for name in ("flux", "ideogram"):
        (settings.root / "prepared" / f"{name}.json").unlink()
    monkeypatch.setattr("sys.argv", ["prepare", "--models", "flux,ideogram"])
    monkeypatch.setattr(prepare.Settings, "from_env", lambda: settings)
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename="model.safetensors", size=1024)])
    monkeypatch.setattr(prepare, "HfApi", lambda **kwargs: SimpleNamespace(model_info=lambda *a, **k: info))


def test_all_gates_checked_before_any_weight_download(settings, monkeypatch):
    configure_prepare(settings, monkeypatch)
    download = Mock()
    monkeypatch.setattr(prepare, "snapshot_download", download)

    def gate(url, **kwargs):
        if "ideogram" in url:
            raise PermissionError("model gate not accepted")

    monkeypatch.setattr(prepare, "get_hf_file_metadata", gate)
    with pytest.raises(RuntimeError, match="Preflight failed before weight downloads"):
        prepare.main()
    download.assert_not_called()
    assert not (settings.root / "prepared/flux.json").exists()


def test_successful_prepare_pins_revisions_and_skips_duplicate_flux_files(settings, monkeypatch):
    configure_prepare(settings, monkeypatch)
    checks = []
    monkeypatch.setattr(prepare, "get_hf_file_metadata", lambda url, **kwargs: checks.append(url))
    downloads = []

    def download(**kwargs):
        assert len(checks) == 2
        downloads.append(kwargs)

    monkeypatch.setattr(prepare, "snapshot_download", download)
    prepare.main()
    assert downloads[0]["ignore_patterns"] == ["ae.safetensors", "flux2-dev.safetensors"]
    registry = Registry(settings)
    for model in (registry.resolve("flux"), registry.resolve("ideogram")):
        assert registry.prepared(model)["revision"] == model["revision"]


def test_cli_upload_submit_poll_download_uses_the_actual_http_contract(client, tmp_path, monkeypatch):
    def bridge(request, **kwargs):
        route = urlsplit(request.full_url).path
        response = client.request(
            request.get_method(), route, content=request.data, headers=dict(request.header_items())
        )
        assert response.status_code < 400, response.text
        return io.BytesIO(response.content)

    monkeypatch.setattr("urllib.request.urlopen", bridge)
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32)).save(reference)
    output = tmp_path / "result.png"
    test_client = Client("http://testserver", KEY)
    job = test_client.generate({"model": "flux", "prompt": "edit", "image_paths": [str(reference)]}, output)
    assert job["status"] == "succeeded"
    assert Image.open(output).size == (1024, 1024)
    assert json.loads(output.with_suffix(".json").read_text())["request"]["task"] == "image-edit"
