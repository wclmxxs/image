import io
import json
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit

import pytest
from conftest import KEY, ROOT
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from PIL import Image
from requests import HTTPError, Response

from image_lab import prepare
from image_lab.common import safe_error
from image_lab.config import Registry
from scripts.client import Client


def configure_prepare(settings, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    for name in ("flux", "ideogram"):
        (settings.root / "prepared" / f"{name}.json").unlink()
    monkeypatch.setattr("sys.argv", ["prepare", "--models", "flux,ideogram"])
    monkeypatch.setattr(prepare.Settings, "from_env", lambda: settings)
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="ae.safetensors", size=512),
            SimpleNamespace(rfilename="flux2-dev.safetensors", size=1024),
            SimpleNamespace(rfilename="transformer/model.safetensors", size=1024),
        ]
    )
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
    assert checks[0].endswith("/transformer/model.safetensors")
    assert downloads[0]["ignore_patterns"] == ["ae.safetensors", "flux2-dev.safetensors"]
    assert all(download["token"] is False for download in downloads)
    registry = Registry(settings)
    for model in (registry.resolve("flux"), registry.resolve("ideogram")):
        assert registry.prepared(model)["revision"] == model["revision"]


def test_missing_token_does_not_attempt_whoami(capsys):
    api = Mock()
    prepare.verify_auth(api, False)
    api.whoami.assert_not_called()
    assert "HF_TOKEN is missing" in capsys.readouterr().out


def test_auth_reports_account_without_printing_token(capsys):
    api = Mock()
    api.whoami.return_value = {"name": "approved-test-user", "token": "hf_test_secret"}
    prepare.verify_auth(api, "hf_test_secret")
    api.whoami.assert_called_once_with(token="hf_test_secret")
    output = capsys.readouterr().out
    assert "authenticated as approved-test-user" in output
    assert "hf_test_secret" not in output


@pytest.mark.parametrize("status", [401, 403, 503])
@pytest.mark.parametrize("error_class", [HTTPError, HfHubHTTPError])
def test_rejected_or_unverifiable_token_stops_before_download(
    settings, monkeypatch, capsys, status, error_class
):
    configure_prepare(settings, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    api = Mock()
    response = Response()
    response.status_code = status
    api.whoami.side_effect = error_class("request contained hf_test_secret", response=response)
    monkeypatch.setattr(prepare, "HfApi", lambda **kwargs: api)
    download = Mock()
    monkeypatch.setattr(prepare, "snapshot_download", download)
    message = "HF_TOKEN was rejected" if status in {401, 403} else "Could not verify HF_TOKEN"
    with pytest.raises(RuntimeError, match=message) as caught:
        prepare.main()
    api.model_info.assert_not_called()
    download.assert_not_called()
    assert "hf_test_secret" not in str(caught.value) + capsys.readouterr().out


@pytest.mark.parametrize("token", ["", "hf_test_secret"])
def test_gate_report_identifies_auxiliary_repo_and_keeps_all_remediation(settings, monkeypatch, token):
    configure_prepare(settings, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", token)
    config = json.loads(settings.registry_path.read_text())
    config["cosmos"] = json.loads((ROOT / "config/models.json").read_text())["cosmos"]
    settings.registry_path.write_text(json.dumps(config))
    (settings.root / "prepared/cosmos.json").unlink()
    monkeypatch.setattr("sys.argv", ["prepare", "--models", "cosmos,flux,ideogram", "--check-only"])
    api = Mock()
    api.whoami.return_value = {"name": "approved-test-user"}
    api.model_info.return_value = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="model.safetensors", size=1024),
        ]
    )
    monkeypatch.setattr(prepare, "HfApi", lambda **kwargs: api)

    def gate(url, **kwargs):
        assert kwargs["token"] == (token or False)
        if "Cosmos3-Super-Text2Image" not in url:
            raise GatedRepoError("mock gated repository")

    monkeypatch.setattr(prepare, "get_hf_file_metadata", gate)
    download = Mock()
    monkeypatch.setattr(prepare, "snapshot_download", download)
    with pytest.raises(RuntimeError) as caught:
        prepare.main()
    report = safe_error(caught.value)
    for repo in ("nvidia/Cosmos-1.0-Guardrail", "black-forest-labs/FLUX.2-dev", "ideogram-ai/ideogram-4-fp8"):
        assert f"https://huggingface.co/{repo}" in report
    assert "Account approval or token read permission" in report if token else "No HF_TOKEN" in report
    assert "fine-grained token" in report
    assert "./start.sh --check-access" in report
    assert report.endswith("./start.sh --models hunyuan,hunyuan-distil")
    download.assert_not_called()


def test_prepared_snapshots_do_not_require_network_or_token(settings, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_expired_test_secret")
    monkeypatch.setattr("sys.argv", ["prepare", "--models", "flux,ideogram", "--check-only"])
    monkeypatch.setattr(prepare.Settings, "from_env", lambda: settings)
    api = Mock()
    monkeypatch.setattr(prepare, "HfApi", lambda **kwargs: api)
    prepare.main()
    api.whoami.assert_not_called()
    api.model_info.assert_not_called()


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
