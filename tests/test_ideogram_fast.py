import json
from unittest.mock import Mock

import pytest

from image_lab import prepare
from image_lab.config import Registry
from image_lab.schemas import JobRequest, validate_request


def test_fast_reuses_existing_instant_weights_without_a_new_download(settings, monkeypatch):
    registry = Registry(settings)
    fast = registry.resolve("ideogram-native-2k")
    assert registry.prepared(fast) == registry.prepared(registry.resolve("ideogram-instant"))
    assert [m["id"] for m in registry.preparation_plan([fast["id"], "ideogram-instant"])] == [
        "ideogram-instant"
    ]
    monkeypatch.setattr(prepare.Settings, "from_env", lambda: settings)
    monkeypatch.setattr("sys.argv", ["prepare", "--models", fast["id"]])
    api, download = Mock(), Mock()
    monkeypatch.setattr(prepare, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(prepare, "snapshot_download", download)
    prepare.main()
    api.model_info.assert_not_called()
    download.assert_not_called()
    (settings.root / "prepared/ideogram-instant.json").unlink()
    assert registry.prepared(fast) is None


def test_shared_weight_revision_mismatch_is_rejected(settings):
    config = json.loads(settings.registry_path.read_text())
    config["ideogram-instant-fast"]["revision"] = "different-weights"
    settings.registry_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Shared weights differ in revision"):
        Registry(settings)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_fast_compile_parameter_is_strict(settings, value):
    model = Registry(settings).resolve("ideogram-instant-fast")
    request = JobRequest(model=model["id"], prompt="fox", parameters={"compile": value})
    with pytest.raises(ValueError, match="compile must be a boolean"):
        validate_request(request, model)


def test_fast_fixed_steps_and_native_resolution(settings):
    model = Registry(settings).resolve("ideogram-instant-fast")
    request = JobRequest(
        model=model["id"], prompt="fox", width=2048, height=2048, parameters={"compile": False}
    )
    validated = validate_request(request, model)
    assert validated["parameters"]["steps"] == 8
    assert validated["parameters"]["guidance"] == 1.0
    assert validated["width"] == validated["height"] == 2048
    assert validated["parameters"]["compile"] is False
    with pytest.raises(ValueError, match="requires steps=8"):
        validate_request(request.model_copy(update={"parameters": {"steps": 4}}), model)
    with pytest.raises(ValueError, match="attention_backend must be one of"):
        validate_request(request.model_copy(update={"parameters": {"attention_backend": "auto"}}), model)
