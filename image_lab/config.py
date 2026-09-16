import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from image_lab.common import read_json


@dataclass
class Settings:
    root: Path
    registry_path: Path
    api_key: str
    load_timeout: float = 3600
    job_timeout: float = 1800
    max_queue: int = 100
    poll_interval: float = 0.5

    @classmethod
    def from_env(cls):
        return cls(
            root=Path(os.environ["DATA_ROOT"]).resolve(),
            registry_path=Path(os.getenv("MODEL_REGISTRY", "/app/config/models.json")),
            api_key=os.environ["API_KEY"],
            load_timeout=float(os.getenv("LOAD_TIMEOUT", "3600")),
            job_timeout=float(os.getenv("JOB_TIMEOUT", "1800")),
        )

    @property
    def deployment_id(self):
        return hashlib.sha256(str(self.root).encode()).hexdigest()[:12]


class Registry:
    def __init__(self, settings):
        self.settings = settings
        self.models = read_json(settings.registry_path)
        self.aliases = {}
        for key, model in self.models.items():
            model["id"] = key
            for alias in [key, model["name"], *model["aliases"]]:
                self.aliases[alias.lower()] = key
        for model in self.models.values():
            if source := model.get("weights_from"):
                shared = self.models.get(source)
                if shared is None or shared.get("weights_from") or source == model["id"]:
                    raise ValueError(f"Invalid shared weight source: {source}")
                for field in (
                    "repo",
                    "revision",
                    "auxiliary",
                    "allow_patterns",
                    "ignore_patterns",
                    "base_model",
                ):
                    if model.get(field) != shared.get(field):
                        raise ValueError(f"Shared weights differ in {field}: {model['id']} / {source}")

    def resolve(self, name):
        key = self.aliases.get(name.lower())
        if key is None:
            raise ValueError(f"Unknown model: {name}")
        return copy.deepcopy(self.models[key])

    def preparation_plan(self, names):
        """Resolve shared base checkpoints before adapters, downloading each only once."""
        ordered, visiting, done = [], set(), set()

        def visit(name):
            model = self.resolve(name)
            key = model["id"]
            if key in visiting:
                raise ValueError(f"Cyclic base_model dependency: {key}")
            if key in done:
                return
            visiting.add(key)
            if model.get("weights_from"):
                visit(model["weights_from"])
                visiting.remove(key)
                done.add(key)
                return
            if model.get("base_model"):
                visit(model["base_model"])
            visiting.remove(key)
            done.add(key)
            ordered.append(model)

        for name in names:
            visit(name)
        return ordered

    def base_snapshot(self, model):
        if not model.get("base_model"):
            return None
        base = self.resolve(model["base_model"])
        prepared = self.prepared(base)
        if prepared is None:
            return None
        return {"id": base["id"], **{key: prepared[key] for key in ("repo", "revision", "path")}}

    def prepared(self, model):
        marker = self.settings.root / "prepared" / f"{model.get('weights_from', model['id'])}.json"
        if not marker.exists():
            return None
        item = read_json(marker)
        if item["repo"] != model["repo"] or item["revision"] != model["revision"]:
            return None
        if not Path(item["path"]).is_dir():
            return None
        if item.get("auxiliary", []) != model.get("auxiliary", []):
            return None
        if item.get("allow_patterns") != model.get("allow_patterns"):
            return None
        if model.get("base_model"):
            base = self.base_snapshot(model)
            if base is None or item.get("base") != base:
                return None
        for auxiliary in model.get("auxiliary", []):
            repo_cache = (
                self.settings.root
                / "cache/huggingface/hub"
                / ("models--" + auxiliary["repo"].replace("/", "--"))
            )
            if not (repo_cache / "snapshots" / auxiliary["revision"] / auxiliary["probe"]).is_file():
                return None
            if auxiliary.get("role") and item.get("auxiliary_paths", {}).get(auxiliary["role"]) != str(
                repo_cache / "snapshots" / auxiliary["revision"]
            ):
                return None
        return item

    def public(self):
        result = []
        for model in self.models.values():
            item = copy.deepcopy(model)
            prepared = self.prepared(model)
            item["status"] = "cached" if prepared else "unavailable"
            item["reason"] = (
                None if prepared else model.get("blocked_reason", "Run start.sh to prepare weights")
            )
            item["resolution"] = (
                "256–2048 per edge, multiples of 16; >1 MP experimental, exact output checked"
            )
            result.append(item)
        return result


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
