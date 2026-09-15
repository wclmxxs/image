"""Check every requested gate first, then resumably download pinned snapshots."""

import argparse
import fnmatch
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download
from huggingface_hub.errors import GatedRepoError
from requests import HTTPError

from image_lab.common import atomic_json, safe_error
from image_lab.config import Registry, Settings

DEFAULT_MODELS = "cosmos,flux,ideogram,hunyuan,hunyuan-distil"


def verify_auth(api, token):
    if not token:
        print(
            "HF auth: HF_TOKEN is missing; using anonymous access. Set HF_TOKEN in .env or export it before start.sh. Host 'hf auth login' is not mounted into the container.",
            flush=True,
        )
        return
    try:
        account = api.whoami(token=token)
    except HTTPError as error:
        status = getattr(error.response, "status_code", None)
        if status in {401, 403}:
            raise RuntimeError(
                "HF_TOKEN was rejected by Hugging Face. Check whether it is invalid, revoked, or restricted; replace it in .env or the invoking shell (or .hf-token.env if using the fallback). Token value is not logged."
            ) from None
        raise RuntimeError(
            f"Could not verify HF_TOKEN (HTTP {status}); check Hugging Face connectivity and retry."
        ) from None
    except Exception:
        raise RuntimeError(
            "Could not verify HF_TOKEN; check Hugging Face connectivity and retry. Token value is not logged."
        ) from None
    print(f"HF auth: authenticated as {account.get('name', '(verified account)')}", flush=True)


def access_error(repo, token, error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(error, GatedRepoError) or status in {401, 403}:
        reason = (
            "Account approval or token read permission is missing."
            if token
            else "No HF_TOKEN was supplied to the download container."
        )
        return f"Download access denied: {repo}. {reason} Access request: https://huggingface.co/{repo}"
    return safe_error(error)


def local_manifest(path):
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and ".cache" not in item.relative_to(path).parts:
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            files.append(
                {
                    "file": str(item.relative_to(path)),
                    "bytes": item.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    if not files or not any(item["file"].endswith(".safetensors") for item in files):
        raise ValueError("Local snapshot must contain actual .safetensors weights")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    registry = Registry(settings)
    models = [registry.resolve(name.strip()) for name in args.models.split(",") if name.strip()]
    if not models:
        raise ValueError("At least one model must be selected")
    token = os.getenv("HF_TOKEN", "").strip() or False
    api = HfApi(token=token)
    if any(model["id"] != "mage" and registry.prepared(model) is None for model in models):
        verify_auth(api, token)
    pending = []
    failures = []
    for model in models:
        if registry.prepared(model):
            print(f"{model['name']}: pinned snapshot already prepared", flush=True)
            continue
        checked_repo = model["repo"]
        try:
            if model["id"] == "mage":
                path = Path(os.environ.get("MAGE_LOCAL_PATH", "/missing-mage-checkpoint")).resolve()
                if not path.is_dir() or not (path / "model_index.json").is_file():
                    raise ValueError(model["blocked_reason"])
                pending.append((model, path))
                continue
            info = api.model_info(model["repo"], revision=model["revision"], files_metadata=True)
            files = [
                item
                for item in info.siblings
                if not any(
                    fnmatch.fnmatch(item.rfilename, pattern) for pattern in model.get("ignore_patterns", [])
                )
            ]
            weight = next(item for item in files if item.rfilename.endswith(".safetensors"))
            # metadata listing alone may succeed for a gated model; HEAD an actual weight file.
            get_hf_file_metadata(
                hf_hub_url(model["repo"], weight.rfilename, revision=model["revision"]),
                token=token,
            )
            for auxiliary in model.get("auxiliary", []):
                checked_repo = auxiliary["repo"]
                get_hf_file_metadata(
                    hf_hub_url(auxiliary["repo"], auxiliary["probe"], revision=auxiliary["revision"]),
                    token=token,
                )
            size = sum(item.size or 0 for item in files)
            print(f"{model['name']}: download authorized; snapshot {size / 1e9:.1f} GB", flush=True)
            pending.append((model, None))
        except Exception as error:
            failures.append(f"{model['name']}: {access_error(checked_repo, token, error)}")
    if failures:
        raise RuntimeError(
            "Preflight failed before weight downloads:\n"
            + "\n".join(failures)
            + "\nFor access denials, open the repository pages with the SAME account as HF_TOKEN, "
            "complete the access requests and wait for approval. A fine-grained token also needs "
            "read access to public gated repositories you can access. Recheck: ./start.sh --check-access"
            + "\nTo test the public Hunyuan models first: ./start.sh --models hunyuan,hunyuan-distil"
        )
    if args.check_only:
        return
    for model, local_source in pending:
        target = settings.root / "models" / model["id"]
        target.mkdir(parents=True, exist_ok=True)
        manifest = None
        if local_source:
            if local_source != target.resolve():
                shutil.copytree(local_source, target, dirs_exist_ok=True)
            manifest = local_manifest(target)
            atomic_json(settings.root / "prepared" / "mage-files.json", manifest)
        else:
            snapshot_download(
                repo_id=model["repo"],
                revision=model["revision"],
                local_dir=target,
                token=token,
                max_workers=8,
                ignore_patterns=model.get("ignore_patterns"),
            )
            for auxiliary in model.get("auxiliary", []):
                cache_dir = settings.root / "cache/huggingface/hub"
                downloaded = Path(
                    snapshot_download(
                        repo_id=auxiliary["repo"],
                        revision=auxiliary["revision"],
                        cache_dir=cache_dir,
                        allow_patterns=auxiliary["patterns"],
                        token=token,
                    )
                )
                # The pinned guardrail library asks for main. Resolve main to our pinned revision
                # in this deployment's private cache, then run with HF_HUB_OFFLINE=1.
                refs = downloaded.parent.parent / "refs"
                refs.mkdir(exist_ok=True)
                (refs / "main").write_text(auxiliary["revision"])
        marker = {
            "repo": model["repo"],
            "revision": model["revision"],
            "path": str(target),
            "prepared_at": time.time(),
            "source": "local" if local_source else "huggingface",
            "auxiliary": model.get("auxiliary", []),
        }
        if manifest:
            import json

            marker["local_manifest_sha256"] = hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest()
        atomic_json(settings.root / "prepared" / f"{model['id']}.json", marker)
        print(f"{model['name']}: snapshot prepared", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        raise SystemExit(1)
