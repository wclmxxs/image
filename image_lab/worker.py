import importlib.metadata
import os
import subprocess
import sys
import time
from pathlib import Path

from image_lab.common import atomic_json, read_json, safe_error


def gpu_usage():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return output.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None


def synchronize():
    import torch

    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def verify_image(image, job):
    expected = (job["width"], job["height"])
    if image.size != expected:
        raise ValueError(f"Backend returned {image.size}, requested {expected}. No resize was applied.")


def main():
    model = read_json(Path(sys.argv[1]))
    session = Path(model["session"])
    # Capture Python, native CUDA, and vLLM child logs in the shared diagnostics directory.
    log = open(session / "worker.log", "a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    from image_lab.backends import load_backend

    began = time.monotonic()
    try:
        backend = load_backend(model)
        synchronize()
        atomic_json(session / "ready.json", {"load_seconds": time.monotonic() - began})
    except Exception as error:
        atomic_json(session / "ready.json", {"error": safe_error(error)})
        print(safe_error(error), flush=True)
        return 1
    inbox = session / "inbox"
    inbox.mkdir(exist_ok=True)
    versions = {}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "peft",
        "ideogram-4",
        "vllm",
        "vllm-omni",
        "flashinfer-python",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    while True:
        for request_path in sorted(inbox.glob("*.json")):
            job = read_json(request_path)
            request_path.unlink()
            output_dir = Path(job["output_dir"])
            try:
                import torch

                for index in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(index)
                synchronize()
                start = time.monotonic()
                with torch.inference_mode():
                    image, details = backend.generate(job)
                synchronize()
                generation_seconds = time.monotonic() - start
                verify_image(image, job)
                image.save(output_dir / "image.png")
                prompt_seconds = details.get("prompt_seconds")
                result = {
                    "model": model["name"],
                    "weight_source": model["weights"],
                    "code_revision": model["code_revision"],
                    "dtype": model["dtype"],
                    "versions": versions,
                    "gpu_ids": model["gpus"],
                    "seed": job["seed"],
                    "width": image.width,
                    "height": image.height,
                    "parameters": job["parameters"],
                    "generation_seconds": generation_seconds,
                    "inference_seconds": generation_seconds - prompt_seconds
                    if prompt_seconds is not None
                    else None,
                    "experimental_resolution": job["width"] * job["height"] > 1024**2,
                    "gpu_memory_after": gpu_usage(),
                    "torch_peak_allocated_bytes": [
                        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
                    ],
                    "memory_note": "Cosmos runs in child processes; torch peaks in this wrapper do not measure vLLM memory",
                    **details,
                }
                atomic_json(output_dir / "result.json", result)
            except Exception as error:
                print(safe_error(error), flush=True)
                atomic_json(output_dir / "result.json", {"error": safe_error(error)})
                return 1
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
