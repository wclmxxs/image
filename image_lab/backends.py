"""GPU dependencies are imported only inside the corresponding worker image."""

import base64
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def json_caption(prompt):
    caption = json.loads(prompt)
    if not isinstance(caption, dict):
        raise ValueError("JSON prompt must be an object")
    return json.dumps(caption, ensure_ascii=False)


def ideogram_template(prompt):
    # Minimal valid schema; deliberately does not invent scene details or a visual style.
    return json.dumps(
        {
            "high_level_description": prompt,
            "compositional_deconstruction": {"background": "", "elements": [{"type": "obj", "desc": prompt}]},
        },
        ensure_ascii=False,
    )


def local_hub_file(repo_id, filename, **kwargs):
    """Ideogram's pinned loader calls hf_hub_download even for local components."""
    from huggingface_hub.errors import EntryNotFoundError

    root = Path(repo_id).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise EntryNotFoundError(f"Local snapshot component missing: {filename}")
    return str(path)


class FluxBackend:
    def __init__(self, model):
        import torch
        from diffusers import Flux2Pipeline

        placement = {"device_map": "balanced"} if len(model["gpus"]) > 1 else {}
        self.pipe = Flux2Pipeline.from_pretrained(
            model["weights"]["path"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **placement,
        )
        if not placement:
            self.pipe.to("cuda")

    def generate(self, job):
        import torch

        kwargs = {}
        if job["image_paths"]:
            kwargs["image"] = [Image.open(p).convert("RGB") for p in job["image_paths"]]
        params = job["parameters"]
        image = self.pipe(
            prompt=job["prompt"],
            width=job["width"],
            height=job["height"],
            num_inference_steps=params["steps"],
            guidance_scale=params["guidance"],
            generator=torch.Generator("cpu").manual_seed(job["seed"]),
            **kwargs,
        ).images[0]
        return image, {"effective_prompt": job["prompt"], "prompt_mode": "text", "prompt_seconds": 0}


class HunyuanBackend:
    def __init__(self, model):
        from hunyuan_image_3 import HunyuanImage3ForCausalMM

        self.distil = model["id"] == "hunyuan-distil"
        self.model = HunyuanImage3ForCausalMM.from_pretrained(
            model["weights"]["path"],
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
            moe_impl="flashinfer",
            moe_drop_tokens=True,
        )
        self.model.load_tokenizer(model["weights"]["path"])
        self.model.eval()

    def generate(self, job):
        paths = job["image_paths"]
        params = job["parameters"]
        actual = self.model.image_processor.vae_reso_group.get_target_size(job["width"], job["height"])
        if tuple(actual) != (job["width"], job["height"]):
            raise ValueError(
                f"Hunyuan resolution preset would produce {actual}; requested {(job['width'], job['height'])}"
            )
        caption, images = self.model.generate_image(
            prompt=job["prompt"],
            seed=job["seed"],
            image_size=(job["height"], job["width"]),
            use_system_prompt="en_unified" if self.distil and paths else None,
            bot_task=params["bot_task"],
            diff_infer_steps=params["steps"],
            verbose=0,
            max_new_tokens=2048,
            image=(paths[0] if len(paths) == 1 else paths) if paths else None,
            infer_align_image_size=False,
            use_taylor_cache=False,
        )
        return images[0], {
            "effective_prompt": job["prompt"],
            "generated_caption": caption,
            "prompt_mode": params["bot_task"],
            "prompt_seconds": None,
            "timing_note": "Native reasoning/recaption is included in generation_seconds",
        }


class IdeogramBackend:
    def __init__(self, model):
        import ideogram4.pipeline_ideogram4 as upstream
        import torch
        from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

        upstream.hf_hub_download = local_hub_file
        self.pipe = Ideogram4Pipeline.from_pretrained(
            config=Ideogram4PipelineConfig(weights_repo=model["weights"]["path"]),
            device="cuda",
            dtype=torch.bfloat16,
        )

    def generate(self, job):
        from ideogram4 import MAGIC_PROMPTS, PRESETS
        from ideogram4.magic_prompt import aspect_ratio_from_size

        params = job["parameters"]
        began = time.monotonic()
        mode = params["prompt_mode"]
        if mode == "json":
            prompt = json_caption(job["prompt"])
        elif mode == "magic":
            if not os.getenv("IDEOGRAM_API_KEY"):
                raise ValueError(
                    "prompt_mode=magic requires IDEOGRAM_API_KEY; it sends the prompt to Ideogram"
                )
            prompt = MAGIC_PROMPTS["ideogram-4-v1"](api_key=os.environ["IDEOGRAM_API_KEY"]).expand(
                job["prompt"],
                aspect_ratio=aspect_ratio_from_size(job["width"], job["height"]),
            )
        else:
            prompt = ideogram_template(job["prompt"])
        prompt_seconds = time.monotonic() - began
        text_key = os.getenv("HIVE_TEXT_MODERATION_KEY")
        visual_key = os.getenv("HIVE_VISUAL_MODERATION_KEY")
        if text_key or visual_key:
            from ideogram4.safety import moderate_image, moderate_prompt
        if text_key and moderate_prompt(prompt, text_key):
            raise ValueError("Prompt rejected by configured Hive text moderation")
        preset = PRESETS[params["preset"]]
        images = self.pipe(
            prompt,
            width=job["width"],
            height=job["height"],
            seed=job["seed"],
            num_steps=preset.num_steps,
            guidance_schedule=preset.guidance_schedule,
            mu=preset.mu,
            std=preset.std,
            raise_on_caption_issues=True,
        )
        if visual_key and moderate_image(images[0], visual_key):
            raise ValueError("Output rejected by configured Hive visual moderation")
        return images[0], {
            "effective_prompt": prompt,
            "prompt_mode": mode,
            "prompt_seconds": prompt_seconds,
            "steps": preset.num_steps,
            "safety": {
                "text": "hive" if text_key else "not_configured",
                "image": "hive" if visual_key else "not_configured",
            },
        }


class MageBackend:
    def __init__(self, model):
        from mage_flow import MageFlowPipeline

        self.pipe = MageFlowPipeline.from_pretrained(model["weights"]["path"], device="cuda")

    def generate(self, job):
        params = job["parameters"]
        images = self.pipe.edit(
            [job["prompt"]],
            [job["image_paths"]],
            heights=[job["height"]],
            widths=[job["width"]],
            seeds=[job["seed"]],
            steps=params["steps"],
            cfg=params["guidance"],
        )
        return images[0], {"effective_prompt": job["prompt"], "prompt_mode": "text", "prompt_seconds": 0}


class CosmosBackend:
    def __init__(self, model):
        self.url = "http://127.0.0.1:8001"
        self.process = subprocess.Popen(
            [
                "vllm",
                "serve",
                model["weights"]["path"],
                "--omni",
                "--host",
                "127.0.0.1",
                "--port",
                "8001",
                "--cfg-parallel-size",
                "2",
                "--ulysses-degree",
                "2",
                "--tensor-parallel-size",
                "1",
                "--init-timeout",
                "3000",
            ]
        )
        deadline = time.monotonic() + 3300
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM-Omni exited with {self.process.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError("Cosmos vLLM-Omni did not become healthy in 3300 seconds")

    def generate(self, job):
        params = job["parameters"]
        prompt = json_caption(job["prompt"]) if params["prompt_mode"] == "json" else job["prompt"]
        payload = {
            "prompt": prompt,
            "size": f"{job['width']}x{job['height']}",
            "n": 1,
            "num_inference_steps": params["steps"],
            "guidance_scale": params["guidance"],
            "flow_shift": 3.0,
            "negative_prompt": "",
            "seed": job["seed"],
            "extra_args": {"use_resolution_template": False, "guardrails": True},
        }
        request = urllib.request.Request(
            self.url + "/v1/images/generations",
            json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=7200) as response:
            data = json.load(response)
        image = Image.open(io.BytesIO(base64.b64decode(data["data"][0]["b64_json"], validate=True))).convert(
            "RGB"
        )
        return image, {
            "effective_prompt": prompt,
            "prompt_mode": params["prompt_mode"],
            "prompt_seconds": 0,
            "safety": "upstream guardrails enabled",
        }


def load_backend(model):
    backends = {
        "cosmos": CosmosBackend,
        "flux": FluxBackend,
        "ideogram": IdeogramBackend,
        "hunyuan": HunyuanBackend,
        "mage": MageBackend,
    }
    return backends[model["backend"]](model)
