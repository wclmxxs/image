import copy
import json
import subprocess
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from conftest import ROOT, FakeRuntime, wait_job

from image_lab.timing import StageTimer, instrument_backend, summarize_operators
from image_lab.worker import generate_with_timings
from scripts.analyze_timings import analyze


class Clock:
    def __init__(self):
        self.now, self.pending, self.syncs = 0.0, 0.0, 0

    def __call__(self):
        return self.now

    def synchronize(self):
        self.now += self.pending
        self.pending = 0
        self.syncs += 1


def test_nested_stages_wait_for_gpu_and_do_not_double_count():
    clock = Clock()
    timer = StageTimer(synchronize=clock.synchronize, clock=clock)
    clock.pending = 50  # Previously queued work must not be charged to this stage.
    with timer.stage("decode_and_postprocess"):
        clock.now += 1
        with timer.stage("vae_decode"):
            clock.pending += 3
        clock.now += 2
    result = timer.result(7)
    assert result["phases"]["decode_and_postprocess"]["total_seconds"] == 6
    assert result["phases"]["decode_and_postprocess"]["self_seconds"] == 3
    assert result["phases"]["vae_decode"]["self_seconds"] == 3
    assert result["generation_unattributed_seconds"] == 1
    assert sum(r["self_seconds"] for r in result["phase_ranking"]) == 7
    assert sum(r["generation_fraction"] for r in result["phase_ranking"]) == pytest.approx(1)


def test_request_local_wrappers_preserve_outputs_and_restore_on_failure():
    class Component:
        def forward(self, value):
            return value + 1

    component = Component()

    def accelerated(value):
        return value * 2

    component.forward = accelerated
    clock = Clock()
    timer = StageTimer(synchronize=clock.synchronize, clock=clock)
    backend = SimpleNamespace(
        pipe=SimpleNamespace(
            transformer=component,
            vae=SimpleNamespace(),
            scheduler=SimpleNamespace(),
        )
    )
    with pytest.raises(ValueError):
        with instrument_backend(timer, backend, "flux"):
            for _ in range(8):
                assert component.forward(3) == 6
            raise ValueError("backend failed")
    assert component.forward is accelerated
    assert not hasattr(backend, "timer")
    result = timer.result(1)
    assert result["phases"]["denoiser"]["calls"] == 8
    assert len(result["denoiser_calls"]) == 8
    assert "vae_decode" in result["missing_probes"]
    assert "vae_decode" not in result["phases"]
    # A second request must not accumulate hooks or records from the first.
    second = StageTimer(synchronize=clock.synchronize, clock=clock)
    with instrument_backend(second, backend, "flux"):
        component.forward(4)
    assert second.result(1)["phases"]["denoiser"]["calls"] == 1


def test_off_does_not_inspect_components_or_synchronize():
    timer = StageTimer("off", synchronize=lambda: pytest.fail("unexpected synchronization"))
    backend = SimpleNamespace()
    with instrument_backend(timer, backend, "flux"):
        with timer.stage("anything"):
            pass
    assert timer.result(2)["phases"] == {}
    assert timer.result(2)["phase_ranking"] == []


def test_fast_ideogram_profiles_actual_engine_instead_of_unused_reference_forward():
    fast = SimpleNamespace(prepare_condition=lambda: None, forward=lambda **kwargs: "velocity")
    backend = SimpleNamespace(
        fast=fast,
        pipe=SimpleNamespace(
            transformer=SimpleNamespace(forward=lambda: pytest.fail("reference forward is not used")),
            vae=SimpleNamespace(),
            scheduler=SimpleNamespace(),
        ),
    )
    timer = StageTimer(synchronize=lambda: None)
    with instrument_backend(timer, backend, "ideogram-instant"):
        fast.prepare_condition()
        for _ in range(8):
            assert fast.forward(hidden_states=SimpleNamespace(shape=(1, 16384, 128))) == "velocity"
    result = timer.result(1)
    assert result["phases"]["conditioning_prepare"]["calls"] == 1
    assert result["phases"]["denoiser"]["calls"] == 8
    assert result["denoiser_calls"][0]["input_shapes"]["hidden_states"] == [1, 16384, 128]


def test_hunyuan_splits_reasoning_from_image_forwards_and_restores():
    class Model:
        def __init__(self):
            self.vae = SimpleNamespace(decode=lambda: None)
            self.image_processor = SimpleNamespace(postprocess_outputs=lambda: None)

        def forward(self):
            return "prediction"

        def prepare_model_inputs(self):
            return {}

        def generate(self, mode="gen_text"):
            self.forward()
            self.forward()
            return mode

    model = Model()
    backend = SimpleNamespace(model=model)
    timer = StageTimer(synchronize=lambda: None)
    with instrument_backend(timer, backend, "hunyuan"):
        assert model.generate(mode="gen_text") == "gen_text"
        assert model.generate(mode="gen_image") == "gen_image"
    phases = timer.result(1)["phases"]
    assert phases["reasoning_recaption"]["calls"] == 1
    assert phases["image_sampling"]["calls"] == 1
    assert phases["denoiser"]["calls"] == 2  # Text autoregression is not mislabeled as diffusion.
    assert "forward" not in vars(model)
    assert "generate" not in vars(model)


def test_profiler_reports_seconds_and_actual_attention_names_without_summing_nested_totals():
    rows = [
        SimpleNamespace(
            key=name, count=8, self_cpu_time_total=1000, self_device_time_total=own, device_time_total=total
        )
        for name, own, total in [
            ("aten::scaled_dot_product_attention", 0, 2000000),
            ("aten::_scaled_dot_product_flash_attention", 2000000, 2000000),
            ("aten::mm", 3000000, 3000000),
            ("image_lab::denoiser", 0, 6000000),
        ]
    ]
    result = summarize_operators(rows)
    assert result["status"] == "collected"
    assert result["top_cuda_operators"][0]["name"] == "aten::mm"
    assert result["top_cuda_operators"][0]["self_cuda_seconds"] == 3
    assert len(result["attention_operators"]) == 2
    assert sum(r["self_cuda_seconds"] for r in result["attention_operators"]) == 2
    assert summarize_operators([])["status"] == "cuda_unavailable"


def test_worker_generation_keeps_details_and_marks_cosmos_operator_visibility(monkeypatch):
    from image_lab.timing import timed_stage

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(device_count=lambda: 0),
            inference_mode=nullcontext,
        ),
    )

    class Backend:
        def generate(self, job):
            with timed_stage(self, "upstream_request"):
                return "image", {"sampling_schedule": "checkpoint-fixed-4step"}

    backend = Backend()
    image, details, seconds, timings = generate_with_timings(
        backend,
        {"backend": "cosmos"},
        {"profiling": "detailed"},
    )
    assert image == "image" and details["sampling_schedule"] == "checkpoint-fixed-4step"
    assert timings["generation_seconds"] == seconds
    assert timings["phases"]["upstream_request"]["calls"] == 1
    assert timings["operator_profile"]["status"] == "unavailable"
    assert timings["output"]["profiling_setup_finalize_seconds"] >= 0
    assert not hasattr(backend, "timer")


def test_flux_worker_measures_all_eight_forwards_and_shapes_without_changing_image(monkeypatch):
    from PIL import Image

    from image_lab.backends import FluxBackend

    generator = SimpleNamespace(manual_seed=lambda seed: seed)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(device_count=lambda: 0),
            inference_mode=nullcontext,
            Generator=lambda device: generator,
        ),
    )

    class Pipe:
        def __init__(self):
            self.transformer = SimpleNamespace(forward=lambda **kwargs: kwargs["hidden_states"])
            self.vae = SimpleNamespace(decode=lambda x: x)
            self.scheduler = SimpleNamespace(step=lambda x: x)
            self.image_processor = SimpleNamespace(postprocess=lambda x: Image.new("RGB", (32, 32), "red"))

        def encode_prompt(self, prompt):
            return prompt

        def prepare_latents(self):
            return SimpleNamespace(shape=(1, 4096, 128))

        def __call__(self, **kwargs):
            self.encode_prompt(kwargs["prompt"])
            state = self.prepare_latents()
            for _ in range(kwargs["num_inference_steps"]):
                state = self.transformer.forward(hidden_states=state)
                state = self.scheduler.step(state)
            return SimpleNamespace(images=[self.image_processor.postprocess(self.vae.decode(state))])

    backend = FluxBackend.__new__(FluxBackend)
    backend.pipe, backend.turbo = Pipe(), True
    job = {
        "profiling": "stages",
        "prompt": "fox",
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "image_paths": [],
        "parameters": {"steps": 8, "guidance": 2.5},
    }
    raw_image = backend.generate(job)[0]
    image, _, _, timings = generate_with_timings(backend, {"backend": "flux"}, job)
    assert image.tobytes() == raw_image.tobytes()
    assert timings["phases"]["denoiser"]["calls"] == 8
    assert timings["phases"]["scheduler_step"]["calls"] == 8
    for name in ("text_encode", "latent_prepare", "vae_decode", "image_postprocess"):
        assert timings["phases"][name]["calls"] == 1
    assert timings["denoiser_calls"][0]["input_shapes"]["hidden_states"] == [1, 4096, 128]
    assert "encode_prompt" not in vars(backend.pipe)


@pytest.mark.parametrize("failure", ["start", "collect", "backend"])
def test_profiler_failures_do_not_break_generation_or_mask_model_errors(monkeypatch, failure):
    tp, torch = ModuleType("torch.profiler"), ModuleType("torch")
    torch.profiler = tp
    tp.ProfilerActivity = SimpleNamespace(CPU="cpu", CUDA="cuda")
    tp.supported_activities = lambda: {"cpu"}
    tp.record_function = lambda name: nullcontext()

    class Profiler:
        def __enter__(self):
            if failure == "start":
                raise RuntimeError("CUPTI unavailable")

        def __exit__(self, *args):
            if failure in {"collect", "backend"}:
                raise RuntimeError("cannot collect events")

    tp.profile = lambda **kwargs: Profiler()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.profiler", tp)
    timer = StageTimer("detailed", synchronize=lambda: None)
    expectation = pytest.raises(ValueError, match="model error") if failure == "backend" else nullcontext()
    with expectation:
        with timer.operators():
            with timer.stage("denoiser"):
                if failure == "backend":
                    raise ValueError("model error")
    assert timer.result(1)["operator_profile"]["status"] == "unavailable"
    assert timer.result(1)["phases"]["denoiser"]["calls"] == 1


@pytest.mark.parametrize("mode", ["stages", "detailed", "off"])
def test_api_modes_and_compat_response_return_timings(client, monkeypatch, mode):
    original = FakeRuntime.generate

    def generate(self, job, interrupted):
        result = original(self, job, interrupted)
        result["timings"] = {"mode": job["request"]["profiling"], "phases": {"denoiser": {"calls": 8}}}
        return result

    monkeypatch.setattr(FakeRuntime, "generate", generate)
    response = client.post("/v1/jobs", json={"model": "flux", "prompt": "fox", "profiling": mode})
    job = wait_job(client, response.json()["id"])
    timings = job["result"]["timings"]
    assert timings["mode"] == mode
    assert timings["request"]["load_seconds"] == job["load"]["load_seconds"]
    assert timings["request"]["queue_seconds"] == job["queue_seconds"]
    response = client.post(
        "/v1/images/generations",
        json={
            "model": "flux",
            "prompt": "fox",
            "profiling": mode,
            "response_format": "url",
        },
    )
    assert response.status_code == 200
    assert response.json()["timings"]["mode"] == mode
    assert response.json()["timings"]["phases"]["denoiser"]["calls"] == 8
    for route in ("/v1/jobs", "/v1/images/generations"):
        assert (
            client.post(route, json={"model": "flux", "prompt": "fox", "profiling": "typo"}).status_code
            == 422
        )


def test_analysis_excludes_cold_failed_and_different_prompt_or_mode(tmp_path):
    def record(size, seconds, *, warmup=False, cold=False, prompt="fox", mode="stages"):
        return {
            "summary": {"warmup": warmup},
            "job": {
                "status": "succeeded",
                "load": {"cold_start": cold},
                "request": {
                    "model": "flux-turbo",
                    "parameters": {"steps": 8},
                    "prompt": prompt,
                    "width": size,
                    "height": size,
                },
                "result": {
                    "generation_seconds": seconds,
                    "timings": {
                        "mode": mode,
                        "phases": {"denoiser": {"self_seconds": seconds - 1}},
                        "generation_unattributed_seconds": 1,
                    },
                },
            },
        }

    rows = [
        record(1024, 5),
        record(2048, 25),
        record(1024, 100, cold=True),
        record(1024, 100, warmup=True),
        record(1024, 7, prompt="cat"),
        record(1024, 9, mode="detailed"),
    ]
    failed = copy.deepcopy(rows[0])
    failed["job"]["status"] = "failed"
    rows.append(failed)
    file = tmp_path / "results.jsonl"
    file.write_text("\n".join(json.dumps(r) for r in rows))
    result = analyze(tmp_path)
    assert len(result["groups"]) == 3
    assert result["skipped"] == {"warmup_or_cold": 2, "failed": 1}
    group = next(g for g in result["groups"] if g["one_k_to_two_k"])
    assert group["one_k_to_two_k"][0]["phase"] == "denoiser"
    assert group["one_k_to_two_k"][0]["added_seconds"] == 20
    assert group["one_k_to_two_k"][0]["ratio"] == 6
    # Exercise the same script entry point as ./lab, not just Python package imports.
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/client.py"), "analyze-timings", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    assert "6.00x" in run.stdout
