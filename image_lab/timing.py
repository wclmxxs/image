"""Request-local instrumentation; never changes sampling or model parameters."""

import functools
import time
from contextlib import contextmanager, nullcontext

from image_lab.common import safe_error


def synchronize_cuda():
    import torch

    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


class StageTimer:
    def __init__(self, mode="stages", synchronize=synchronize_cuda, clock=time.perf_counter):
        self.mode, self.synchronize, self.clock = mode, synchronize, clock
        self.records, self.stack, self.restorations = [], [], []
        self.notes, self.missing = [], []
        self.profile = None
        self.operator_profile = {"status": "not_requested"}

    @contextmanager
    def stage(self, name, input_shapes=None):
        if self.mode == "off":
            yield
            return
        self.synchronize()
        frame = {"name": name, "start": self.clock(), "children": 0.0, "depth": len(self.stack)}
        self.stack.append(frame)
        annotation = self.profile.record_function("image_lab::" + name) if self.profile else nullcontext()
        try:
            with annotation:
                yield
        finally:
            self.synchronize()
            seconds = self.clock() - frame["start"]
            self.stack.pop()
            if self.stack:
                self.stack[-1]["children"] += seconds
            self.records.append(
                {
                    "name": name,
                    "seconds": seconds,
                    "self_seconds": max(0.0, seconds - frame["children"]),
                    "depth": frame["depth"],
                    "input_shapes": input_shapes or {},
                }
            )

    def wrap(self, obj, attribute, name, *, required=True):
        if self.mode == "off":
            return
        original = getattr(obj, attribute, None) if obj is not None else None
        if not callable(original):
            if required:
                self.missing.append(str(name) if isinstance(name, str) else attribute)
            return
        # Restore the instance dictionary exactly, including Accelerate's forward hooks.
        own = attribute in vars(obj)
        previous = vars(obj).get(attribute)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            label = name(args, kwargs) if callable(name) else name
            shapes = input_shapes(args, kwargs) if label.startswith("denoiser") else None
            with self.stage(label, shapes):
                return original(*args, **kwargs)

        setattr(obj, attribute, measured)
        self.restorations.append((obj, attribute, own, previous))

    def restore(self):
        for obj, attribute, own, previous in reversed(self.restorations):
            if own:
                setattr(obj, attribute, previous)
            else:
                delattr(obj, attribute)
        self.restorations.clear()

    @contextmanager
    def operators(self, supported=True):
        """Kineto is optional. Profiling errors must not turn a valid image into a failed job."""
        if self.mode != "detailed" or not supported:
            if self.mode == "detailed":
                self.operator_profile = {"status": "unavailable", "reason": "inference_in_child_process"}
            yield
            return
        profiler = None
        entered = False
        try:
            import torch.profiler as tp

            activities = [tp.ProfilerActivity.CPU]
            if tp.ProfilerActivity.CUDA in tp.supported_activities():
                activities.append(tp.ProfilerActivity.CUDA)
            profiler = tp.profile(activities=activities, record_shapes=False, with_stack=False)
            profiler.__enter__()
            entered = True
            self.profile = tp
        except Exception as error:
            self.operator_profile = {"status": "unavailable", "reason": safe_error(error)}
        try:
            yield
        finally:
            self.profile = None
            if entered:
                try:
                    profiler.__exit__(None, None, None)
                    self.operator_profile = summarize_operators(profiler.key_averages())
                except Exception as error:
                    self.operator_profile = {"status": "unavailable", "reason": safe_error(error)}

    def result(self, generation_seconds):
        phases = {}
        denoiser_calls = []
        for record in self.records:
            name, seconds = record["name"], record["seconds"]
            phase = phases.setdefault(
                name, {"total_seconds": 0.0, "self_seconds": 0.0, "calls": 0, "per_call_seconds": []}
            )
            phase["total_seconds"] += seconds
            phase["self_seconds"] += record["self_seconds"]
            phase["calls"] += 1
            phase["per_call_seconds"].append(seconds)
            if name in {"denoiser", "denoiser_unconditional"}:
                denoiser_calls.append(
                    {
                        "index": len(denoiser_calls) + 1,
                        "component": name,
                        "seconds": seconds,
                        "input_shapes": record["input_shapes"],
                    }
                )
        for phase in phases.values():
            phase["mean_seconds"] = phase["total_seconds"] / phase["calls"]
            phase["max_seconds"] = max(phase["per_call_seconds"])
        observed = sum(r["seconds"] for r in self.records if r["depth"] == 0)
        unattributed = max(0.0, generation_seconds - observed)
        ranking = [
            {
                "phase": name,
                "self_seconds": phase["self_seconds"],
                "generation_fraction": phase["self_seconds"] / generation_seconds
                if generation_seconds
                else 0,
            }
            for name, phase in phases.items()
        ]
        ranking.append(
            {
                "phase": "unattributed",
                "self_seconds": unattributed,
                "generation_fraction": unattributed / generation_seconds if generation_seconds else 0,
            }
        )
        ranking.sort(key=lambda row: row["self_seconds"], reverse=True)
        return {
            "schema_version": 1,
            "mode": self.mode,
            "unit": "seconds",
            "clock": "cuda_synchronized_wall" if self.mode != "off" else "disabled",
            "generation_seconds": generation_seconds,
            "phases": phases,
            "generation_unattributed_seconds": unattributed,
            "phase_ranking": ranking if self.mode != "off" else [],
            "denoiser_calls": denoiser_calls,
            "operator_profile": self.operator_profile,
            "missing_probes": self.missing,
            "notes": [
                "Phase totals are inclusive; nested totals must not be added together. "
                "Use self_seconds plus generation_unattributed_seconds to reconcile generation_seconds.",
                "Denoiser calls are forward calls, not necessarily sampler steps; CFG can use two branches.",
                "Stage synchronization and detailed profiling add overhead. Use profiling=off for baseline latency.",
                *self.notes,
            ],
        }


def summarize_operators(events):
    rows = []
    for event in events:
        if not str(event.key).startswith("aten::"):
            continue
        rows.append(
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_seconds": max(0.0, event.self_cpu_time_total) / 1e6,
                "self_cuda_seconds": max(0.0, event.self_device_time_total) / 1e6,
                "total_cuda_seconds": max(0.0, event.device_time_total) / 1e6,
            }
        )
    rows.sort(key=lambda row: row["self_cuda_seconds"], reverse=True)
    attention = [r for r in rows if any(k in r["name"].lower() for k in ("attention", "flash", "efficient"))]
    return {
        "status": "collected" if any(r["self_cuda_seconds"] for r in rows) else "cuda_unavailable",
        "scope": "backend_generate",
        "top_cuda_operators": rows[:30],
        "attention_operators": attention,
        "note": "Operator CUDA time sums work across devices/streams; it is not wall-clock latency. "
        "Inclusive total_cuda_seconds overlaps between nested operators. "
        "Matmul operators include projections and other work, not only FFN. "
        "Attention names outside aten (e.g. custom fused kernels) may not appear here.",
    }


def input_shapes(args, kwargs):
    shapes = {}
    values = {
        key: kwargs[key]
        for key in ("hidden_states", "encoder_hidden_states", "x", "llm_features", "input_ids")
        if key in kwargs
    }
    if args:
        values["first_argument"] = args[0]
    for key, value in values.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes[key] = [int(d) for d in shape]
    return shapes


def install_probes(timer, backend, backend_name):
    """Pinned runtime entrypoints; missing probes are returned explicitly, never reported as zero."""
    if timer.mode == "off":
        return
    if backend_name in {"flux", "ideogram-instant"}:
        pipe = backend.pipe
        for method, name in (("encode_prompt", "text_encode"), ("prepare_latents", "latent_prepare")):
            timer.wrap(pipe, method, name, required=method == "encode_prompt")
        timer.wrap(pipe, "prepare_image_latents", "reference_encode", required=False)
        timer.wrap(pipe.transformer, "forward", "denoiser")
        timer.wrap(
            getattr(pipe, "unconditional_transformer", None),
            "forward",
            "denoiser_unconditional",
            required=False,
        )
        timer.wrap(pipe.vae, "encode", "vae_encode", required=False)
        timer.wrap(pipe.vae, "decode", "vae_decode")
        timer.wrap(pipe.scheduler, "step", "scheduler_step", required=False)
        timer.wrap(getattr(pipe, "image_processor", None), "postprocess", "image_postprocess", required=False)
    elif backend_name == "ideogram":
        pipe = backend.pipe
        timer.wrap(pipe, "_build_inputs", "input_prepare")
        timer.wrap(pipe, "_encode_text", "text_encode")
        timer.wrap(pipe.conditional_transformer, "forward", "denoiser")
        timer.wrap(pipe.unconditional_transformer, "forward", "denoiser_unconditional")
        timer.wrap(pipe, "_decode", "decode_and_postprocess")
        timer.wrap(pipe.autoencoder.decoder, "forward", "vae_decode")
    elif backend_name == "hunyuan":
        model = backend.model
        timer.wrap(model, "prepare_model_inputs", "input_prepare")
        timer.wrap(
            model,
            "generate",
            lambda args, kwargs: (
                "reasoning_recaption" if kwargs.get("mode", "gen_text") == "gen_text" else "image_sampling"
            ),
        )
        # The outer image generation forward also includes vision projections. Do not time
        # thousands of autoregressive reasoning tokens with a CUDA synchronization per token.
        original = model.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            if any(frame["name"] == "image_sampling" for frame in timer.stack):
                with timer.stage("denoiser", input_shapes(args, kwargs)):
                    return original(*args, **kwargs)
            return original(*args, **kwargs)

        own, previous = "forward" in vars(model), vars(model).get("forward")
        model.forward = forward
        timer.restorations.append((model, "forward", own, previous))
        timer.wrap(model.vae, "decode", "vae_decode")
        timer.wrap(model.vae, "encode", "vae_encode", required=False)
        timer.wrap(model.image_processor, "postprocess_outputs", "image_postprocess")
    elif backend_name == "mage":
        model = backend.pipe.model
        timer.wrap(model.txt_enc, "forward", "text_encode")
        timer.wrap(model.transformer, "forward", "denoiser")
        timer.wrap(model.vae, "decode", "vae_decode")
        timer.wrap(model.vae, "encode", "vae_encode", required=False)
    elif backend_name == "cosmos":
        timer.notes.append(
            "Cosmos inference runs in vLLM child processes. upstream_request includes "
            "tokenization, denoising, VAE, Guardrail and response encoding; those internal "
            "stages and CUDA operators are unavailable to this worker's profiler."
        )


@contextmanager
def instrument_backend(timer, backend, backend_name):
    # Worker owns one backend and runs one request at a time; no global class mutation.
    previous = getattr(backend, "timer", None)
    backend.timer = timer
    try:
        install_probes(timer, backend, backend_name)
        yield
    finally:
        timer.restore()
        if previous is None:
            del backend.timer
        else:
            backend.timer = previous


def timed_stage(backend, name):
    timer = getattr(backend, "timer", None)
    return timer.stage(name) if timer is not None else nullcontext()
