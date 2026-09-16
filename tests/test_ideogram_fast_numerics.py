"""Differential tests against the pinned real Diffusers model, using tiny weights.

Run in the worker image (or with torch/diffusers installed). No model download or
CUDA is required for the math checks. These do not establish H200 performance.
"""

from types import MethodType
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers import (  # noqa: E402
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Ideogram4Pipeline,
    Ideogram4Transformer2DModel,
)

from image_lab.ideogram_fast import FastIdeogram  # noqa: E402


@pytest.fixture
def pipe():
    torch.manual_seed(12)
    model = Ideogram4Transformer2DModel(
        in_channels=8,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        intermediate_size=48,
        adaln_dim=16,
        llm_features_dim=12,
        mrope_section=(2, 1, 1),
    ).eval()

    class Zero(torch.nn.Module):
        @property
        def dtype(self):
            return model.dtype

        def forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states),)

    pipeline = Ideogram4Pipeline(
        transformer=model,
        unconditional_transformer=Zero(),
        vae=AutoencoderKLFlux2(
            in_channels=3,
            out_channels=3,
            latent_channels=2,
            block_out_channels=(32, 32, 32, 32),
            layers_per_block=1,
            norm_num_groups=8,
        ).eval(),
        text_encoder=None,
        tokenizer=None,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipeline.set_progress_bar_config(disable=True)

    def encode(self, prompt, grid_h, grid_w, **kwargs):
        # Distinct prompts with identical lengths expose cross-request cache leakage.
        n = 4 if prompt == "short" else 6
        positions, segments, indicator = self._prepare_ids([n], grid_h, grid_w, 7, torch.device("cpu"))
        features = torch.zeros(1, 7 + grid_h * grid_w, 12)
        features[:, 7 - n : 7] = torch.randn(1, n, 12, generator=torch.Generator().manual_seed(n))
        return features, positions, segments, indicator

    pipeline.encode_prompt = MethodType(encode, pipeline)
    return pipeline


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prompt", ["short", "long"])
def test_compact_forward_matches_masked_reference(pipe, dtype, prompt):
    pipe.transformer.to(dtype=dtype)
    engine = FastIdeogram(pipe, attention_backend="_native_math")
    features, positions, segments, roles = pipe.encode_prompt(prompt, 2, 3)
    image = torch.randn(1, 6, 8).to(dtype)
    packed = torch.cat((torch.zeros(1, 7, 8, dtype=dtype), image), dim=1)
    with torch.inference_mode():
        reference = pipe.transformer(
            hidden_states=packed,
            timestep=torch.tensor([0.4], dtype=dtype),
            encoder_hidden_states=features.to(dtype),
            position_ids=positions,
            segment_ids=segments,
            indicator=roles,
            return_dict=False,
        )[0][:, 7:]
        condition = engine.prepare_condition(features, positions, segments, roles)
        actual = engine.forward(image, torch.tensor([0.4], dtype=dtype), condition)
    tolerance = 1e-5 if dtype == torch.float32 else 0.015
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    assert condition.image_tokens == 6
    assert condition.text_tokens + condition.removed_padding == 7


def test_eight_step_latents_and_schedule_match_stock_pipeline(pipe):
    engine = FastIdeogram(pipe, attention_backend="_native_math")
    with torch.inference_mode():
        reference = pipe(
            "short",
            width=32,
            height=48,
            max_sequence_length=7,
            num_inference_steps=8,
            guidance_scale=1.0,
            guidance_schedule=None,
            mu=0.0,
            std=1.75,
            generator=torch.Generator().manual_seed(43),
            output_type="latent",
        ).images
        expected_sigmas = pipe.scheduler.sigmas.clone()
        actual, metadata = engine.sample(
            "short", width=32, height=48, seed=43, compiled=False, output_type="latent"
        )
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(pipe.scheduler.sigmas, expected_sigmas, atol=0, rtol=0)
    assert metadata["denoiser_calls"] == 8


def test_native_decode_and_postprocess_match_stock_pipeline(pipe):
    engine = FastIdeogram(pipe, attention_backend="_native_math")
    with torch.inference_mode():
        reference = pipe(
            "short",
            width=32,
            height=48,
            max_sequence_length=7,
            num_inference_steps=8,
            guidance_scale=1.0,
            guidance_schedule=None,
            mu=0.0,
            std=1.75,
            generator=torch.Generator().manual_seed(43),
            output_type="pt",
        ).images[0]
        actual, _ = engine.sample("short", width=32, height=48, seed=43, compiled=False, output_type="pt")
    assert actual.shape == reference.shape == (3, 48, 32)
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)


def test_conditioning_does_not_leak_between_requests(pipe):
    engine = FastIdeogram(pipe, attention_backend="_native_math")
    with torch.inference_mode():
        first, meta = engine.sample(
            "short", width=32, height=32, seed=7, compiled=False, output_type="latent"
        )
        other, _ = engine.sample("long", width=32, height=32, seed=7, compiled=False, output_type="latent")
        repeated, repeated_meta = engine.sample(
            "short", width=32, height=32, seed=7, compiled=False, output_type="latent"
        )
    assert meta["first_shape_request"]
    assert not repeated_meta["first_shape_request"]
    assert not torch.equal(first, other)
    torch.testing.assert_close(first, repeated, atol=0, rtol=0)


@pytest.mark.parametrize("bad", ["segment", "padding", "roles", "batch"])
def test_mask_is_not_removed_for_unsupported_layout(pipe, bad):
    engine = FastIdeogram(pipe, attention_backend="_native_math")
    features, positions, segments, roles = pipe.encode_prompt("short", 2, 3)
    if bad == "segment":
        segments[0, -1] = 2
    elif bad == "padding":
        segments[0, 0] = 1
    elif bad == "roles":
        roles[0, -1] = 3
    else:
        roles = roles.repeat(2, 1)
    with torch.inference_mode(), pytest.raises(ValueError):
        engine.prepare_condition(features, positions, segments, roles)


def test_native_2k_keeps_all_16384_image_tokens(pipe):
    engine = FastIdeogram(pipe, attention_backend="_native_math")
    engine.forward = Mock(side_effect=lambda hidden_states, **kwargs: torch.zeros_like(hidden_states))
    with torch.inference_mode():
        latent, metadata = engine.sample(
            "short", width=2048, height=2048, seed=42, compiled=False, output_type="latent"
        )
    assert latent.shape == (1, 16384, 8)
    assert metadata["native_resolution"] and metadata["image_tokens"] == 16384
    assert engine.forward.call_count == 8
    assert all(c.kwargs["hidden_states"].shape[1] == 16384 for c in engine.forward.call_args_list)


def test_compiled_blocks_match_eager_with_dynamo_graph_capture(pipe, monkeypatch):
    # The eager backend exercises Dynamo's fullgraph tracing without relying on a
    # local GPU compiler. CUDA/Inductor performance still needs the H200 benchmark.
    compile_fn = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda block, **kwargs: compile_fn(block, backend="eager", fullgraph=True, dynamic=True),
    )
    engine = FastIdeogram(pipe, attention_backend="_native_math")
    with torch.inference_mode():
        eager, _ = engine.sample("short", width=32, height=32, seed=42, compiled=False, output_type="latent")
        compiled, _ = engine.sample(
            "short", width=32, height=32, seed=42, compiled=True, output_type="latent"
        )
    torch.testing.assert_close(eager, compiled, atol=1e-5, rtol=1e-5)
