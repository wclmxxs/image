"""Single-image Ideogram Instant execution with request-local, immutable conditioning.

Forward/sampling equations adapted from Hugging Face Diffusers v0.39.0:
Copyright 2026 Ideogram AI and The HuggingFace Team. Apache-2.0.
See THIRD_PARTY_NOTICES.md. No weights, step schedule, or resolution are changed.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_ideogram4 import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
    _logit_normal_sigmas,
    _resolution_aware_mu,
)


@dataclass(frozen=True)
class Conditioning:
    text: torch.Tensor
    image_indicator: torch.Tensor
    rotary: tuple[torch.Tensor, torch.Tensor]
    text_tokens: int
    image_tokens: int
    removed_padding: int


class FastIdeogram:
    """Keep all tensors local to a request; compiled blocks only retain model weights.

    Only batch=1, [left padding][text][image], one active segment is accepted.
    Padding can then be removed because it cannot attend to/from the active segment.
    Unsupported layouts fail explicitly instead of silently dropping an attention mask.
    """

    def __init__(self, pipe, *, attention_backend="_native_flash"):
        self.pipe = pipe
        self.model = pipe.transformer
        self.attention_backend = attention_backend
        self.model.eval()
        self.model.set_attention_backend(attention_backend)
        self.compiled_blocks = None
        self.seen_shapes = set()

    def prepare_condition(self, encoder_hidden_states, position_ids, segment_ids, indicator):
        if indicator.ndim != 2 or indicator.shape[0] != 1:
            raise ValueError("Fast Ideogram requires batch=1")
        if segment_ids.shape != indicator.shape:
            raise ValueError("Ideogram segment shape does not match token layout")
        roles = indicator[0].tolist()  # One CPU sync per request, outside the denoising loop.
        try:
            text_start = roles.index(LLM_TOKEN_INDICATOR)
            image_start = roles.index(OUTPUT_IMAGE_INDICATOR)
        except ValueError:
            raise ValueError("Fast Ideogram requires nonempty text and image regions") from None
        expected = (
            [0] * text_start
            + [LLM_TOKEN_INDICATOR] * (image_start - text_start)
            + [OUTPUT_IMAGE_INDICATOR] * (len(roles) - image_start)
        )
        if roles != expected or image_start <= text_start:
            raise ValueError("Unsupported Ideogram layout: expected [padding][text][image]")
        segments = segment_ids[0].tolist()
        if (
            len(set(segments[text_start:])) != 1
            or segments[text_start] == SEQUENCE_PADDING_INDICATOR
            or any(s != SEQUENCE_PADDING_INDICATOR for s in segments[:text_start])
        ):
            raise ValueError("Fast Ideogram requires one isolated active attention segment")
        if encoder_hidden_states.shape[:2] != indicator.shape or position_ids.shape != (*indicator.shape, 3):
            raise ValueError("Ideogram conditioning shapes do not match the token layout")

        # Image positions have zero text features and are masked again after projection
        # in the reference. Project only actual text, once, rather than L x 53248 eight times.
        text = encoder_hidden_states[:, text_start:image_start].to(self.model.dtype)
        text = self.model.llm_cond_proj(self.model.llm_cond_norm(text))
        text_role = torch.zeros((1, 1), device=text.device, dtype=torch.long)
        image_role = torch.ones((1, 1), device=text.device, dtype=torch.long)
        text = text + self.model.embed_image_indicator(text_role)
        image_indicator = self.model.embed_image_indicator(image_role)
        cos, sin = self.model.rotary_emb(position_ids[:, text_start:])
        return Conditioning(
            text=text,
            image_indicator=image_indicator,
            rotary=(cos.to(text.dtype), sin.to(text.dtype)),
            text_tokens=image_start - text_start,
            image_tokens=len(roles) - image_start,
            removed_padding=text_start,
        )

    def enable_compilation(self):
        if self.compiled_blocks is None:
            # Prompt lengths vary after exact padding removal. Dynamic compilation avoids
            # specializing every prompt length; leave CUDA graphs out of this first profile
            # to avoid retained-output aliasing across blocks and sampling iterations.
            self.compiled_blocks = [
                torch.compile(block, fullgraph=True, dynamic=True, mode="max-autotune-no-cudagraphs")
                for block in self.model.layers
            ]

    def forward(self, hidden_states, timestep, condition, *, compiled=False):
        if hidden_states.shape[1] != condition.image_tokens:
            raise ValueError("Image token count changed inside a request")
        # Preserve the reference operation order: projected image, plus zero text
        # condition, then role embedding. Text has zero image projection by construction.
        image = self.model.input_proj(hidden_states) + condition.image_indicator
        hidden = torch.cat((condition.text, image), dim=1)
        t_cond = self.model.t_embedding(timestep)
        if timestep.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln = F.silu(self.model.adaln_proj(t_cond))
        blocks = self.compiled_blocks if compiled else self.model.layers
        if blocks is None:
            raise RuntimeError("Compiled blocks have not been initialized")
        for block in blocks:
            hidden = block(hidden, None, condition.rotary, adaln)
        # final_layer is token-local. The sampler only consumes image velocities.
        return self.model.final_layer(hidden[:, condition.text_tokens :], conditioning=adaln)

    def sample(
        self,
        prompt,
        *,
        width,
        height,
        seed,
        compiled=True,
        attention_backend=None,
        output_type="pil",
        latents=None,
    ):
        pipe = self.pipe
        if attention_backend is not None and attention_backend != self.attention_backend:
            self.model.set_attention_backend(attention_backend)
            self.attention_backend = attention_backend
            self.compiled_blocks = None
            self.seen_shapes.clear()
        pipe.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=8,
            guidance_scale=1.0,
            guidance_schedule=None,
        )
        device = pipe._execution_device
        grid_h = height // (pipe.vae_scale_factor * pipe.patch_size)
        grid_w = width // (pipe.vae_scale_factor * pipe.patch_size)
        encoded = pipe.encode_prompt(
            prompt=prompt, grid_h=grid_h, grid_w=grid_w, max_sequence_length=2048, device=device
        )
        condition = self.prepare_condition(*encoded)
        del encoded  # Release the reference encoder's full padded feature tensor before sampling.
        if compiled:
            self.enable_compilation()
        signature = (width, height, condition.text_tokens, compiled, self.attention_backend)
        first_shape_request = signature not in self.seen_shapes

        # Exactly the released 8-step Diffusers schedule, including its terminal behavior.
        mu = _resolution_aware_mu(height=height, width=width, base_mu=0.0)
        sigmas = _logit_normal_sigmas(8, mu, std=1.75, device=device)
        pipe.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        latents = pipe.prepare_latents(
            batch_size=1,
            num_image_tokens=grid_h * grid_w,
            latent_dim=self.model.config.in_channels,
            dtype=torch.float32,
            device=device,
            generator=torch.Generator(device).manual_seed(seed),
            latents=latents,
        )
        for t in pipe.scheduler.timesteps:
            t_model = (1.0 - t.float() / pipe.scheduler.config.num_train_timesteps).expand(1)
            velocity = self.forward(
                hidden_states=latents.to(self.model.dtype),
                timestep=t_model.to(self.model.dtype),
                condition=condition,
                compiled=compiled,
            ).float()
            latents = pipe.scheduler.step(-velocity, t, latents, return_dict=False)[0]

        if output_type == "latent":
            image = latents
        else:
            mean = pipe.vae.bn.running_mean.view(1, 1, -1).to(device=device, dtype=latents.dtype)
            std = torch.sqrt(pipe.vae.bn.running_var + pipe.vae.config.batch_norm_eps)
            std = std.view(1, 1, -1).to(device=device, dtype=latents.dtype)
            z = latents * std + mean
            patch = pipe.patch_size
            channels = z.shape[-1] // (patch * patch)
            z = z.view(1, grid_h, grid_w, patch, patch, channels)
            z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
            z = z.view(1, channels, grid_h * patch, grid_w * patch)
            decoded = pipe.vae.decode(z.to(pipe.vae.dtype), return_dict=False)[0]
            image = pipe.image_processor.postprocess(decoded.float(), output_type=output_type)[0]
        pipe.maybe_free_model_hooks()
        self.seen_shapes.add(signature)  # Failed requests do not count as prewarmed.
        return image, {
            "profile": "ideogram-native-v1",
            "native_resolution": True,
            "attention_backend": self.attention_backend,
            "compiled_blocks": compiled,
            "compile_mode": "max-autotune-no-cudagraphs" if compiled else None,
            "first_shape_request": first_shape_request,
            "text_tokens": condition.text_tokens,
            "image_tokens": condition.image_tokens,
            "removed_padding_tokens": condition.removed_padding,
            "conditioning_preparations": 1,
            "denoiser_calls": 8,
            "precision": str(self.model.dtype),
            "target_seconds": 4.0,
            "target_status": "requires_gpu_benchmark",
        }


def cuda_self_test():
    """Validate the forced Flash path on the deployment GPU, without downloading weights."""
    import json
    from types import SimpleNamespace

    from diffusers import Ideogram4Pipeline, Ideogram4Transformer2DModel

    if not torch.cuda.is_available():
        raise RuntimeError("Ideogram fast self-test requires CUDA")
    torch.manual_seed(7)
    model = (
        Ideogram4Transformer2DModel(
            in_channels=8,
            num_layers=1,
            attention_head_dim=256,
            num_attention_heads=2,
            intermediate_size=768,
            adaln_dim=32,
            llm_features_dim=12,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    position, segment, roles = Ideogram4Pipeline._prepare_ids([5], 4, 4, 8, torch.device("cuda"))
    features = torch.zeros(1, 24, 12, device="cuda", dtype=torch.bfloat16)
    features[:, 3:8] = torch.randn(1, 5, 12, device="cuda", dtype=torch.bfloat16)
    image = torch.randn(1, 16, 8, device="cuda", dtype=torch.bfloat16)
    packed = torch.cat((torch.zeros(1, 8, 8, device="cuda", dtype=torch.bfloat16), image), dim=1)
    timestep = torch.tensor([0.4], device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        model.set_attention_backend("_native_math")
        reference = model(
            hidden_states=packed,
            timestep=timestep,
            encoder_hidden_states=features,
            position_ids=position,
            segment_ids=segment,
            indicator=roles,
            return_dict=False,
        )[0][:, 8:]
        engine = FastIdeogram(SimpleNamespace(transformer=model))
        condition = engine.prepare_condition(features, position, segment, roles)
        actual = engine.forward(image, timestep, condition)
        torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
        error = (actual.float() - reference.float()).abs().max().item()
    print(
        json.dumps(
            {
                "ideogram_flash_self_test": "passed",
                "max_abs_error": error,
                "gpu": torch.cuda.get_device_name(),
                "head_dim": 256,
            }
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    cuda_self_test()
