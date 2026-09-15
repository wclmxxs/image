import math
import secrets

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=32000)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int = Field(default_factory=lambda: secrets.randbelow(2**31), ge=0, lt=2**31)
    images: list[str] = Field(default_factory=list, max_length=8)
    parameters: dict = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


def validate_request(request, model):
    if request.width % model["multiple"] or request.height % model["multiple"]:
        raise ValueError(f"width and height must be multiples of {model['multiple']}")
    task = "image-edit" if request.images else "text-to-image"
    if task not in model["tasks"]:
        raise ValueError(f"{model['name']} does not support {task}; supported: {model['tasks']}")
    if len(request.images) > model["max_images"]:
        raise ValueError(f"At most {model['max_images']} reference images are supported")
    unknown = set(request.parameters) - set(model["parameters"])
    if unknown:
        raise ValueError(f"Unsupported parameters: {sorted(unknown)}")
    params = {**model["defaults"], **request.parameters}
    for key in ("steps",):
        if key in params and (type(params[key]) is not int or not 1 <= params[key] <= 100):
            raise ValueError("steps must be an integer from 1 to 100")
    if "guidance" in params:
        number = params["guidance"]
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 20:
            raise ValueError("guidance must be a finite number from 0 to 20")
    allowed = {
        "bot_task": {"image", "recaption", "think_recaption"},
        "preset": {"V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"},
        "prompt_mode": {"template", "json", "magic"} if model["backend"] == "ideogram" else {"text", "json"},
    }
    for key, values in allowed.items():
        if key in params and (not isinstance(params[key], str) or params[key] not in values):
            raise ValueError(f"{key} must be one of {sorted(values)}")
    return {**request.model_dump(), "model": model["id"], "parameters": params, "task": task}
