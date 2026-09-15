import asyncio
import base64
import hmac
import io
import re
import time
import uuid
import warnings
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from image_lab.config import Registry, Settings
from image_lab.runtime import DockerRuntime
from image_lab.scheduler import Scheduler
from image_lab.schemas import JobRequest, validate_request
from image_lab.store import TERMINAL, JobStore

IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")
MAX_UPLOAD = 25 * 1024 * 1024


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    prompt: str
    size: str = "1024x1024"
    n: int = Field(default=1, ge=1, le=1)
    seed: int | None = None
    response_format: str = "b64_json"
    parameters: dict = Field(default_factory=dict)


def create_app(settings=None, runtime_factory=DockerRuntime):
    settings = settings or Settings.from_env()
    if len(settings.api_key) < 24:
        raise ValueError("API_KEY must contain at least 24 characters")
    settings.root.mkdir(parents=True, exist_ok=True)
    registry = Registry(settings)
    store = JobStore(settings.root / "jobs.sqlite3")
    runtime = runtime_factory(settings, registry)
    scheduler = Scheduler(settings, registry, store, runtime)

    @asynccontextmanager
    async def lifespan(app):
        scheduler.start()
        try:
            yield
        finally:
            scheduler.close()
            store.close()

    app = FastAPI(title="Image Model Lab", version="0.1.0", lifespan=lifespan)
    app.state.scheduler, app.state.store = scheduler, store
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not credentials or not hmac.compare_digest(credentials.credentials, settings.api_key):
            raise HTTPException(401, "Bearer API key required", headers={"WWW-Authenticate": "Bearer"})

    def find_job(job_id):
        if not IDENTIFIER.fullmatch(job_id):
            raise HTTPException(404, "Job not found")
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    def submit(request):
        try:
            model = registry.resolve(request.model)
            payload = validate_request(request, model)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        for upload_id in request.images:
            if (
                not IDENTIFIER.fullmatch(upload_id)
                or not (settings.root / "uploads" / f"{upload_id}.png").is_file()
            ):
                raise HTTPException(422, f"Unknown upload ID: {upload_id[:64]}")
        if registry.prepared(model) is None:
            raise HTTPException(
                409, model.get("blocked_reason", "Model is not prepared. Run start.sh --models MODEL")
            )
        if not scheduler.thread.is_alive():
            raise HTTPException(503, "Scheduler is not running")
        try:
            return store.add(payload, settings.max_queue)
        except OverflowError as error:
            raise HTTPException(429, str(error)) from error

    @app.get("/health")
    def health():
        if not scheduler.thread.is_alive():
            raise HTTPException(503, "Scheduler stopped")
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        return {
            "data": registry.public(),
            "active_model": runtime.active_model,
            "current_job": scheduler.current_job,
            "scheduling": "serial",
        }

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
    def create_job(request: JobRequest):
        return submit(request)

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def get_job(job_id: str):
        return find_job(job_id)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
    def cancel(job_id: str):
        find_job(job_id)
        return store.cancel(job_id)

    @app.get("/v1/jobs/{job_id}/image", dependencies=[Depends(authorize)])
    def get_image(job_id: str):
        job = find_job(job_id)
        if job["status"] != "succeeded":
            raise HTTPException(409, "Job has not succeeded")
        return FileResponse(settings.root / "jobs" / job_id / "image.png", media_type="image/png")

    @app.post("/v1/uploads", status_code=201, dependencies=[Depends(authorize)])
    async def upload(file: UploadFile):
        try:
            data = await file.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "Maximum upload size is 25 MiB")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.width * source.height > 24_000_000:
                        raise HTTPException(413, "Maximum reference image size is 24 megapixels")
                    result = ImageOps.exif_transpose(source).convert("RGB")
                    result.load()
            upload_id = uuid.uuid4().hex
            path = settings.root / "uploads" / f"{upload_id}.png"
            path.parent.mkdir(exist_ok=True)
            await asyncio.to_thread(result.save, path, format="PNG")
            return {"id": upload_id, "width": result.width, "height": result.height}
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as error:
            raise HTTPException(422, "Invalid or oversized image") from error
        finally:
            await file.close()

    @app.post("/v1/images/generations", dependencies=[Depends(authorize)])
    async def generations(request: GenerationRequest):
        if request.response_format not in {"b64_json", "url"}:
            raise HTTPException(422, "response_format must be b64_json or url")
        if not re.fullmatch(r"\d{3,4}x\d{3,4}", request.size):
            raise HTTPException(422, "size must be WIDTHxHEIGHT")
        width, height = map(int, request.size.split("x"))
        fields = {
            "model": request.model,
            "prompt": request.prompt,
            "width": width,
            "height": height,
            "parameters": request.parameters,
        }
        if request.seed is not None:
            fields["seed"] = request.seed
        try:
            payload = JobRequest(**fields)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        job = submit(payload)
        deadline = time.monotonic() + 55
        while time.monotonic() < deadline:
            current = store.get(job["id"])
            if current["status"] in TERMINAL:
                if current["status"] != "succeeded":
                    raise HTTPException(
                        500, {"job_id": job["id"], "error": current.get("error", current["status"])}
                    )
                output = {"url": current["image_url"]}
                if request.response_format == "b64_json":
                    image = (settings.root / "jobs" / job["id"] / "image.png").read_bytes()
                    output = {"b64_json": base64.b64encode(image).decode()}
                return {"created": int(time.time()), "data": [output], "job_id": job["id"]}
            await asyncio.sleep(0.25)
        # Cold starts are long: preserve the job and provide an explicit polling handle.
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=202,
            content={"job_id": job["id"], "status": "pending", "poll_url": f"/v1/jobs/{job['id']}"},
        )

    return app
