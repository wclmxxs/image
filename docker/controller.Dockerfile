FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml ./
COPY docker/requirements-controller.txt ./requirements-controller.txt
COPY image_lab ./image_lab
COPY config ./config
RUN pip install --no-cache-dir -r requirements-controller.txt . && pip check
CMD ["uvicorn", "image_lab.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
