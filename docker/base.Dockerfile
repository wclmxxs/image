# nvidia/cuda:12.6.3-devel-ubuntu24.04, resolved 2026-09-15.
ARG CUDA_IMAGE=nvidia/cuda@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a
FROM ${CUDA_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev git ca-certificates build-essential ninja-build libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" PYTHONPATH=/app
RUN pip install --no-cache-dir pip==26.0.1 setuptools==80.9.0 wheel==0.46.3 packaging==26.0 ninja==1.13.0
WORKDIR /app
