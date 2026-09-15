ARG COSMOS_IMAGE=vllm/vllm-omni@sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587
FROM ${COSMOS_IMAGE}
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
WORKDIR /app
COPY image_lab /app/image_lab
# The upstream release container supplies torch, Pillow, vLLM and its tested dependency set.
RUN pip install --no-cache-dir cosmos-guardrail==0.3.1 \
    && python3 -c "import torch; import PIL; import vllm; import cosmos_guardrail"
