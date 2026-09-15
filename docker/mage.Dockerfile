FROM image-lab/base-cu126:0.1.0
ENV MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=9.0
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir diffusers==0.38.0 transformers==5.5.0 accelerate==1.13.0 \
    safetensors==0.8.0 einops==0.8.2 pydantic==2.12.5 pillow==12.3.0 numpy==2.4.3 loguru==0.7.3 \
    "git+https://github.com/microsoft/Mage.git@76bec2bb3818863f470de7e867c2dc7f1d0bfd83#subdirectory=mage_flow"
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3 && pip check
COPY image_lab /app/image_lab
RUN python3 -c "from mage_flow import MageFlowPipeline"
