FROM image-lab/base-cu128:0.1.0
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir einops==0.8.1 numpy==2.2.0 pillow==12.0.0 diffusers==0.35.2 safetensors==0.7.0 \
    tokenizers==0.22.0 "transformers[accelerate,tiktoken]==4.57.1" huggingface-hub==0.36.2 loguru==0.7.3 \
    flashinfer-python==0.5.0
RUN git init /opt/hunyuan && cd /opt/hunyuan \
    && git remote add origin https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git \
    && git fetch --depth 1 origin 6e9113a692a27a0751d82aba3b2015a876646c03 \
    && git checkout --detach FETCH_HEAD && pip check
ENV PYTHONPATH=/app:/opt/hunyuan
COPY image_lab /app/image_lab
RUN python3 -c "from hunyuan_image_3 import HunyuanImage3ForCausalMM"
