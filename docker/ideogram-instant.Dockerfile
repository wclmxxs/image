FROM image-lab/base-cu126:0.1.0
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
# Isolated from the native original Ideogram runtime. Its package supplies only
# prompt formatting / optional magic prompt and moderation helpers here.
RUN pip install --no-cache-dir diffusers==0.39.0 transformers==5.5.0 accelerate==1.13.0 \
    bitsandbytes==0.49.2 pillow==12.3.0 sentencepiece==0.2.1 \
    "git+https://github.com/ideogram-oss/ideogram4.git@990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2" && pip check
COPY image_lab /app/image_lab
RUN python3 -c "from diffusers import Ideogram4Pipeline, Ideogram4Transformer2DModel; from ideogram4 import MAGIC_PROMPTS"
