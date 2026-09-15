FROM image-lab/base-cu126:0.1.0
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir transformers==5.5.0 accelerate==1.13.0 peft==0.20.0 sentencepiece==0.2.1 pillow==12.3.0 einops==0.8.2 \
    "git+https://github.com/huggingface/diffusers.git@759164b7ad116e091e9d3e222211c9aa27d835f6" && pip check
COPY image_lab /app/image_lab
RUN python3 -c "from diffusers import Flux2Pipeline; import peft"
