# Wan2.1-T2V-1.3B Video Generation Worker for RunPod Serverless
# Diffusers-native (unlike LTX-2.3, which turned out not to be a simple
# from_pretrained model at all - see runpod-ltx2-worker's history). Small
# enough (1.3B params, ~8GB VRAM per Wan's own docs) to run comfortably on
# a 16GB tier instead of forcing a much pricier GPU tier.

FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Requires a RunPod Network Volume attached at /runpod-volume on the
# endpoint (see README) - the model (~29GB, mostly the UMT5 text encoder
# shared across every Wan size) is downloaded here on the first cold
# start and reused by every worker after that. Not baked into the image
# itself: at this size it would risk exceeding CI runner disk space, and
# a volume-backed cache is the standard RunPod pattern for large models
# anyway.
ENV HF_HOME=/runpod-volume/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY requirements-worker.txt /app/requirements-worker.txt
RUN pip install --no-cache-dir -r requirements-worker.txt

RUN pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
