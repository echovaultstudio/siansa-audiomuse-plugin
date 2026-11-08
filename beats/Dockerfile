# Siansa Beats: Beat This! (CPJKU, 2024) for a test library from Navidrome, with a page to watch and compare it.
#
# Two builds from this file: the CPU one (the default) and, with --build-arg TORCH_INDEX=…/whl/cu121, the cuda one
# for an NVIDIA GPU (it also runs without one, on the CPU). The model's checkpoints are downloaded while the image is
# built and kept in /models, so the service itself needs no internet: only your Navidrome (and AudioMuse-AI).
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/echovaultstudio/siansa-audiomuse-plugin" \
      org.opencontainers.image.description="Siansa Beats: Beat This! beat and downbeat tracking for a Navidrome playlist, with a page to compare it" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Beat This!'s own "known working versions for inference" (its requirements.txt): torch for the CPU, or for CUDA 12.1
# (NVIDIA driver 525 or newer on the host).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# Downloads retry with a longer timeout: GitHub and the mirrors can be slow to answer from CI.
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install --index-url "$TORCH_INDEX" torch==2.3.1 torchaudio==2.3.1 \
 && pip install numpy==1.26.4 einops==0.8.0 rotary-embedding-torch==0.6.4 soxr==0.3.7 tqdm==4.66.4

# Beat This! itself, at a fixed commit (version 1.1.0, 28 May 2026).
ARG BEAT_THIS_COMMIT=b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c
RUN pip install --no-deps "https://github.com/CPJKU/beat_this/archive/${BEAT_THIS_COMMIT}.tar.gz"

# The checkpoints: final0 (the full model, the default) and small0 (smaller and faster, a little less exact).
ARG CHECKPOINTS="final0 small0"
RUN mkdir -p /models && for name in $CHECKPOINTS; do \
      for try in 1 2 3 4 5; do \
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
          "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/${name}.ckpt" "/models/${name}.ckpt" \
          && break || { echo "download $try failed, trying again"; sleep 15; }; \
      done \
      && python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu'); print(sys.argv[1], 'loads')" \
        "/models/${name}.ckpt"; \
    done

WORKDIR /app
COPY beats/ /app/beats/
RUN python -c "import torch, beat_this.inference, beats.server; print('imports ok, torch', torch.__version__, 'cuda', torch.version.cuda)"

RUN useradd --uid 1000 --create-home beats && mkdir -p /data && chown beats /data
USER beats
# NVIDIA's container toolkit gives a GPU to containers that ask with these (harmless without one).
ENV DATA_DIR=/data MODELS_DIR=/models PORT=8090 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME /data
EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/api/health', timeout=4)"

CMD ["python", "-m", "beats.server"]
