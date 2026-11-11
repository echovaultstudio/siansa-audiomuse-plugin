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
 && pip install numpy==1.26.4 einops==0.8.0 rotary-embedding-torch==0.6.4 soxr==0.3.7 tqdm==4.66.4 \
# cryptography: what Siansa Beats has to say to other servers is kept encrypted (beats/vault.py, 0.24.0).
 && pip install cryptography==43.0.3

# Beat This! itself, at a fixed commit (version 1.1.0, 28 May 2026).
ARG BEAT_THIS_COMMIT=b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c
# Fetched first with retries: GitHub's archives can answer 504 for a while, which pip doesn't retry.
RUN for try in 1 2 3 4 5 6; do \
      python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
        "https://github.com/CPJKU/beat_this/archive/${BEAT_THIS_COMMIT}.tar.gz" /tmp/beat_this.tar.gz \
        && break || { echo "download $try failed, trying again"; sleep 20; }; \
    done \
 && pip install --no-deps /tmp/beat_this.tar.gz && rm /tmp/beat_this.tar.gz

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

# The style model (each song's genre from its sound): Essentia's Discogs-EffNet (MTG, UPF; CC BY-NC-ND 4.0), as ONNX
# with any batch size, and its metadata (the 400 Discogs styles in its order). It runs on the CPU with ONNX Runtime.
# On a GPU it runs through PyTorch instead, converted with onnx2torch (checked below against ONNX Runtime).
# onnx2torch without its dependencies, so torch stays at Beat This!'s version; torchvision (its converters import it)
# from the same index, matching torch 2.3.1.
RUN pip install onnxruntime==1.19.2 onnx==1.17.0 \
 && pip install --index-url "$TORCH_INDEX" torchvision==0.18.1 \
 && pip install --no-deps onnx2torch==1.5.15 \
 && python -c "import torch, torchaudio, torchvision, onnx2torch; assert torch.__version__.startswith('2.3.1'), torch.__version__; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
ARG ESSENTIA_MODELS=https://essentia.upf.edu/models/feature-extractors/discogs-effnet
RUN for file in discogs-effnet-bsdynamic-1.onnx discogs-effnet-bs64-1.json; do \
      for try in 1 2 3 4 5; do \
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
          "${ESSENTIA_MODELS}/${file}" "/models/${file}" && break || { echo "download $try failed, trying again"; sleep 15; }; \
      done; \
    done \
 && python -c "import json, numpy, onnxruntime; \
s = onnxruntime.InferenceSession('/models/discogs-effnet-bsdynamic-1.onnx', providers=['CPUExecutionProvider']); \
print('inputs', [(i.name, i.shape) for i in s.get_inputs()], 'outputs', [(o.name, o.shape) for o in s.get_outputs()]); \
out = s.run(None, {s.get_inputs()[0].name: numpy.zeros((2, 128, 96), numpy.float32)}); \
labels = json.load(open('/models/discogs-effnet-bs64-1.json'))['classes']; \
assert any(o.shape == (2, len(labels)) for o in out), [o.shape for o in out]; print(len(labels), 'styles')" \
 && python -c "import numpy, torch, onnxruntime; from onnx2torch import convert; \
path = '/models/discogs-effnet-bsdynamic-1.onnx'; x = numpy.random.RandomState(0).rand(4, 128, 96).astype(numpy.float32) * 4; \
s = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider']); ref = s.run(None, {s.get_inputs()[0].name: x}); \
m = convert(path).eval(); out = m(torch.from_numpy(x)); out = out if isinstance(out, (list, tuple)) else (out,); \
diff = max(float(numpy.abs(a - b.detach().numpy()).max()) for a, b in zip(ref, out)); \
assert diff < 1e-3, diff; print('onnx2torch matches ONNX Runtime within', diff)"

WORKDIR /app
COPY beats/ /app/beats/
RUN python -c "import torch, beat_this.inference, beats.server; print('imports ok, torch', torch.__version__, 'cuda', torch.version.cuda)" \
 && python -c "import numpy, beats.style as s; m = s.StyleModel('/models'); assert m.available; \
r = m(numpy.zeros(16000 * 5, numpy.float32)); assert len(r) == 400 and 0 <= r[0][1] <= 1, r[:3]; print('style model ok:', r[0])"

RUN useradd --uid 1000 --create-home beats && mkdir -p /data && chown beats /data
USER beats
# As the service runs it: the model converted for the GPU, from read-only /models (0.17.0 wrote a temporary file there).
RUN python -c "import onnx; from onnx2torch import convert; \
convert(onnx.load('/models/discogs-effnet-bsdynamic-1.onnx')); print('converts as the service user')"
# NVIDIA's container toolkit gives a GPU to containers that ask with these (harmless without one).
ENV DATA_DIR=/data MODELS_DIR=/models PORT=8090 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME /data
EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/api/health', timeout=4)"

CMD ["python", "-m", "beats.server"]
