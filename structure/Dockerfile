# Siansa Structure: All-In-One (github.com/mir-aidj/all-in-one, MIT) in its own container, for Siansa Beats to
# ask where a song's sections start. Kept apart from Siansa Beats' image: its parts (NATTEN 0.17, madmom from
# source, Demucs) are fussy, and if they misbehave Beat This! carries on.
#
#   docker build -t siansa-structure structure                                   # CPU
#   docker build -t siansa-structure:cuda --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
#     --build-arg NATTEN_BUILD=torch230cu121 structure                            # NVIDIA GPU
FROM python:3.10-slim-bookworm
LABEL org.opencontainers.image.source="https://github.com/echovaultstudio/siansa-audiomuse-plugin" \
      org.opencontainers.image.description="Siansa Structure: All-In-One's sections of a song, for Siansa Beats" \
      org.opencontainers.image.licenses="MIT"

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG NATTEN_BUILD=torch230cpu
# madmom from GitHub (the release on PyPI doesn't build on today's numpy), pinned.
ARG MADMOM_COMMIT=27f032e8947204902c675e5e341a3faf5dc86dae

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 \
    HF_HOME=/models/hf TORCH_HOME=/models/torch MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg build-essential git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# The versions stay put: every later install is held to them, so Demucs or All-In-One can't pull another torch.
# python:3.10's pip is too old for PyTorch's index (it rejects "Jinja2" asked for as "jinja2").
RUN pip install --upgrade pip==24.2
RUN printf 'torch==2.3.1\ntorchaudio==2.3.1\nnumpy==1.26.4\n' > /constraints.txt
RUN pip install --index-url "$TORCH_INDEX" torch==2.3.1 torchaudio==2.3.1 \
 && pip install -c /constraints.txt numpy==1.26.4 cython==3.0.11 scipy==1.13.1 mido==1.3.2
RUN pip install -c /constraints.txt "natten==0.17.1+${NATTEN_BUILD}" -f https://shi-labs.com/natten/wheels/
RUN pip install -c /constraints.txt --no-build-isolation "git+https://github.com/CPJKU/madmom@${MADMOM_COMMIT}"
RUN pip install -c /constraints.txt demucs==4.0.1 soundfile==0.12.1 librosa==0.10.2.post1 matplotlib==3.9.2 \
      huggingface-hub==0.25.2 hydra-core==1.3.2 omegaconf==2.3.0 tqdm \
 && pip install -c /constraints.txt --no-deps allin1==1.1.0

# The models, baked in: All-In-One's eight (Hugging Face) and Demucs' htdemucs. Nothing is downloaded at run time.
RUN python -c "from allin1.models import load_pretrained_model; load_pretrained_model('harmonix-all', device='cpu'); print('All-In-One loads')" \
 && python -c "from demucs.pretrained import get_model; get_model('htdemucs'); print('htdemucs loads')"
ENV HF_HUB_OFFLINE=1

COPY server.py /app/server.py
RUN python -c "import torch, natten, madmom, demucs, allin1; print('imports ok, torch', torch.__version__, 'cuda', torch.version.cuda)"

EXPOSE 8091
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8091/health', timeout=4)"
CMD ["python", "/app/server.py"]
