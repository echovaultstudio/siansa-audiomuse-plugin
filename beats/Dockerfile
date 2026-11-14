# Siansa Beats: Beat This! (CPJKU, 2024) for a test library from Navidrome, with a page to watch and compare it.
#
# Two builds from this file: the CPU one (the default) and, with --build-arg TORCH_INDEX=…/whl/cu121, the cuda one
# for an NVIDIA GPU (it also runs without one, on the CPU). The model's checkpoints are downloaded while the image is
# built and kept in /models, so the service itself needs no internet: only your music server.
# How energetic a song is (0.35): Essentia's emoMusic arousal model (MTG, UPF; CC BY-NC-SA 4.0), published as a
# TensorFlow file, made into ONNX here. TensorFlow is in this stage only: the image itself gets the one small file.
FROM python:3.15-slim-bookworm AS arousal
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install tensorflow-cpu==2.15.1 tf2onnx==1.16.1 "numpy<2"
COPY arousal_model.py /tmp/arousal_model.py
RUN python /tmp/arousal_model.py /emomusic-msd-musicnn-2.onnx && ls -la /emomusic-msd-musicnn-2.onnx

FROM python:3.15-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/echovaultstudio/siansa-beats" \
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
# pillow: an animated cover's first frame, for the listening page (beats/art.py, 0.34.1).
 && pip install cryptography==43.0.3 tzdata==2025.2 pillow==11.3.0

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
# They're asked of this repository's own release first (a copy kept by .github/workflows/models.yml), then of the
# authors' host: a build stopped whenever that host didn't answer (the owner, 14 Nov). Either way the file must be
# the one expected (its SHA-256), and must load.
ARG CHECKPOINTS="final0 small0"
ARG CHECKPOINT_SHA256="final0=8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331 small0=6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b"
RUN mkdir -p /models && for name in $CHECKPOINTS; do \
      want=""; for pair in $CHECKPOINT_SHA256; do [ "${pair%%=*}" = "$name" ] && want="${pair#*=}"; done; \
      ok=""; \
      for from in "https://github.com/echovaultstudio/siansa-beats/releases/download/models-1" \
                  "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"; do \
        for try in 1 2 3; do \
          python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
            "$from/${name}.ckpt" "/models/${name}.ckpt" \
            && { [ -z "$want" ] || echo "$want  /models/${name}.ckpt" | sha256sum -c - ; } \
            && { ok=yes; break; } || { echo "$name from $from: try $try failed"; rm -f "/models/${name}.ckpt"; sleep 10; }; \
        done; \
        [ -n "$ok" ] && break; \
      done; \
      [ -n "$ok" ] || { echo "$name.ckpt couldn't be downloaded from either place"; exit 1; }; \
      python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu'); print(sys.argv[1], 'loads')" \
        "/models/${name}.ckpt" || exit 1; \
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

# MAEST (0.39): the same group's 2023 transformer, the same 400 styles, markedly more accurate; the one to choose in
# Setup beyond electronic music. Essentia's ONNX (350 MB; CC BY-NC-SA 4.0, as EffNet), with its labels. Its scores
# are the output within 0 and 1 (its logits come too); checked here through onnx2torch for the GPU, as EffNet is.
ARG MAEST_MODELS=https://essentia.upf.edu/models/feature-extractors/maest
RUN for file in discogs-maest-30s-pw-1.onnx discogs-maest-30s-pw-1.json; do \
      for try in 1 2 3 4 5; do \
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
          "${MAEST_MODELS}/${file}" "/models/${file}" && break || { echo "download $try failed, trying again"; sleep 15; }; \
      done; \
    done \
 && python -c "import json, numpy, onnxruntime, torch; from onnx2torch import convert; \
path = '/models/discogs-maest-30s-pw-1.onnx'; x = numpy.random.RandomState(0).rand(1, 1876, 96).astype(numpy.float32) * 4 - 2; \
s = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider']); ref = s.run(['activations'], {'melspectrogram': x})[0]; \
labels = json.load(open('/models/discogs-maest-30s-pw-1.json'))['classes']; effnet = json.load(open('/models/discogs-effnet-bs64-1.json'))['classes']; \
assert labels == effnet, 'MAEST and EffNet name different styles'; assert ref.shape == (1, 400), ref.shape; \
m = convert(path).eval(); out = m(torch.from_numpy(x)); out = out if isinstance(out, (list, tuple)) else (out,); \
act = next(o for o in out if tuple(o.shape) == (1, 400) and float(o.min()) >= 0 and float(o.max()) <= 1); \
diff = float(numpy.abs(ref - act.detach().numpy()).max()); assert diff < 1e-3, diff; print('MAEST: 400 styles, onnx2torch matches ONNX Runtime within', diff)"

# The search by words (0.26): AudioMuse-AI's own models, so a description lands among the songs as it does there.
# LAION CLAP's text tower as ONNX (500 MB) and the two small models its concept steering uses (a sparse autoencoder
# over CLAP's space), from AudioMuse-AI's releases; RoBERTa's tokenizer is in the repository (beats/roberta-tokenizer.json).
ARG AUDIOMUSE_MODELS=https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v5.0.0-model
ARG AUDIOMUSE_SAE=https://github.com/NeptuneHub/AudioMuse-AI-SAE/releases/download/v1
RUN pip install tokenizers==0.20.3 \
 && for url in "${AUDIOMUSE_MODELS}/clap_text_model.onnx" "${AUDIOMUSE_SAE}/dclap_sae_k20_d1024_best_encoder.onnx" \
               "${AUDIOMUSE_SAE}/dclap_sae_k20_d1024_best_decoder.onnx"; do \
      for try in 1 2 3 4 5; do \
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
          "$url" "/models/$(basename "$url")" && break || { echo "download $try failed, trying again"; sleep 15; }; \
      done; \
      test -s "/models/$(basename "$url")"; \
    done

# How a song sounds, heard here (0.27): AudioMuse-AI's own models and its own recipe, so a new song needs no
# AudioMuse-AI and lands where AudioMuse-AI would have put it. MusiCNN (its embedding and its 50 tags) and the distilled
# CLAP audio model (AudioMuse-AI's DCLAP, AGPL-3.0), from AudioMuse-AI's releases; librosa makes the spectrograms, as
# in AudioMuse-AI, so they're the same to the last digit (numba's wheels go with numpy 1.26).
ARG AUDIOMUSE_DCLAP=https://github.com/NeptuneHub/AudioMuse-AI-DCLAP/releases/download/v1
# Everything librosa brings is pinned to versions that go with numpy 1.26.4 (0.27's first build let pip take the
# newest, which brought numpy 2 and broke soxr).
RUN pip install numpy==1.26.4 librosa==0.10.2.post1 soundfile==0.12.1 scipy==1.13.1 scikit-learn==1.5.2 \
      numba==0.60.0 llvmlite==0.43.0 soxr==0.3.7 \
 && python -c "import numpy, librosa, soxr; assert numpy.__version__ == '1.26.4', numpy.__version__; print('librosa', librosa.__version__)" \
 && for url in "${AUDIOMUSE_MODELS}/musicnn_embedding.onnx" "${AUDIOMUSE_MODELS}/musicnn_prediction.onnx" \
               "${AUDIOMUSE_DCLAP}/model_epoch_36.onnx" "${AUDIOMUSE_DCLAP}/model_epoch_36.onnx.data"; do \
      for try in 1 2 3 4 5; do \
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
          "$url" "/models/$(basename "$url")" && break || { echo "download $try failed, trying again"; sleep 15; }; \
      done; \
      test -s "/models/$(basename "$url")"; \
    done

WORKDIR /app
COPY beats/ /app/beats/
# A song heard for real: a made-up one, read through librosa and through ffmpeg, twice the same.
RUN ffmpeg -v error -f lavfi -i "sine=frequency=220:duration=14,aformat=channel_layouts=stereo" -f lavfi -i "anoisesrc=d=14:c=pink:a=0.2" \
      -filter_complex amix=inputs=2 -ar 44100 /tmp/heard.flac \
 && python -c "import numpy; from beats.hear import Hearer; from beats.words import Words; \
h = Hearer('/models', Words('/models')); assert h.available; a = h('/tmp/heard.flac'); b = h('/tmp/heard.flac'); \
assert len(a['musicnn']) == 800 and len(a['clap']) == 2048, (len(a['musicnn']), len(a['clap'])); \
assert a['musicnn'] == b['musicnn'] and a['clap'] == b['clap']; \
c = numpy.frombuffer(a['clap'], numpy.float32); assert abs(float(numpy.linalg.norm(c)) - 1) < 1e-4; \
assert a['moods'].count(':') == 5 and a['other'].startswith('danceable:0.') and 0 < a['energy'] < 1, a; \
print('heard:', a['moods'], '|', a['other'], '| energy', round(a['energy'], 3), '| models loaded in', h.load_seconds, 's')" \
 && rm /tmp/heard.flac
# The text model for real: words that mean alike land together, and steering moves a query towards its concept.
RUN python -c "import numpy; from beats.words import Words; w = Words('/models'); assert w.available and w.steers; \
a, b, c = w.embed('calm quiet soft piano'), w.embed('gentle peaceful piano music'), w.embed('aggressive heavy metal'); \
assert abs(float(numpy.linalg.norm(a)) - 1) < 1e-4 and a.shape == (512,), a.shape; \
assert float(a @ b) > float(a @ c) + 0.2, (float(a @ b), float(a @ c)); \
terms, problems = w.terms([{'term': 'piano', 'direction': 'more'}]); assert terms and not problems, problems; \
q, p = w.embed('electronic music'), w.embed('solo piano'); s = w.steer(q, terms); \
assert float(s @ p) > float(q @ p), (float(s @ p), float(q @ p)); \
print('text model ok: alike', round(float(a @ b), 3), 'unlike', round(float(a @ c), 3), 'steered towards piano', round(float(s @ p) - float(q @ p), 3), 'in', w.load_seconds, 's')"
RUN python -c "import torch, beat_this.inference, beats.server; print('imports ok, torch', torch.__version__, 'cuda', torch.version.cuda)" \
 && python -c "import numpy, beats.style as s; m = s.StyleModel('/models'); assert m.available; \
r = m(numpy.zeros(16000 * 5, numpy.float32)); assert len(r) == 400 and 0 <= r[0][1] <= 1, r[:3]; print('style model ok:', r[0])"

# The arousal model for real: any number of songs at once, the same answer each time, each about 1 to 9.
COPY --from=arousal /emomusic-msd-musicnn-2.onnx /models/emomusic-msd-musicnn-2.onnx
RUN python -c "import numpy, tempfile, os; from beats.store import Store; from beats.drive import Arousal; \
a = Arousal('/models', Store(os.path.join(tempfile.mkdtemp(), 'beats.db'))); assert a.available; \
v = numpy.random.default_rng(1).normal(0, 1, (5, 200)).astype(numpy.float32); one, again, single = a.score(v), a.score(v), a.score(v[:1]); \
assert one.shape == (5,) and numpy.allclose(one, again) and abs(float(single[0]) - float(one[0])) < 1e-4, (one, single); \
assert numpy.all(numpy.isfinite(one)), one; print('arousal model ok:', numpy.round(one, 3), os.path.getsize('/models/emomusic-msd-musicnn-2.onnx'), 'bytes')"

RUN useradd --uid 1000 --create-home beats && mkdir -p /data && chown beats /data
USER beats
# As the service runs it: the model converted for the GPU, from read-only /models (0.17.0 wrote a temporary file there).
RUN python -c "import onnx; from onnx2torch import convert; \
convert(onnx.load('/models/discogs-effnet-bsdynamic-1.onnx')); print('converts as the service user')"
# NVIDIA's container toolkit gives a GPU to containers that ask with these (harmless without one).
# MALLOC_ARENA_MAX: two pools of memory for the C library, not one a thread: freed memory goes back to the system
# instead of staying with whichever thread had it (beats/memory.py, 0.34).
ENV DATA_DIR=/data MODELS_DIR=/models PORT=8090 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MALLOC_ARENA_MAX=2
VOLUME /data
EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/api/health', timeout=4)"

CMD ["python", "-m", "beats.server"]
