"""Each song's style from its sound: Essentia's Discogs-EffNet (MTG, UPF), 400 Discogs styles.

The model is an EfficientNet-B0 trained by the Music Technology Group on over two million recordings labelled with the
Discogs styles of their releases (``Electronic---Progressive Trance``, ``Rock---Punk``…). It hears 2-second patches of
a mel spectrogram and gives each style a score from 0 to 1; a song's scores are the mean over its patches, as
Essentia's own examples do. The model (``discogs-effnet-bsdynamic-1.onnx``, about 18 MB, downloaded while the image is
built) runs on the GPU when there is one, through the PyTorch Beat This! already uses (converted from ONNX with
onnx2torch; the image build checks it gives ONNX Runtime's scores), and on the CPU with ONNX Runtime otherwise: the
model is small, so it barely slows Beat This! down.

The mel spectrogram is Essentia's ``TensorflowInputMusiCNN``, reproduced here in numpy (checked against Essentia to
within 0.0001): 16 kHz mono, 512-sample frames every 256 samples, a symmetric Hann window, the power spectrum through
96 Slaney mel bands (unit-area triangles, 0 to 8 kHz), then ``log10(1 + 10000 x)``. Patches of 128 frames every 62
frames, as ``TensorflowPredictEffnetDiscogs``; a long song is read at :data:`MAX_PATCHES` patches spread evenly
through it, which gives the same mean within a few hundredths and keeps a song to about a second on the CPU.
"""

import json
import os
import threading
import time

RATE = 16_000
FRAME = 512
HOP = 256
BANDS = 96
PATCH = 128
PATCH_HOP = 62
MAX_PATCHES = 64
MODEL_FILE = "discogs-effnet-bsdynamic-1.onnx"
LABELS_FILE = "discogs-effnet-bs64-1.json"
KEEP = 15  # the top styles kept for each song (with their scores), for the vote and the page


def _hz_to_mel(f):
    import numpy as np

    f = np.asarray(f, dtype=np.float64)
    f_sp, min_hz = 200.0 / 3, 1000.0
    step = np.log(6.4) / 27.0
    return np.where(f >= min_hz, min_hz / f_sp + np.log(np.maximum(f, 1e-10) / min_hz) / step, f / f_sp)


def _mel_to_hz(m):
    import numpy as np

    m = np.asarray(m, dtype=np.float64)
    f_sp, min_hz = 200.0 / 3, 1000.0
    step = np.log(6.4) / 27.0
    return np.where(m >= min_hz / f_sp, min_hz * np.exp(step * (m - min_hz / f_sp)), f_sp * m)


_FILTERS = None


def _filters():
    """96 Slaney mel triangles over the 257 bins of a 512-sample frame at 16 kHz, each of unit area."""
    global _FILTERS
    if _FILTERS is None:
        import numpy as np

        freqs = np.linspace(0, RATE / 2, FRAME // 2 + 1)
        edges = _mel_to_hz(np.linspace(_hz_to_mel(0), _hz_to_mel(RATE / 2), BANDS + 2))
        bank = np.zeros((BANDS, FRAME // 2 + 1))
        for i in range(BANDS):
            lo, mid, hi = edges[i:i + 3]
            bank[i] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid))) * 2.0 / (hi - lo)
        _FILTERS = bank.T.astype(np.float32)
    return _FILTERS


def mel(samples):
    """The log mel bands of 16 kHz mono ``samples``: ``(frames, 96)`` float32, as ``TensorflowInputMusiCNN``."""
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < FRAME:
        samples = np.pad(samples, (0, FRAME - samples.size))
    count = 1 + (samples.size - FRAME) // HOP
    index = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]
    frames = samples[index] * np.hanning(FRAME).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log10(1.0 + 10000.0 * (power.astype(np.float32) @ _filters())).astype(np.float32)


def patches(bands, most=MAX_PATCHES):
    """``(n, 128, 96)``: the patches the model hears, every :data:`PATCH_HOP` frames, at most ``most`` of them spread
    evenly through the song; a song shorter than one patch is padded with silence."""
    import numpy as np

    if len(bands) < PATCH:
        bands = np.pad(bands, ((0, PATCH - len(bands)), (0, 0)))
    starts = list(range(0, len(bands) - PATCH + 1, PATCH_HOP))
    if len(starts) > most:
        starts = [starts[round(i * (len(starts) - 1) / (most - 1))] for i in range(most)]
    return np.stack([bands[s:s + PATCH] for s in starts])


def labels(models_dir):
    """The 400 style names in the model's order (``Electronic---Trance``), from its metadata file."""
    with open(os.path.join(models_dir, LABELS_FILE), encoding="utf-8") as f:
        return list(json.load(f)["classes"])


def split(label):
    """``("Electronic", "Trance")`` from ``"Electronic---Trance"``."""
    genre, _, style = label.partition("---")
    return genre, style or genre


class StyleModel:
    """Discogs-EffNet, loaded with the first song. Thread-safe; ``available`` is False when the image has no model
    (an older image, or a test), and then nothing is asked of it."""

    def __init__(self, models_dir, threads=2, device="auto"):
        self.models_dir = models_dir
        self.threads = threads
        self.requested = device  # BEATS_DEVICE: auto, cpu, cuda, cuda:N
        self.device = None  # where it runs once loaded: "cuda", "cuda:1" or "cpu"
        self._torch = None  # the model as a PyTorch module, on the GPU
        self._lock = threading.Lock()
        self._session = None
        self._labels = None
        self._input = None
        self._output = None
        self.load_seconds = None
        self.problem = None

    @property
    def available(self):
        return os.path.exists(os.path.join(self.models_dir, MODEL_FILE)) and \
            os.path.exists(os.path.join(self.models_dir, LABELS_FILE))

    @property
    def loaded(self):
        return self._session is not None or self._torch is not None

    def load(self):
        with self._lock:
            if self._session is None and self._torch is None:
                started = time.monotonic()
                self._labels = labels(self.models_dir)
                if self._load_gpu():
                    self.load_seconds = time.monotonic() - started
                    return self._torch
                import onnxruntime

                options = onnxruntime.SessionOptions()
                options.intra_op_num_threads = self.threads
                options.inter_op_num_threads = 1
                session = onnxruntime.InferenceSession(os.path.join(self.models_dir, MODEL_FILE), options,
                                                       providers=["CPUExecutionProvider"])
                names = self._labels
                self._input = session.get_inputs()[0].name
                # The output with a score for each of the 400 styles (the other is the 1280-number embedding).
                outputs = session.get_outputs()
                self._output = next((o.name for o in outputs if o.shape and o.shape[-1] == len(names)), outputs[0].name)
                self._session = session
                self.device = "cpu"
                self.load_seconds = time.monotonic() - started
        return self._torch if self._torch is not None else self._session

    def _load_gpu(self):
        """The model on the GPU (onnx2torch), when there is one and it isn't turned off; False otherwise, and the CPU
        takes it. A GPU that fails to take it is said in :attr:`problem`, never an error."""
        if (self.requested or "auto").strip().lower() == "cpu":
            return False
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            from onnx2torch import convert

            from .analyser import resolve_device

            device, _ = resolve_device(self.requested, True, torch.cuda.device_count())
            if not device.startswith("cuda"):
                return False
            module = convert(os.path.join(self.models_dir, MODEL_FILE)).eval().to(device)
            with torch.inference_mode():
                outputs = module(torch.zeros((1, PATCH, BANDS), device=device))
            outputs = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
            self._pick = next(i for i, o in enumerate(outputs) if o.shape[-1] == len(self._labels))
            self._torch, self.device = module, device
            return True
        except Exception as e:  # no onnx2torch, an odd driver: the CPU does it
            self.problem = f"the style model couldn't use the GPU ({e.__class__.__name__}: {e}), so it runs on the CPU"
            return False

    @property
    def backend(self):
        """Where it runs, for the page: "GPU" or "CPU" (None until loaded)."""
        return None if self.device is None else ("GPU" if self.device.startswith("cuda") else "CPU")

    def __call__(self, samples, rate=RATE):
        """``[(label, score)]``: every style's score for the song, best first. ``samples`` mono at ``rate``."""
        import numpy as np

        model = self.load()
        if rate != RATE:
            import soxr

            samples = soxr.resample(np.asarray(samples, dtype=np.float32), rate, RATE)
        batch = patches(mel(samples))
        if model is self._torch:
            import torch

            with torch.inference_mode():
                outputs = model(torch.from_numpy(batch).to(self.device))
            outputs = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
            scores = outputs[self._pick].float().cpu().numpy()
        else:
            scores = model.run([self._output], {self._input: batch})[0]
        mean = np.asarray(scores, dtype=np.float64).reshape(len(batch), -1).mean(axis=0)
        ranked = sorted(zip(self._labels, (float(v) for v in mean)), key=lambda p: -p[1])
        return ranked
