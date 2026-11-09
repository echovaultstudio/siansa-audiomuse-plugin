"""Each song's style from its sound: Essentia's Discogs-EffNet (MTG, UPF), 400 Discogs styles.

The model is an EfficientNet-B0 trained by the Music Technology Group on over two million recordings labelled with the
Discogs styles of their releases (``Electronic---Progressive Trance``, ``Rock---Punk``…). It hears 2-second patches of
a mel spectrogram and gives each style a score from 0 to 1; a song's scores are the mean over its patches, as
Essentia's own examples do. It runs on the CPU with ONNX Runtime (``discogs-effnet-bsdynamic-1.onnx``, about 18 MB,
downloaded while the image is built), so the GPU stays free for Beat This!.

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

    def __init__(self, models_dir, threads=2):
        self.models_dir = models_dir
        self.threads = threads
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
        return self._session is not None

    def load(self):
        with self._lock:
            if self._session is None:
                started = time.monotonic()
                import onnxruntime

                options = onnxruntime.SessionOptions()
                options.intra_op_num_threads = self.threads
                options.inter_op_num_threads = 1
                session = onnxruntime.InferenceSession(os.path.join(self.models_dir, MODEL_FILE), options,
                                                       providers=["CPUExecutionProvider"])
                names = labels(self.models_dir)
                self._input = session.get_inputs()[0].name
                # The output with a score for each of the 400 styles (the other is the 1280-number embedding).
                outputs = session.get_outputs()
                self._output = next((o.name for o in outputs if o.shape and o.shape[-1] == len(names)), outputs[0].name)
                self._labels = names
                self._session = session
                self.load_seconds = time.monotonic() - started
        return self._session

    def __call__(self, samples, rate=RATE):
        """``[(label, score)]``: every style's score for the song, best first. ``samples`` mono at ``rate``."""
        import numpy as np

        session = self.load()
        if rate != RATE:
            import soxr

            samples = soxr.resample(np.asarray(samples, dtype=np.float32), rate, RATE)
        batch = patches(mel(samples))
        scores = session.run([self._output], {self._input: batch})[0]
        mean = np.asarray(scores, dtype=np.float64).reshape(len(batch), -1).mean(axis=0)
        ranked = sorted(zip(self._labels, (float(v) for v in mean)), key=lambda p: -p[1])
        return ranked
