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

from . import memory

RATE = 16_000
FRAME = 512
HOP = 256
BANDS = 96
PATCH = 128
PATCH_HOP = 62
MAX_PATCHES = 64
MODEL_FILE = "discogs-effnet-bsdynamic-1.onnx"
LABELS_FILE = "discogs-effnet-bs64-1.json"


class Spec:
    """One of the style models (0.39, the owner, 13 Nov: a choice of model in Setup). Both take the same mel bands
    (:func:`mel`) and give the same 400 Discogs styles, so a genre page reads the same whichever heard the songs;
    they differ in the patch they hear at once, how many patches of a song, and whether the bands are normalised
    first (MAEST's are: its mean and twice its standard deviation, from its own code)."""

    def __init__(self, id, name, file, labels_file, patch, hop, most, blurb, znorm=None, cost=""):
        self.id, self.name, self.file, self.labels_file = id, name, file, labels_file
        self.patch, self.hop, self.most, self.znorm = patch, hop, most, znorm
        self.blurb, self.cost = blurb, cost

    def available(self, models_dir):
        return os.path.exists(os.path.join(models_dir, self.file)) and os.path.exists(os.path.join(models_dir, self.labels_file))

    def summary(self, models_dir):
        return {"id": self.id, "name": self.name, "blurb": self.blurb, "cost": self.cost, "available": self.available(models_dir)}


MODELS = {
    "effnet": Spec(
        "effnet", "Discogs-EffNet (2021)", MODEL_FILE, LABELS_FILE, PATCH, PATCH_HOP, MAX_PATCHES,
        "A small, quick model. Sure of itself on electronic music, where it was mostly trained; it can misread a "
        "quiet, sparse recording (a piano ballad as dance-pop).",
        cost="About a second a song on a CPU.",
    ),
    "maest": Spec(
        "maest", "MAEST (2023)", "discogs-maest-30s-pw-1.onnx", "discogs-maest-30s-pw-1.json", 1876, 938, 8,
        "A larger transformer from the same group, hearing 30 seconds at a time. Markedly more accurate on the "
        "same 400 styles, and surer of what it hears; the better choice beyond electronic music.",
        znorm=(2.06755686098554, 2 * 1.268292820667291),
        cost="Slow on a CPU (about 4 seconds per 30 seconds heard, a few windows a song); quick on a GPU.",
    ),
}
DEFAULT_MODEL = "effnet"
KEEP = 15  # the top styles kept for each song (with their scores), its genres (beats.genres) among them, and for the page
CHECK_TOLERANCE = 0.05  # the GPU's scores may differ from ONNX Runtime's by this much before the GPU is distrusted
TENSOR_CORES_FROM = 7  # compute capability (major) from which a card has tensor cores: half precision pays there
SONGS_AT_ONCE_GPU = 2  # songs heard in one pass on a GPU (16 MAEST windows: fits a 6 GB card in half precision)


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


def patches(bands, most=MAX_PATCHES, patch=PATCH, hop=PATCH_HOP):
    """``(n, patch, 96)``: the patches the model hears, every ``hop`` frames, at most ``most`` of them spread evenly
    through the song; a song shorter than one patch is padded with silence."""
    import numpy as np

    if len(bands) < patch:
        bands = np.pad(bands, ((0, patch - len(bands)), (0, 0)))
    starts = list(range(0, len(bands) - patch + 1, hop))
    if len(starts) > most:
        starts = [starts[round(i * (len(starts) - 1) / (most - 1))] for i in range(most)]
    return np.stack([bands[s:s + patch] for s in starts])


def labels(models_dir, spec=None):
    """The 400 style names in the model's order (``Electronic---Trance``), from its metadata file."""
    with open(os.path.join(models_dir, (spec or MODELS[DEFAULT_MODEL]).labels_file), encoding="utf-8") as f:
        return list(json.load(f)["classes"])


def split(label):
    """``("Electronic", "Trance")`` from ``"Electronic---Trance"``."""
    genre, _, style = label.partition("---")
    return genre, style or genre


class StyleModel:
    """Discogs-EffNet, loaded with the first song. Thread-safe; ``available`` is False when the image has no model
    (an older image, or a test), and then nothing is asked of it."""

    def __init__(self, models_dir, threads=2, device="auto", spec=None, precision="auto"):
        self.spec = spec or MODELS[DEFAULT_MODEL]
        self.models_dir = models_dir
        self.threads = threads
        self.requested = device  # BEATS_DEVICE: auto, cpu, cuda, cuda:N
        self.device = None  # where it runs once loaded: "cuda", "cuda:1" or "cpu"
        # STYLE_PRECISION (0.41, the owner, 13 Nov): "auto" hears in half precision on a GPU with tensor cores (an
        # RTX, any card of compute capability 7 or more), full precision elsewhere; "fp16" and "fp32" insist.
        self.requested_precision = (precision or "auto").strip().lower()
        self.half = False  # hearing in half precision now (fp16 autocast on the GPU)
        self._torch = None  # the model as a PyTorch module, on the GPU
        self._lock = threading.Lock()
        self._session = None
        self._labels = None
        self._input = None
        self._output = None
        self.load_seconds = None
        self.problem = None
        # The GPU checked against ONNX Runtime on the first real song (0.37.1, the owner, 12 Nov: a country album's
        # songs scored 0.7 as electro house): {"diff", "ok"} once done; a GPU that disagrees is dropped for the CPU.
        self.checked = None

    @property
    def available(self):
        return self.spec.available(self.models_dir)

    @property
    def loaded(self):
        return self._session is not None or self._torch is not None

    def load(self):
        with self._lock:
            if self._session is None and self._torch is None:
                started = time.monotonic()
                self._labels = labels(self.models_dir, self.spec)
                if self._load_gpu():
                    self.load_seconds = time.monotonic() - started
                    return self._torch
                # The output with a score for each of the 400 styles (the other is the 1280-number embedding).
                self._session = self._cpu_session()
                self.device = "cpu"
                self.load_seconds = time.monotonic() - started
        return self._torch if self._torch is not None else self._session

    def _patches(self, samples):
        """The song's bands as the model's patches, normalised first when the model wants that (MAEST)."""
        bands = mel(samples)
        if self.spec.znorm:
            mean, scale = self.spec.znorm
            bands = ((bands - mean) / scale).astype(bands.dtype)
        return patches(bands, self.spec.most, self.spec.patch, self.spec.hop)

    def _cpu_session(self):
        """ONNX Runtime on the CPU, made when first needed (the CPU's own run, or the check of the GPU)."""
        import onnxruntime

        options = memory.session_options(onnxruntime, self.threads, inter_threads=1)
        session = onnxruntime.InferenceSession(os.path.join(self.models_dir, self.spec.file), options,
                                               providers=["CPUExecutionProvider"])
        self._input = session.get_inputs()[0].name
        outputs = session.get_outputs()
        # The output with a score for each style: MAEST gives its logits as well, so the one within 0 and 1.
        named = [o.name for o in outputs if o.shape and o.shape[-1] == len(self._labels)] or [outputs[0].name]
        self._output = named[0]
        if len(named) > 1:
            import numpy as np

            probe = np.zeros((1, self.spec.patch, BANDS), np.float32)
            scores = session.run(named, {self._input: probe})
            self._output = next((n for n, v in zip(named, scores) if float(v.min()) >= 0 and float(v.max()) <= 1), named[0])
        return session

    def _cpu_scores(self, session, batch):
        return session.run([self._output], {self._input: batch})[0]

    def _gpu_scores(self, model, batch):
        """The GPU's scores, in half precision when :attr:`half`; a GPU that can't run it in half (an op autocast
        doesn't cover, a driver) goes back to full precision for good, and the batch is heard again."""
        import torch

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
                outputs = model(torch.from_numpy(batch).to(self.device))
        except Exception as e:
            if not self.half:
                raise
            self._full_precision(f"half precision failed on the GPU ({e.__class__.__name__}: {e})")
            with torch.inference_mode():
                outputs = model(torch.from_numpy(batch).to(self.device))
        outputs = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
        return outputs[self._pick].float().cpu().numpy()

    def _full_precision(self, why):
        """Back to full precision on the GPU, saying why (kept in :attr:`problem` for the page)."""
        self.half = False
        self.problem = f"the style model runs in full precision on the GPU: {why}"

    def _wants_half(self, device):
        """Whether to hear in half precision on ``device``: asked for, or auto on a card with tensor cores."""
        if self.requested_precision in ("fp32", "full", "float32", "32"):
            return False
        if self.requested_precision in ("fp16", "half", "float16", "16"):
            return True
        try:
            import torch

            index = int(device.split(":")[1]) if ":" in device else 0
            major, _ = torch.cuda.get_device_capability(index)
            return major >= TENSOR_CORES_FROM
        except Exception:
            return False

    def _check(self, batch, gpu_mean):
        """The GPU's scores for a real song against ONNX Runtime's, once: a GPU that disagrees (a conversion or a
        driver gone wrong) is let go, and the CPU takes the model from here. The CPU's mean when it took over."""
        import numpy as np

        session = self._cpu_session()
        cpu_mean = np.asarray(self._cpu_scores(session, batch), dtype=np.float64).reshape(len(batch), -1).mean(axis=0)
        diff = float(np.abs(cpu_mean - gpu_mean).max())
        if diff > CHECK_TOLERANCE and self.half:
            # Half precision strayed: full precision gets its own chance before the GPU is given up on.
            self._full_precision(f"its half-precision scores differed from ONNX Runtime's by {diff:.2f}")
            gpu_mean = np.asarray(self._gpu_scores(self._torch, batch), dtype=np.float64).reshape(len(batch), -1).mean(axis=0)
            diff = float(np.abs(cpu_mean - gpu_mean).max())
            if diff <= CHECK_TOLERANCE:
                self.checked = {"diff": round(diff, 4), "ok": True, "precision": "fp32"}
                return gpu_mean
        self.checked = {"diff": round(diff, 4), "ok": diff <= CHECK_TOLERANCE, "precision": self.precision}
        if diff <= CHECK_TOLERANCE:
            return None
        self.problem = (f"the style model's scores on the GPU differ from ONNX Runtime's by {diff:.2f} (more than "
                        f"{CHECK_TOLERANCE}), so the CPU runs it; songs styled on the GPU are heard again")
        self._torch = None
        self._session = session
        self.device = "cpu"
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return cpu_mean

    def unload(self):
        """Lets the model go (it loads again with the next song); True when it was loaded."""
        with self._lock:
            if self._session is None and self._torch is None:
                return False
            gpu = self._torch is not None
            self._session = self._torch = None
            if gpu:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return True

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
            import onnx

            # Loaded here and handed over in memory: given the path, onnx2torch writes a temporary file beside the
            # model, and /models is read-only to the service (0.17.0: "Permission denied: '/models/tmp_…'").
            module = convert(onnx.load(os.path.join(self.models_dir, self.spec.file))).eval().to(device)
            with torch.inference_mode():
                outputs = module(torch.zeros((1, self.spec.patch, BANDS), device=device))
            outputs = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
            # The scores, within 0 and 1 (MAEST gives its logits too).
            self._pick = next(i for i, o in enumerate(outputs)
                              if o.shape[-1] == len(self._labels) and float(o.min()) >= 0 and float(o.max()) <= 1)
            self._torch, self.device = module, device
            self.half = self._wants_half(device)
            return True
        except Exception as e:  # no onnx2torch, an odd driver: the CPU does it
            self.problem = f"the style model couldn't use the GPU ({e.__class__.__name__}: {e}), so it runs on the CPU"
            return False

    @property
    def backend(self):
        """Where it runs, for the page: "GPU" or "CPU" (None until loaded)."""
        return None if self.device is None else ("GPU" if self.device.startswith("cuda") else "CPU")

    @property
    def precision(self):
        """How it hears, for the page and the log: "fp16" or "fp32" (None until loaded)."""
        return None if self.device is None else ("fp16" if self.half else "fp32")

    @property
    def songs_at_once(self):
        """How many songs are best heard in one pass: a few on a GPU (one launch, the card kept busy), one on a CPU."""
        return SONGS_AT_ONCE_GPU if self._torch is not None else 1

    def __call__(self, samples, rate=RATE):
        """``[(label, score)]``: every style's score for the song, best first. ``samples`` mono at ``rate``."""
        return self.many([(samples, rate)])[0]

    def many(self, songs):
        """The styles of several songs in one pass (0.41): ``songs`` as ``[(samples, rate)]``, each answered as
        :meth:`__call__` does, in order. Every song's patches go to the model together, so a GPU runs once for all
        of them; the first real song is still checked against the CPU, and a change of mind there (full precision,
        or the CPU) hears the whole pass again the trusted way."""
        import numpy as np

        model = self.load()
        batches = []
        for samples, rate in songs:
            if rate != RATE:
                import soxr

                samples = soxr.resample(np.asarray(samples, dtype=np.float32), rate, RATE)
            batches.append(self._patches(samples))
        means = self._means(model, batches)
        if model is self._torch and self.checked is None:
            with self._lock:
                taken_over = self._check(batches[0], means[0])
            if taken_over is not None:
                # The CPU took over, or full precision did: every song of the pass heard the trusted way.
                means = self._means(self._torch if self._torch is not None else self._session, batches)
        return [sorted(zip(self._labels, (float(v) for v in mean)), key=lambda p: -p[1]) for mean in means]

    def _means(self, model, batches):
        """Each song's mean scores over its patches, the patches of all of them run as one batch."""
        import numpy as np

        stacked = np.concatenate(batches)
        scores = self._gpu_scores(model, stacked) if model is self._torch else self._cpu_scores(model, stacked)
        scores = np.asarray(scores, dtype=np.float64).reshape(len(stacked), -1)
        means, at = [], 0
        for batch in batches:
            means.append(scores[at:at + len(batch)].mean(axis=0))
            at += len(batch)
        return means

    def both(self, samples, rate=RATE):
        """``{"gpu": ranked, "cpu": ranked}`` for one song, each way the model can run (the GPU's None without one):
        to see for oneself whether the two agree."""
        import numpy as np

        model = self.load()
        if rate != RATE:
            import soxr

            samples = soxr.resample(np.asarray(samples, dtype=np.float32), rate, RATE)
        batch = self._patches(samples)
        out = {}
        for way, scores in (("gpu", self._gpu_scores(model, batch) if model is self._torch else None),
                            ("cpu", self._cpu_scores(self._session or self._cpu_session(), batch))):
            if scores is None:
                out[way] = None
                continue
            mean = np.asarray(scores, dtype=np.float64).reshape(len(batch), -1).mean(axis=0)
            out[way] = sorted(zip(self._labels, (float(v) for v in mean)), key=lambda p: -p[1])
        return out
