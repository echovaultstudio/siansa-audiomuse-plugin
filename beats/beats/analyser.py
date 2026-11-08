"""Beat This! (CPJKU, 2024): every beat and downbeat of a song, from its audio.

The song is decoded with ffmpeg to mono 22 050 Hz (what the model listens at), then the model finds the beats and
downbeats over the whole song. The model is loaded once, on the first song, from the checkpoints the image
carries (``/models``), so the service needs no internet. It runs on an NVIDIA GPU when the ``cuda`` image is given
one (BEATS_DEVICE ``auto``, the default), else on the CPU. torch is imported only here, so the rest of the service
and its tests run without it.
"""

import os
import subprocess
import threading
import time

RATE = 22_050


def decode(path, rate=RATE):
    """The song at ``path`` as mono float32 samples at ``rate``, via ffmpeg (any format ffmpeg reads)."""
    import numpy as np

    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
    done = subprocess.run(command, capture_output=True, timeout=600)
    if done.returncode != 0:
        raise RuntimeError("ffmpeg couldn't read the song: " + done.stderr.decode(errors="replace").strip()[-300:])
    samples = np.frombuffer(done.stdout, dtype=np.float32)
    if samples.size < rate:  # under a second
        raise RuntimeError("the song has less than a second of audio")
    return samples


def resolve_device(requested, cuda_available, gpu_count=0):
    """``(device, note)``: where the model runs for BEATS_DEVICE ``requested`` (``auto``, ``cpu``, ``cuda`` or
    ``cuda:N``). ``auto`` takes the GPU when there is one. A GPU asked for but not there falls back to the CPU, with a
    note saying why, rather than not working at all."""
    requested = (requested or "auto").strip().lower()
    if requested == "cpu":
        return "cpu", None
    if requested == "auto":
        return ("cuda", None) if cuda_available else ("cpu", None)
    if requested.startswith("cuda"):
        if not cuda_available:
            return "cpu", ("A GPU was asked for (BEATS_DEVICE=" + requested + ") but none is visible to the container, "
                           "so the CPU is used. Use the cuda image and give the container the GPU (see the README).")
        index = requested.partition(":")[2]
        if index and (not index.isdigit() or int(index) >= gpu_count):
            return "cuda", f"There's no GPU {index} (the container sees {gpu_count}), so the first is used."
        return requested, None
    return "cpu", f"BEATS_DEVICE={requested} isn't known (auto, cpu, cuda or cuda:N), so the CPU is used."


class BeatThis:
    """Loads the model on first use. ``__call__(samples)`` gives ``(beats, downbeats)``, lists of seconds."""

    def __init__(self, model="final0", device="auto", models_dir="/models", threads=0, float16=False):
        self.model = model
        self.requested = device
        self.device = None  # where it runs, once loaded
        self.device_name = None  # "NVIDIA GeForce RTX 3060", or "CPU"
        self.device_note = None  # why it's not where it was asked to be
        self.float16 = float16
        self.models_dir = models_dir
        self.threads = threads
        self._lock = threading.Lock()
        self._tracker = None
        self.load_seconds = None

    @property
    def loaded(self):
        return self._tracker is not None

    def checkpoint(self):
        """The model's file in the image, or its name (Beat This! then downloads it) when the image has none."""
        local = os.path.join(self.models_dir, f"{self.model}.ckpt")
        return local if os.path.exists(local) else self.model

    def load(self):
        with self._lock:
            if self._tracker is None:
                started = time.monotonic()
                import torch
                from beat_this.inference import Audio2Beats

                if self.threads:
                    torch.set_num_threads(self.threads)
                available = torch.cuda.is_available()
                self.device, self.device_note = resolve_device(
                    self.requested, available, torch.cuda.device_count() if available else 0)
                gpu = self.device.startswith("cuda")
                self.device_name = torch.cuda.get_device_name(torch.device(self.device)) if gpu else "CPU"
                self._tracker = Audio2Beats(checkpoint_path=self.checkpoint(), device=self.device,
                                            float16=self.float16 and gpu, dbn=False)
                self.load_seconds = time.monotonic() - started
        return self._tracker

    def __call__(self, samples, rate=RATE):
        tracker = self.load()
        import torch

        with torch.inference_mode():
            beats, downbeats = tracker(samples, rate)
        if self.device and self.device.startswith("cuda"):
            torch.cuda.empty_cache()  # a long song's memory, handed back before the next
        return [float(b) for b in beats], [float(d) for d in downbeats]
