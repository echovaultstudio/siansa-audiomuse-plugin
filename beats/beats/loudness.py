"""A song's loudness (ITU-R BS.1770 / EBU R128) and how it starts and ends, from its audio: what Siansa levels the
volume and shapes its fades on. The same measure as the Siansa plugin's (``siansa_analysis.loudness`` and ``ramps``),
moved into Siansa Beats so the app needs only one service (the owner, 10 Nov).

The song is decoded a second time with ffmpeg, to stereo 48 kHz on a pipe a second at a time (a whole song in memory
at 48 kHz would be ~100 MB), K-weighted and cut into 100 ms chunks. From the chunks come:

* the integrated loudness in LUFS, gated as BS.1770-4 says (400 ms blocks, 75 % overlap, an absolute gate at
  -70 LUFS and a relative gate 10 LU below);
* each chunk's own loudness relative to that, which the ramps turn into fade shapes.

The K-weighting runs as its impulse response by FFT (overlap-add): numpy only, no scipy in the image. Its two filters
die away within a few ms, so a third of a second of response is the filter itself to well under 0.01 dB.

A ramp is a list of ``(level, chunk)`` pairs, ``level`` in dB relative to the song's loudness, ``chunk`` counting 100 ms
chunks: the start ramp says how soon the song first gets up to each level, the end ramp how long before the end it's
last at each level. On the wire each pair is 3 bytes (signed 8-bit level, little-endian unsigned 16-bit chunk),
base64: the layout the app has read since its issue #60 (``RampCodec``).
"""

import base64
import math
import shutil
import struct
import subprocess
import tempfile
import time

import numpy as np

CHUNK_SECONDS = 0.1
RATE = 48_000
LUFS_OFFSET = -0.691
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0
BLOCK_CHUNKS = 4  # a 400 ms gating block is four 100 ms chunks, stepped by one (75 % overlap)
RESPONSE = 16_384  # samples of the K-weighting's impulse response kept (0.34 s at 48 kHz)
VERSION = 1  # of this measure: a newer one measures songs again

# The levels a ramp records, quietest first. Siansa fades on -18 dB; the others let it pick another level later
# without measuring again. At most 15 levels fit a ramp.
LEVELS = (-60, -48, -36, -30, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6)
MAX_LEVELS = 15
MAX_CHUNK = 0xFFFF


def k_weighting(rate):
    """The two K-weighting filters (a high shelf, then a high-pass) as ``[(b, a), (b, a)]`` for ``rate``: the
    analogue design by the bilinear transform, so at 48 kHz it's BS.1770's coefficient table."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = ((
        (vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0),
        (1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0))
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1 + k / q + k * k
    highpass = ((1.0, -2.0, 1.0), (1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0))
    return [shelf, highpass]


def _biquad(b, a, x):
    """One biquad over ``x`` (a list), sample by sample: only for the short impulse response."""
    y, x1, x2, y1, y2 = [], 0.0, 0.0, 0.0, 0.0
    for v in x:
        out = b[0] * v + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1, y2, y1 = x1, v, y1, out
        y.append(out)
    return y


_RESPONSES = {}


def impulse_response(rate, length=RESPONSE):
    """The K-weighting's impulse response at ``rate``, ``length`` samples."""
    key = (rate, length)
    if key not in _RESPONSES:
        signal = [1.0] + [0.0] * (length - 1)
        for b, a in k_weighting(rate):
            signal = _biquad(b, a, signal)
        _RESPONSES[key] = np.asarray(signal, dtype=np.float64)
    return _RESPONSES[key]


def to_lufs(mean_square):
    """Loudness of a summed channel mean square; silence is ``-inf``."""
    return LUFS_OFFSET + 10 * math.log10(mean_square) if mean_square > 0 else float("-inf")


class Meter:
    """Takes audio in blocks (``(channels, frames)``, float, -1..1) and keeps the K-weighted energy of every 100 ms
    chunk. Left and right count fully, as BS.1770 says for stereo; mono is one channel."""

    def __init__(self, rate, channels):
        self.rate, self.channels = rate, channels
        self.chunk = max(1, int(round(rate * CHUNK_SECONDS)))
        self.response = impulse_response(rate)
        self.spectra = {}  # FFT size -> the response's spectrum
        self.carry = np.zeros((channels, self.response.size - 1))  # the filters' ringing into the next block
        self.energy = []
        self.partial = 0.0
        self.filled = 0
        self.frames = 0

    def _filter(self, block):
        n = block.shape[1]
        size = 1 << (n + self.response.size - 1 - 1).bit_length()
        if size not in self.spectra:
            self.spectra[size] = np.fft.rfft(self.response, size)
        out = np.fft.irfft(np.fft.rfft(block, size, axis=1) * self.spectra[size], size, axis=1)[:, :n + self.response.size - 1]
        out[:, :self.carry.shape[1]] += self.carry
        self.carry = out[:, n:].copy()
        return out[:, :n]

    def add(self, block):
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            block = block[np.newaxis, :]
        if block.shape[0] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {block.shape[0]}")
        if block.shape[1] == 0:
            return
        weighted = self._filter(block)
        squares = np.sum(weighted * weighted, axis=0)
        self.frames += squares.size
        offset = 0
        while offset < squares.size:
            take = min(self.chunk - self.filled, squares.size - offset)
            self.partial += float(np.sum(squares[offset:offset + take]))
            self.filled += take
            offset += take
            if self.filled == self.chunk:
                self.energy.append(self.partial / self.chunk)
                self.partial = 0.0
                self.filled = 0

    def finish(self):
        """Closes the last, shorter chunk: ``(chunk energies, duration in ms)``."""
        if self.filled:
            self.energy.append(self.partial / self.filled)
            self.partial, self.filled = 0.0, 0
        return list(self.energy), int(round(self.frames * 1000 / self.rate))


def integrated(chunk_energy):
    """Gated integrated loudness from the chunks' energies, or ``None`` when the song is (near) silent."""
    if not chunk_energy:
        return None
    energy = np.asarray(chunk_energy, dtype=np.float64)
    if energy.size < BLOCK_CHUNKS:
        blocks = np.array([energy.mean()])
    else:
        blocks = np.convolve(energy, np.ones(BLOCK_CHUNKS) / BLOCK_CHUNKS, mode="valid")
    with np.errstate(divide="ignore"):
        loudness = LUFS_OFFSET + 10 * np.log10(blocks)
    kept = blocks[loudness > ABSOLUTE_GATE]
    if kept.size == 0:
        return None
    relative = to_lufs(float(kept.mean())) + RELATIVE_GATE
    with np.errstate(divide="ignore"):
        kept = kept[LUFS_OFFSET + 10 * np.log10(kept) > relative]
    if kept.size == 0:
        return None
    return to_lufs(float(kept.mean()))


def relative_levels(chunk_energy, lufs):
    """Each chunk's loudness minus the song's, in dB (``-inf`` for digital silence)."""
    if lufs is None:
        return []
    return [to_lufs(e) - lufs for e in chunk_energy]


def start_ramp(relative):
    """For each level, the first chunk at or above it; stops at the first level the song never reaches."""
    out = []
    for level in LEVELS:
        index = next((i for i, value in enumerate(relative) if value >= level), None)
        if index is None:
            break
        out.append((level, min(index, MAX_CHUNK)))
    return out


def end_ramp(relative):
    """For each level, how many chunks before the end the song is last at or above it."""
    out = []
    last = len(relative) - 1
    for level in LEVELS:
        index = next((i for i in range(last, -1, -1) if relative[i] >= level), None)
        if index is None:
            break
        out.append((level, min(last - index, MAX_CHUNK)))
    return out


def encode(pairs):
    """``[(level, chunk), …]`` as the base64 blob the app decodes."""
    if len(pairs) > MAX_LEVELS:
        raise ValueError(f"a ramp holds at most {MAX_LEVELS} levels")
    return base64.b64encode(b"".join(struct.pack("<bH", int(lv), int(ch)) for lv, ch in pairs)).decode("ascii")


def decode_ramp(blob):
    raw = base64.b64decode(blob) if blob else b""
    if len(raw) % 3:
        raise ValueError("a ramp is a whole number of 3-byte entries")
    return [struct.unpack_from("<bH", raw, i) for i in range(0, len(raw), 3)]


def profile(chunk_energy, duration_ms):
    """``{"ref_lufs", "start_ramp", "end_ramp", "duration_ms"}``: the app's loudness and fades for one song."""
    lufs = integrated(chunk_energy)
    relative = relative_levels(chunk_energy, lufs)
    return {"ref_lufs": None if lufs is None else round(lufs, 2), "start_ramp": encode(start_ramp(relative)),
            "end_ramp": encode(end_ramp(relative)), "duration_ms": duration_ms}


def measure_array(samples, rate):
    """The profile of audio in memory (``(channels, frames)`` or one channel)."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
    meter = Meter(rate, samples.shape[0])
    for start in range(0, samples.shape[1], rate):
        meter.add(samples[:, start:start + rate])
    return profile(*meter.finish())


def _channels(path):
    """1 or 2: the first audio stream's channels as ``ffprobe`` sees them; 2 when it can't tell."""
    probe = shutil.which("ffprobe")
    if not probe:
        return 2
    try:
        out = subprocess.run([probe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
                              "-of", "csv=p=0", path], capture_output=True, text=True, timeout=20).stdout.strip()
        return 1 if out.splitlines() and out.splitlines()[0].strip() == "1" else 2
    except (OSError, subprocess.SubprocessError, ValueError):
        return 2


def measure_file(path, deadline_seconds=300.0):
    """The profile of the song at ``path``, decoded by ffmpeg to 48 kHz on a pipe a second at a time. Raises
    ``RuntimeError`` when ffmpeg can't read it and ``TimeoutError`` after ``deadline_seconds``."""
    started = time.monotonic()
    channels = _channels(path)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-vn", "-f", "f32le",
               "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(RATE), "pipe:1"]
    meter = Meter(RATE, channels)
    block = RATE * channels * 4
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, stdin=subprocess.DEVNULL)
        try:
            rest = b""
            while True:
                data = process.stdout.read(block)
                if not data:
                    break
                data = rest + data
                usable = len(data) - len(data) % (4 * channels)
                rest = data[usable:]
                if usable:
                    meter.add(np.frombuffer(data[:usable], dtype="<f4").reshape(-1, channels).T)
                if time.monotonic() - started > deadline_seconds:
                    raise TimeoutError(f"measuring took longer than {deadline_seconds:.0f} s")
            code = process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
        if code != 0 or meter.frames == 0:
            errors.seek(0)
            said = errors.read().decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError("ffmpeg couldn't read the song for its loudness: "
                               + (said[-1] if said else f"ended with code {code} and no audio"))
    return profile(*meter.finish())
