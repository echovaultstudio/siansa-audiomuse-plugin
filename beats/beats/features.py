"""How loud a song's sub bass, bass, mids and highs are along it: what tells where its phrases start.

In dance music a phrase (8 or 16 bars) starts where the music changes: the bass comes in, the drop, the hats
stop. Those changes are in these levels. They're measured once, while the song is decoded for the model, and kept
small (a byte per band every 46 ms, compressed: about 20 kB for a five-minute song), so the phrases can be worked
out again later without downloading the song again.

:func:`measure` needs numpy (the image has it); :func:`unpack` is plain Python, so the phrase maths is tested
without it.
"""

import array
import base64
import zlib

RATE = 22_050
FFT = 2048
HOP = 1024  # 46 ms at 22 050 Hz
BANDS = (20.0, 120.0, 500.0, 2500.0, 11_025.0)  # sub bass, bass, mids, highs
FLOOR_DB = -100.0  # levels are kept in half dB steps from here, as bytes


def measure(samples, rate=RATE):
    """The levels of ``samples`` (mono float32 numpy array at ``rate``), packed for the database: a dict with
    ``hop`` (seconds between frames), ``bands`` (their edges in Hz) and ``data``, or ``None`` for too short a song."""
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < FFT * 4:
        return None
    frames = 1 + (samples.size - FFT) // HOP
    window = np.hanning(FFT).astype(np.float32)
    scale = float(window.sum()) ** 2  # so a full-scale tone is about 0 dB, well inside what a byte keeps
    freqs = np.fft.rfftfreq(FFT, 1.0 / rate)
    edges = [int(np.searchsorted(freqs, f)) for f in BANDS]
    out = np.empty((frames, len(BANDS) - 1), dtype=np.float32)
    step = 2048  # frames at a time, to keep memory small for long songs
    for start in range(0, frames, step):
        count = min(step, frames - start)
        index = (np.arange(count)[:, None] + start) * HOP + np.arange(FFT)[None, :]
        power = np.abs(np.fft.rfft(samples[index] * window, axis=1)) ** 2 / scale
        for band in range(len(BANDS) - 1):
            energy = power[:, edges[band]:max(edges[band] + 1, edges[band + 1])].sum(axis=1)
            out[start:start + count, band] = 10.0 * np.log10(energy + 1e-10)
    codes = np.clip(np.round((out - FLOOR_DB) * 2.0), 0, 255).astype(np.uint8)
    return {"hop": HOP / rate, "bands": list(BANDS), "data": base64.b64encode(zlib.compress(codes.tobytes(), 9)).decode()}


def unpack(packed):
    """``(hop, frames)`` from :func:`measure`'s dict: ``frames`` a list of per-band levels (dB) every ``hop``
    seconds. ``(None, [])`` without levels."""
    if not packed or not packed.get("data"):
        return None, []
    width = len(packed.get("bands") or BANDS) - 1
    codes = array.array("B", zlib.decompress(base64.b64decode(packed["data"])))
    levels = [c / 2.0 + FLOOR_DB for c in codes]
    return packed["hop"], [levels[i:i + width] for i in range(0, len(levels) - width + 1, width)]
