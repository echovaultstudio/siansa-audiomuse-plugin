"""A song's key and scale (C# minor), from the samples already decoded for the beats: for the app's blends in key (the
Camelot wheel). Until 0.20 the app had them from AudioMuse-AI through the plugin; Siansa Beats finds them itself now.

The classic way (Krumhansl and Kessler's key profiles): the sound's spectrum, frame by frame, folded onto the twelve
pitch classes from ~65 Hz to ~2 kHz (a chromagram), averaged over the song, and compared with each of the 24 major and
minor profiles turned to every tonic. The best match is the key, and how well it matches its strength.
"""

import numpy as np

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FRAME = 8192
LOW_HZ, HIGH_HZ = 65.0, 2100.0
MOST_FRAMES = 600  # spread over the song: enough for its key, quick on a long mix


def chroma(samples, rate):
    """The song's average energy in each of the twelve pitch classes (C first), summing to 1; zeros for silence."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < FRAME:
        return np.zeros(12)
    starts = np.arange(0, samples.size - FRAME + 1, FRAME // 2)
    if starts.size > MOST_FRAMES:
        starts = starts[np.linspace(0, starts.size - 1, MOST_FRAMES).astype(int)]
    freqs = np.fft.rfftfreq(FRAME, 1.0 / rate)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    classes = (np.round(12 * np.log2(freqs[band] / 440.0)).astype(int) + 9) % 12  # A440 is class 9
    window = np.hanning(FRAME).astype(np.float32)
    total = np.zeros(12)
    for start in starts:
        spectrum = np.abs(np.fft.rfft(samples[start:start + FRAME] * window))[band]
        frame = np.bincount(classes, weights=np.sqrt(spectrum), minlength=12)
        peak = frame.max()
        if peak > 0:
            total += frame / peak
    s = total.sum()
    return total / s if s > 0 else total


def find(samples, rate):
    """``{"key", "scale", "key_strength"}`` (``scale`` "major" or "minor"; the strength, how well the best profile matches, 0..1), or ``None`` for
    silence."""
    c = chroma(samples, rate)
    if not c.any():
        return None
    scores = []
    for tonic in range(12):
        for scale, shape in (("major", MAJOR), ("minor", MINOR)):
            scores.append((float(np.corrcoef(c, np.roll(shape, tonic))[0, 1]), tonic, scale))
    scores.sort(reverse=True)
    best = scores[0]
    if not np.isfinite(best[0]):
        return None
    return {"key": NAMES[best[1]], "scale": best[2], "key_strength": round(max(0.0, min(1.0, best[0])), 3)}
