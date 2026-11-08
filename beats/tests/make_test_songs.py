#!/usr/bin/env python3
"""Makes the end-to-end check's songs: short pieces of music at known tempos and metres, as FLAC, with a
catalog.json for the fake Navidrome (tests/fakes.py) that says what each should measure.

Each has a kick on every beat, a bass note and a chord change on every downbeat, a clap on the backbeats and
hi-hats between, so its beats and bars are unambiguous to a listener. And phrases, as dance music has them: after
a lead-in of a few bars (the kick alone), 8-bar sections that change in turn: the groove (kick, bass, chords),
the full beat (with claps and hats), a breakdown (chords alone, no kick), and the peak (everything, louder hats).
So the phrases start a known number of bars in (``expect_phrase_bar``). One also carries the plugin's misreading
of Control (4/3 of its tempo) and one a BPM tag at half its tempo, so the comparison's checks get exercised.

    python3 tests/make_test_songs.py <folder>
"""

import json
import math
import os
import random
import struct
import subprocess
import sys
import wave

RATE = 44_100

SONGS = [
    # id, title, artist, genre, bpm, beats a bar, seconds, BPM tag, plugin's readings, lead-in bars
    ("t140", "Steady 140", "Test Trance", "Trance", 140.0, 4, 100, 140, {"intro_bpm": 187.1, "outro_bpm": 140.1}, 2),
    ("t174", "Rolling 174", "Test Breaks", "Drum & Bass", 174.0, 4, 80, 174, {"intro_bpm": 87.0, "outro_bpm": 174.0}, 4),
    ("t122", "Deep 122", "Test House", "House", 122.0, 4, 90, 61, None, 0),
    ("t96w", "Waltz 96", "Test Waltz", "Pop", 96.0, 3, 70, None, None, 0),
    ("t128", "Club 128", "Test Club", "Techno", 128.0, 4, 100, 128, {"intro_bpm": 128.0, "outro_bpm": 128.0}, 3),
]


def note(freq, t, length):
    return math.sin(2 * math.pi * freq * t) * max(0.0, 1.0 - t / length)


def render(bpm, per_bar, seconds, seed, lead=0):
    rng = random.Random(seed)
    n = int(seconds * RATE)
    out = [0.0] * n
    beat = 60.0 / bpm
    chords = [(110.0, 138.6, 164.8), (98.0, 123.5, 146.8), (87.3, 110.0, 130.8), (82.4, 103.8, 123.5)]
    noise = [rng.uniform(-1, 1) for _ in range(int(0.06 * RATE))]

    def add(start, samples, gain):
        s = int(start * RATE)
        for i, v in enumerate(samples):
            if 0 <= s + i < n:
                out[s + i] += gain * v

    kick = [math.sin(2 * math.pi * (55 + 90 * math.exp(-i / 600)) * i / RATE) * math.exp(-i / 4000)
            for i in range(int(0.25 * RATE))]
    clap = [noise[i % len(noise)] * math.exp(-i / 1800) for i in range(int(0.15 * RATE))]
    hat = [noise[(i * 7) % len(noise)] * math.exp(-i / 300) for i in range(int(0.05 * RATE))]
    total = int(seconds / beat)
    for b in range(total):
        t = b * beat
        position = b % per_bar
        bar = b // per_bar
        # The section: the lead-in, then groove, full, breakdown, peak, 8 bars each.
        part = "lead" if bar < lead else ("groove", "full", "breakdown", "peak")[((bar - lead) // 8) % 4]
        if part != "breakdown":
            add(t, kick, 0.9 if position == 0 else 0.7)
        if part == "lead":
            continue
        if part in ("full", "peak"):
            if per_bar == 4 and position in (1, 3):
                add(t, clap, 0.35)
            if per_bar == 3 and position in (1, 2):
                add(t, clap, 0.2)
            add(t + beat / 2, hat, 0.3 if part == "peak" else 0.15)
        if position == 0:
            root, third, fifth = chords[bar % len(chords)]
            length = beat * per_bar
            pad = [0.12 * (note(root, i / RATE, length) + note(third * 2, i / RATE, length) + note(fifth * 2, i / RATE, length))
                   for i in range(int(length * RATE))]
            add(t, pad, 1.0)
            if part != "breakdown":
                bass = [note(root / 2, i / RATE, beat) for i in range(int(beat * RATE))]
                add(t, bass, 0.45)
    peak = max(1e-9, max(abs(v) for v in out))
    return [v / peak * 0.85 for v in out]


def write(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(v * 32767)) for v in samples))


def main(folder):
    os.makedirs(folder, exist_ok=True)
    catalog = []
    for i, (sid, title, artist, genre, bpm, per_bar, seconds, tag, plugin, lead) in enumerate(SONGS):
        wav = os.path.join(folder, f"{sid}.wav")
        write(wav, render(bpm, per_bar, seconds, i, lead))
        name = f"{sid}.flac"
        done = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, os.path.join(folder, name)])
        if done.returncode == 0:
            os.remove(wav)
        else:
            name = f"{sid}.wav"
        catalog.append({"id": sid, "title": title, "artist": artist, "album": "Siansa Beats test", "genre": genre,
                        "bpm": tag, "duration": seconds, "file": name, "expect_bpm": bpm, "expect_bar": per_bar,
                        "expect_phrase_bar": lead % 8, "plugin": plugin})
        print(f"{name}: {bpm} BPM, {per_bar} a bar, {seconds} s", flush=True)
    with open(os.path.join(folder, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test-songs")
