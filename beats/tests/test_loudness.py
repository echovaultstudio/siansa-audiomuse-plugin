"""The loudness and fades measure (beats.loudness) and the key finder (beats.key)."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats import key, loudness  # noqa: E402

RATE = loudness.RATE


def tone(freq, seconds, amplitude, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return amplitude * np.sin(2 * np.pi * freq * t)


class LoudnessTest(unittest.TestCase):
    def test_a_sine_at_minus_20_is_minus_20_lufs(self):
        # BS.1770: a 997 Hz sine in both channels reads its level in dBFS; at 0.1 that's -20.
        s = tone(997, 10, 0.1)
        p = loudness.measure_array(np.vstack([s, s]), RATE)
        self.assertAlmostEqual(p["ref_lufs"], -20.0, delta=0.05)
        self.assertEqual(p["duration_ms"], 10_000)

    def test_the_same_as_filtering_sample_by_sample(self):
        try:
            from scipy.signal import lfilter
        except ImportError:
            self.skipTest("scipy isn't here")
        noise = np.random.default_rng(1).standard_normal((2, RATE * 20)) * 0.05
        weighted = noise.copy()
        for b, a in loudness.k_weighting(RATE):
            weighted = lfilter(b, a, weighted, axis=1)
        squares = (weighted * weighted).sum(axis=0)
        exact = [squares[i:i + 4800].mean() for i in range(0, squares.size, 4800)]
        meter = loudness.Meter(RATE, 2)
        for start in range(0, noise.shape[1], 7_000):  # blocks of any size
            meter.add(noise[:, start:start + 7_000])
        measured, _ = meter.finish()
        self.assertEqual(len(measured), len(exact))
        self.assertLess(max(abs(10 * np.log10(a / b)) for a, b in zip(exact, measured)), 0.001)

    def test_ramps_follow_a_fade_in_and_out(self):
        s = tone(1000, 30, 0.3)
        s[:RATE * 5] *= np.linspace(0, 1, RATE * 5) ** 3  # a slow intro
        s[-RATE * 2:] *= np.linspace(1, 0, RATE * 2)  # a short fade out
        p = loudness.measure_array(s, RATE)
        start = dict(loudness.decode_ramp(p["start_ramp"]))
        end = dict(loudness.decode_ramp(p["end_ramp"]))
        self.assertGreater(start[-18], start[-60])  # it takes a while to get loud
        self.assertGreater(start[-6], 20)  # about 2–5 s in, in 100 ms chunks
        self.assertLess(end[-18], 20)  # it's loud until under 2 s before the end
        self.assertEqual(loudness.decode_ramp(loudness.encode([(-18, 7), (0, 300)])), [(-18, 7), (0, 300)])

    def test_silence_has_no_loudness(self):
        p = loudness.measure_array(np.zeros(RATE * 3), RATE)
        self.assertIsNone(p["ref_lufs"])
        self.assertEqual(p["start_ramp"], "")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg isn't here")
    def test_a_file_through_ffmpeg(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        path = os.path.join(folder, "song.flac")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=44100:duration=8",
                        "-af", "volume=0.1", "-ac", "2", path], check=True)
        p = loudness.measure_file(path)
        said = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                              capture_output=True, text=True).stderr
        reference = float(said.rsplit("I:", 1)[1].split("LUFS")[0])  # ffmpeg's own EBU R128 meter
        self.assertAlmostEqual(p["ref_lufs"], reference, delta=0.1)
        self.assertAlmostEqual(p["duration_ms"], 8000, delta=50)
        broken = os.path.join(folder, "broken.mp3")
        with open(broken, "w") as f:
            f.write("not audio")
        with self.assertRaises(RuntimeError):
            loudness.measure_file(broken)


class KeyTest(unittest.TestCase):
    def chord(self, freqs, rate=22_050):
        return sum(tone(f, 15, 0.2, rate) for f in freqs).astype(np.float32)

    def test_a_minor_and_c_major(self):
        found = key.find(self.chord((110, 220, 261.63, 329.63)), 22_050)
        self.assertEqual((found["key"], found["scale"]), ("A", "minor"))
        found = key.find(self.chord((130.81, 261.63, 329.63, 392.0)), 22_050)
        self.assertEqual((found["key"], found["scale"]), ("C", "major"))
        self.assertTrue(0 < found["key_strength"] <= 1)

    def test_silence_has_no_key(self):
        self.assertIsNone(key.find(np.zeros(22_050 * 5, np.float32), 22_050))
        self.assertIsNone(key.find(np.zeros(100, np.float32), 22_050))


if __name__ == "__main__":
    unittest.main()
