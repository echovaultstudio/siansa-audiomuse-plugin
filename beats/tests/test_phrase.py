"""Bars and phrases from where the music changes (beats/phrase.py, grid.with_sound), and the sound's levels
(beats/features.py). The levels are made here as a song's would be: a level per band every 46 ms."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats import features, grid, phrase  # noqa: E402

HOP = features.HOP / features.RATE


def song_levels(bpm, seconds, sections, lead_bars=0, per_bar=4, seed=1, accent=3.0):
    """Packed levels of a song: ``lead_bars`` bars of intro, then 8-bar sections whose sound changes (each a
    (sub, bass, mids, highs) level in dB from ``sections`` in turn). Beat 1 of each bar has its bass ``accent`` dB
    louder (a bass note on the downbeat), and a little noise everywhere."""
    rng = random.Random(seed)
    beat = 60.0 / bpm
    frames = []
    for i in range(int(seconds / HOP)):
        t = i * HOP
        b = int(t / beat)
        bar = b // per_bar
        if bar < lead_bars:
            level = [-40.0, -35.0, -30.0, -45.0]
        else:
            level = list(sections[((bar - lead_bars) // 8) % len(sections)])
        if b % per_bar == 0:
            level[1] += accent
        frames.append([v + rng.uniform(-0.5, 0.5) for v in level])
    import base64
    import zlib

    codes = bytes(max(0, min(255, round((v - features.FLOOR_DB) * 2))) for f in frames for v in f)
    return {"hop": HOP, "bands": list(features.BANDS), "data": base64.b64encode(zlib.compress(codes)).decode()}


def grid_beats(bpm, seconds):
    period = 60.0 / bpm
    return [round(i * period, 3) for i in range(int(seconds / period))]


SECTIONS = [(-10.0, -12.0, -20.0, -40.0), (-8.0, -10.0, -14.0, -18.0), (-30.0, -25.0, -12.0, -20.0),
            (-6.0, -8.0, -12.0, -15.0)]


class LevelsTest(unittest.TestCase):
    def test_packed_levels_come_back(self):
        hop, frames = features.unpack(song_levels(128, 10, SECTIONS))
        self.assertAlmostEqual(hop, HOP)
        self.assertEqual(len(frames[0]), 4)
        self.assertAlmostEqual(frames[0][0], -10.0, delta=1.0)
        self.assertEqual(features.unpack(None), (None, []))

    def test_measure_hears_the_bands(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy")
        t = np.arange(features.RATE * 4) / features.RATE
        sub = np.sin(2 * np.pi * 60 * t).astype(np.float32)
        high = (np.random.default_rng(1).uniform(-1, 1, t.size) * 0.2).astype(np.float32)
        _, frames = features.unpack(features.measure(np.concatenate([sub, high])))
        first, last = frames[5], frames[-5]
        self.assertGreater(first[0], first[3] + 20)  # the sine: sub bass, no highs
        self.assertGreater(last[3], last[0] + 10)  # the noise: highs over sub bass
        self.assertIsNone(features.measure(np.zeros(100, dtype=np.float32)))


class DownbeatTest(unittest.TestCase):
    def test_the_changes_move_bars_the_model_got_wrong(self):
        bpm, seconds = 174, 120
        beats = grid_beats(bpm, seconds)
        levels = song_levels(bpm, seconds, SECTIONS, lead_bars=0)
        hop, frames = features.unpack(levels)
        per_beat = phrase.per_beat(hop, frames, beats)
        # The model voted for beat 2 of 4, but weakly (as 1111's wandering downbeats).
        start, source, details = phrase.downbeats(per_beat, 4, 2, 0.4)
        self.assertEqual((start, source), (0, "changes"))
        self.assertGreater(details["changes_share"], 0.5)

    def test_a_clear_vote_stands_when_the_changes_agree(self):
        beats = grid_beats(128, 120)
        hop, frames = features.unpack(song_levels(128, 120, SECTIONS))
        start, source, _ = phrase.downbeats(phrase.per_beat(hop, frames, beats), 4, 0, 0.95)
        self.assertEqual((start, source), (0, "both"))

    def test_a_few_weak_changes_dont_outweigh_a_clear_vote(self):
        bpm, seconds = 128, 40  # two sections: one change
        beats = grid_beats(bpm, seconds)
        hop, frames = features.unpack(song_levels(bpm, seconds, SECTIONS[:2], lead_bars=0))
        start, source, _ = phrase.downbeats(phrase.per_beat(hop, frames, beats), 4, 2, 0.95)
        self.assertEqual((start, source), (2, "model"))

    def test_no_changes_leave_the_vote(self):
        beats = grid_beats(128, 60)
        flat = song_levels(128, 60, [(-10.0, -10.0, -10.0, -10.0)], accent=0.0)
        hop, frames = features.unpack(flat)
        start, source, _ = phrase.downbeats(phrase.per_beat(hop, frames, beats), 4, 1, 0.9)
        self.assertEqual(start, 1)
        self.assertEqual(source, "model")


class PhraseTest(unittest.TestCase):
    def check(self, bpm, lead_bars, seconds=180):
        beats = grid_beats(bpm, seconds)
        steady = grid.steady_grid(beats, beats[::4], float(bpm))
        found = grid.with_sound(steady, song_levels(bpm, seconds, SECTIONS, lead_bars=lead_bars))
        bar = 4 * 60.0 / bpm
        self.assertTrue(found["phrases"], found)
        self.assertAlmostEqual(found["phrases"][0], lead_bars % 8 * bar, delta=0.02)
        step = found["phrase_bars"]
        self.assertTrue(all(abs((b - a) / bar - step) < 0.01 for a, b in zip(found["phrases"], found["phrases"][1:])))
        return found

    def test_phrases_start_where_the_sections_do(self):
        self.check(128, 0)

    def test_a_two_bar_lead_in_moves_the_phrases(self):
        found = self.check(140, 2)
        self.assertGreaterEqual(found["phrase_big_on"] * 2, found["phrase_big"])

    def test_drum_and_bass_with_an_odd_lead_in(self):
        self.check(174, 3)

    def test_sixteen_bar_phrases(self):
        # The sound changes every 16 bars (two sections of 8 alike), after a 2-bar lead-in.
        sixteen = [SECTIONS[0], SECTIONS[0], SECTIONS[1], SECTIONS[1], SECTIONS[2], SECTIONS[2], SECTIONS[3], SECTIONS[3]]
        beats = grid_beats(128, 300)
        steady = grid.steady_grid(beats, beats[::4], 128.0)
        found = grid.with_sound(steady, song_levels(128, 300, sixteen, lead_bars=2))
        bar = 4 * 60.0 / 128
        self.assertEqual(found["phrase_bars"], 16)
        self.assertAlmostEqual(found["phrases"][0], 2 * bar, delta=0.02)
        self.assertAlmostEqual(found["phrases"][1] - found["phrases"][0], 16 * bar, delta=0.02)

    def test_no_sections_no_phrases(self):
        beats = grid_beats(128, 120)
        steady = grid.steady_grid(beats, beats[::4], 128.0)
        flat = song_levels(128, 120, [(-10.0, -10.0, -10.0, -10.0)])
        found = grid.with_sound(steady, flat)
        self.assertEqual(found["phrases"], [])

    def test_without_levels_the_grid_stays_as_it_was(self):
        beats = grid_beats(128, 60)
        steady = grid.steady_grid(beats, beats[::4], 128.0)
        found = grid.with_sound(steady, None)
        self.assertEqual((found["downbeats"], found["downbeat_source"], found["phrases"]),
                         (steady["downbeats"], "model", []))

    def test_analyse_reports_phrases_and_notes_them(self):
        beats = grid_beats(128, 180)
        result = grid.analyse(beats, beats[1::4], 180.0, song_levels(128, 180, SECTIONS))
        self.assertEqual(result["phrase_first_ms"], 0)
        self.assertGreater(result["phrase_count"], 3)
        notes = grid.verdict(result)
        self.assertTrue(any("Bars moved" in t for _, t in notes), notes)  # the model voted for beat 2

    def test_drum_and_bass_read_at_half_speed_is_counted_at_174(self):
        half = grid_beats(87, 180)  # the model heard every other beat
        result = grid.analyse(half, half[::2], 180.0, song_levels(174, 180, SECTIONS, lead_bars=4),
                              genres=["Drum & Bass"])
        bar = 4 * 60.0 / 174
        self.assertEqual(result["grid_counted"], "double")
        self.assertAlmostEqual(result["grid_bpm"], 174.0, delta=0.1)
        self.assertEqual(result["beats_per_bar"], 4)
        self.assertAlmostEqual(result["phrase_first_ms"], 4 * bar * 1000, delta=20)

    def test_no_clear_phrases_is_flagged(self):
        result = {"bpm": 128.0, "steady": 0.99, "on_grid": 0.98, "phrase_count": 0, "phrase_big": 6, "phrase_big_on": 2}
        self.assertIn(("warn", "No clear phrases (2 of 6 big changes on one bar of 8)"), grid.verdict(result))


if __name__ == "__main__":
    unittest.main()
