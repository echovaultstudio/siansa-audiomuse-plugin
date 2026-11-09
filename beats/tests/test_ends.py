"""Where a song's beat locks in, its bass at each end, drops and breakdowns (beats/ends.py), and the ends the app
reads (grid.app_ends) taken inside the lock."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats import ends, grid  # noqa: E402
from test_phrase import SECTIONS, grid_beats, song_levels  # noqa: E402


class LockTest(unittest.TestCase):
    def test_a_song_that_follows_its_grid_locks_at_its_first_and_last_bar(self):
        beats = grid_beats(128, 60)
        intro, outro = ends.lock_points(beats, [True] * len(beats), beats[::4])
        self.assertEqual(intro, beats[0])
        self.assertEqual(outro, beats[::4][-1])

    def test_a_mix_track_locks_after_the_djs_transition(self):
        # The first 40 beats wander off the grid (the DJ's last record still playing), and the last 24.
        beats = grid_beats(124.6, 120)
        heard = [i % 3 == 0 if i < 40 or i >= len(beats) - 24 else True for i in range(len(beats))]
        intro, outro = ends.lock_points(beats, heard, beats[::4])
        self.assertGreaterEqual(intro, beats[40] - 0.01)
        self.assertAlmostEqual(intro % (4 * 60 / 124.6), 0.0, delta=0.02)  # on a bar
        self.assertLessEqual(outro, beats[len(beats) - 24])

    def test_one_missed_beat_doesnt_break_the_lock(self):
        beats = grid_beats(128, 30)
        heard = [i != 2 for i in range(len(beats))]
        self.assertEqual(ends.lock_points(beats, heard, beats[::4])[0], beats[0])

    def test_no_four_bars_on_the_grid_no_lock(self):
        beats = grid_beats(128, 30)
        self.assertEqual(ends.lock_points(beats, [i % 2 == 0 for i in range(len(beats))], beats[::4]), (None, None))


class EndsTest(unittest.TestCase):
    def analyse(self, model_beats, levels, bpm=128.0, seconds=180):
        return grid.analyse(model_beats, model_beats[::4], seconds, levels)["ends"]

    def test_the_app_ends_are_taken_inside_the_lock(self):
        beats = grid_beats(124.6, 180)
        bar = 4 * 60 / 124.6
        # The model hears the start off the grid for 12 bars (a DJ's transition), then the song's own beat.
        wobbly = [b + (0.12 if i % 2 else -0.11) if i < 48 else b for i, b in enumerate(beats)]
        found = grid.analyse(wobbly, beats[::4], 180.0, song_levels(124.6, 180, SECTIONS))["ends"]
        self.assertGreaterEqual(found["intro_lock_ms"], 12 * bar * 1000 - 50)
        self.assertGreaterEqual(found["intro_beat_ms"], found["intro_lock_ms"] - 1)
        self.assertGreaterEqual(found["intro_bar_ms"], found["intro_lock_ms"] - 1)
        self.assertGreater(found["intro_rhythm"], 0.9)  # inside the lock, the beat is all there

    def test_a_steady_song_is_as_before(self):
        beats = grid_beats(128, 180)
        found = self.analyse(beats, song_levels(128, 180, SECTIONS))
        self.assertLess(found["intro_lock_ms"], 50)
        self.assertEqual((found["intro_rhythm"], found["outro_rhythm"]), (1.0, 1.0))
        self.assertTrue(found["mix_track"])  # SECTIONS is at full volume at both ends, as a track cut from a mix

    def test_bass_bars_drops_and_breakdowns(self):
        # 16 bars of drums alone, then the bass; 8-bar sections loud, quiet (a breakdown), loud (the drop), and 8 bars
        # of drums alone at the end.
        drums = (-40.0, -45.0, -12.0, -15.0)
        loud = (-8.0, -8.0, -12.0, -15.0)
        sections = [drums, drums, loud, loud, drums, loud, loud, drums]
        beats = grid_beats(128, 8 * 8 * 4 * 60 / 128)
        seconds = beats[-1] + 60 / 128
        levels = song_levels(128, seconds, sections, accent=0.0)
        found = grid.analyse(beats, beats[::4], seconds, levels)["ends"]
        bar = 4 * 60 / 128
        self.assertEqual(found["intro_bass_bars"], 16)
        self.assertGreaterEqual(found["outro_bass_bars"], 7)
        self.assertTrue(any(abs(d - 40 * bar * 1000) < 100 for d in found["drops_ms"]), found["drops_ms"])
        self.assertTrue(any(abs(d - 32 * bar * 1000) < 100 for d in found["breakdowns_ms"]), found["breakdowns_ms"])
        self.assertLess(found["intro_energy"], -3)  # drums alone are quieter than the full song


if __name__ == "__main__":
    unittest.main()
