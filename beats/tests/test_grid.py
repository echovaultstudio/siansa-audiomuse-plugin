import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats import grid  # noqa: E402


def beats_at(bpm, seconds, start=0.5, jitter=0.0, quantum=0.02, seed=1):
    """Beats as Beat This! gives them: on a 20 ms grid, optionally a little off."""
    rng = random.Random(seed)
    period, out, t = 60.0 / bpm, [], start
    while t < seconds:
        v = t + rng.uniform(-jitter, jitter)
        out.append(round(v / quantum) * quantum if quantum else v)
        t += period
    return out


class TempoTest(unittest.TestCase):
    def test_a_steady_beat_reads_to_a_fraction_of_a_bpm_despite_the_20_ms_grid(self):
        for bpm in (87.0, 122.0, 128.0, 140.0, 174.0):
            t = grid.tempo(beats_at(bpm, 180))
            self.assertAlmostEqual(t["bpm"], bpm, delta=0.05, msg=bpm)
            self.assertGreater(t["steady"], 0.95)
            self.assertGreater(t["grid_share"], 0.95)

    def test_a_missed_beat_doesnt_bend_the_tempo(self):
        beats = beats_at(140, 120)
        del beats[40], beats[80]
        self.assertAlmostEqual(grid.tempo(beats)["bpm"], 140.0, delta=0.05)

    def test_a_live_drummer_reads_as_less_steady(self):
        t = grid.tempo(beats_at(100, 120, jitter=0.035, quantum=0))
        self.assertAlmostEqual(t["bpm"], 100.0, delta=0.5)
        self.assertLess(t["steady"], 0.9)
        self.assertLess(t["grid_share"], 0.95)

    def test_too_few_beats_give_no_tempo(self):
        self.assertIsNone(grid.tempo([1.0, 1.5, 2.0]))
        self.assertIsNone(grid.tempo([]))

    def test_each_end_is_read_on_its_own(self):
        beats = beats_at(120, 60) + [b + 60.0 for b in beats_at(140, 60, start=0.0)]
        result = grid.analyse(beats, beats[::4], 120.0)
        self.assertAlmostEqual(result["intro_bpm"], 120.0, delta=0.1)
        self.assertAlmostEqual(result["outro_bpm"], 140.0, delta=0.1)


class BarsTest(unittest.TestCase):
    def test_four_and_three_beats_a_bar(self):
        beats = beats_at(128, 60)
        self.assertEqual(grid.bars(beats, beats[::4]), (4, 1.0))
        waltz = beats_at(96, 60)
        per_bar, share = grid.bars(waltz, waltz[::3])
        self.assertEqual(per_bar, 3)
        self.assertGreater(share, 0.9)

    def test_too_few_downbeats(self):
        self.assertEqual(grid.bars(beats_at(120, 10), [0.5]), (None, 0.0))

    def test_analyse_reports_the_first_downbeat_and_counts(self):
        beats = beats_at(140, 60, start=1.0)
        result = grid.analyse(beats, beats[1::4], 60.0)
        self.assertEqual(result["beats_per_bar"], 4)
        self.assertAlmostEqual(result["first_downbeat_ms"], beats[1] * 1000, delta=1)
        self.assertEqual(result["beat_count"], len(beats))


class RelationTest(unittest.TestCase):
    def test_controls_misreading_is_four_thirds(self):
        # Siansa's log: the plugin read Control at 187.1 BPM, a trance track at about 140.
        self.assertEqual(grid.relation(187.1, 140.3), "4/3")
        self.assertEqual(grid.relation(140.3, 187.1), "3/4")

    def test_octaves_and_thirds(self):
        self.assertEqual(grid.relation(140, 70), "double")
        self.assertEqual(grid.relation(87, 174), "half")
        self.assertEqual(grid.relation(180, 120), "3/2")
        self.assertEqual(grid.relation(80, 120), "2/3")
        self.assertEqual(grid.relation(140.5, 140.0), "same")
        self.assertEqual(grid.relation(150, 128), "different")
        self.assertIsNone(grid.relation(None, 120))
        self.assertIsNone(grid.relation(0, 120))


class GenreTest(unittest.TestCase):
    def test_genres_find_their_range(self):
        self.assertEqual(grid.genre_range(["Trance"])[:2], (125, 150))
        self.assertEqual(grid.genre_range(["Progressive Trance"])[:2], (125, 150))
        self.assertEqual(grid.genre_range(["Drum & Bass"])[:2], (160, 180))
        self.assertEqual(grid.genre_range(["Electronic/Deep House"])[:2], (115, 132))
        self.assertEqual(grid.genre_range(["Hip-Hop"])[:2], (70, 105))
        self.assertIsNone(grid.genre_range(["Spoken word"]))
        self.assertIsNone(grid.genre_range([]))

    def test_the_first_known_genre_wins(self):
        self.assertEqual(grid.genre_range(["Spoken word", "Techno"])[2], "Techno")

    def test_words_not_parts_of_words(self):
        self.assertIsNone(grid.genre_range(["Trapeze"]))
        self.assertIsNone(grid.genre_range(["Popcorn"]))


class VerdictTest(unittest.TestCase):
    def test_control_shows_the_plugins_misreading(self):
        result = {"bpm": 140.2, "intro_bpm": 140.3, "outro_bpm": 140.1, "steady": 0.98}
        notes = grid.verdict(result, tag_bpm=140, plugin={"intro_bpm": 187.1, "outro_bpm": 140.0}, genres=["Trance"])
        texts = [t for _, t in notes]
        self.assertIn("Matches the BPM tag", texts)
        self.assertIn("Plugin's intro 187.1 is 4/3 of this", texts)
        self.assertIn("Plugin's outro agrees", texts)
        self.assertFalse(any("outside" in t for t in texts))

    def test_a_tag_at_half_and_a_tempo_outside_the_genre(self):
        notes = dict((t, level) for level, t in grid.verdict({"bpm": 93.0, "steady": 0.99}, tag_bpm=186, genres=["Trance"]))
        self.assertEqual(notes["BPM tag 186 is double of this"], "warn")
        self.assertEqual(notes["93.0 is outside Trance's usual 125–150"], "warn")

    def test_no_beat(self):
        self.assertEqual(grid.verdict({"bpm": None}), [("warn", "No steady beat found")])

    def test_unsteady_is_noted(self):
        self.assertIn(("info", "Tempo varies"), grid.verdict({"bpm": 100.0, "steady": 0.4}))


class CurveTest(unittest.TestCase):
    def test_the_curve_follows_a_tempo_change_and_stays_small(self):
        beats = beats_at(120, 300) + [b + 300.0 for b in beats_at(140, 300, start=0.0)]
        curve = grid.tempo_curve(beats, points=200)
        self.assertLessEqual(len(curve), 200)
        self.assertAlmostEqual(curve[5][1], 120.0, delta=1.5)
        self.assertAlmostEqual(curve[-5][1], 140.0, delta=1.5)
        self.assertEqual(grid.tempo_curve([1.0, 2.0]), [])


if __name__ == "__main__":
    unittest.main()
