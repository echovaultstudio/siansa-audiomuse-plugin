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
        self.assertAlmostEqual(result["first_downbeat_ms"], beats[1] * 1000, delta=12)  # the even grid, not the 20 ms one
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


class VerdictTest(unittest.TestCase):
    def test_control_matches_its_tag(self):
        result = {"bpm": 140.2, "intro_bpm": 140.3, "outro_bpm": 140.1, "steady": 0.98}
        texts = [t for _, t in grid.verdict(result, tag_bpm=140)]
        self.assertEqual(texts, ["Matches the BPM tag"])

    def test_a_tag_at_double(self):
        notes = dict((t, level) for level, t in grid.verdict({"bpm": 93.0, "steady": 0.99}, tag_bpm=186))
        self.assertEqual(notes, {"BPM tag 186 is double of this": "warn"})

    def test_genres_dont_make_notes(self):
        # Tags are partly guessed: a song at 87 is just 87, whatever its genre says.
        self.assertEqual(grid.verdict({"bpm": 87.0, "steady": 0.99}), [])

    def test_a_half_time_intro_is_noted_not_flagged(self):
        result = {"bpm": 174.0, "intro_bpm": 87.0, "intro_steady": 0.95, "outro_bpm": 174.1, "steady": 0.9}
        self.assertEqual(grid.verdict(result), [("info", "Starts at half time (87.0)")])

    def test_an_end_at_another_tempo_is_noted(self):
        result = {"bpm": 174.0, "intro_bpm": 140.0, "intro_steady": 0.95, "outro_bpm": None, "outro_steady": 0.3,
                  "steady": 0.9}
        notes = grid.verdict(result)
        self.assertIn(("info", "Starts at 140.0, not the song's 174.0"), notes)
        self.assertIn(("info", "Ends without a steady beat"), notes)

    def test_no_beat(self):
        self.assertEqual(grid.verdict({"bpm": None}), [("warn", "No steady beat found")])

    def test_unsteady_is_noted(self):
        self.assertIn(("info", "Tempo varies"), grid.verdict({"bpm": 100.0, "steady": 0.4}))


class MainTempoTest(unittest.TestCase):
    def test_a_breakdown_doesnt_drag_the_tempo_off(self):
        # As Control: 140 most of the way, with a slower stretch in the middle; one line through it all read 149.
        beats = beats_at(140, 120) + [120 + b for b in beats_at(100, 40, start=0.0)] + [160 + b for b in beats_at(140, 120, start=0.0)]
        result = grid.analyse(beats, beats[::4], 280.0)
        self.assertAlmostEqual(result["bpm"], 140.0, delta=0.3)
        self.assertGreater(result["main_share"], 0.8)
        self.assertGreater(abs(result["bpm_fit"] - 140.0), 1.0)

    def test_a_half_time_intro_doesnt_drag_the_tempo_off(self):
        beats = beats_at(87, 60) + [60 + b for b in beats_at(174, 240, start=0.0)]
        result = grid.analyse(beats, beats[::4], 300.0)
        self.assertAlmostEqual(result["bpm"], 174.0, delta=0.4)
        self.assertAlmostEqual(result["intro_bpm"], 87.0, delta=0.4)

    def test_a_steady_song_keeps_the_fine_reading(self):
        beats = beats_at(128, 240)
        result = grid.analyse(beats, beats[::4], 240.0)
        self.assertEqual(result["bpm"], result["bpm_fit"])
        self.assertAlmostEqual(result["main_share"], 1.0, places=2)

    def test_too_few_beats(self):
        self.assertEqual(grid.main_tempo([0.5, 1.0, 1.5]), (None, 0.0))


class SteadyGridTest(unittest.TestCase):
    def test_extra_beats_in_a_breakdown_dont_move_the_grid(self):
        # As Control at 57 s: the model hears beats between the beats in a breakdown.
        beats = beats_at(140, 200)
        noisy = beats + [b + 0.21 for b in beats if 60 < b < 75]
        steady = grid.steady_grid(noisy, beats[::4], 140.0)
        self.assertAlmostEqual(steady["bpm"], 140.0, delta=0.05)
        self.assertGreater(steady["on_grid"], 0.8)
        on = [b for b in steady["beats"] if 60 < b < 75]
        self.assertTrue(all(abs(b - min(beats, key=lambda x: abs(x - b))) < 0.015 for b in on))

    def test_bars_follow_the_vote_not_the_wandering_downbeats(self):
        # As 1111: most downbeats on beat 1 of 4, but some on beat 3 and some on any beat in a breakdown.
        beats = beats_at(174, 240)
        downs = beats[::4]
        wandering = downs[:20] + beats[82:140:2] + downs[35:]
        steady = grid.steady_grid(beats, wandering, 174.0)
        gaps = {round((b - a) * 174 / 60) for a, b in zip(steady["downbeats"], steady["downbeats"][1:])}
        self.assertEqual(gaps, {4})
        self.assertAlmostEqual(steady["downbeats"][0], downs[0], delta=0.015)
        self.assertLess(steady["bar_share"], 1.0)

    def test_the_grid_carries_on_through_a_beatless_intro(self):
        beats = [12.0 + b for b in beats_at(128, 180, start=0.0)]
        steady = grid.steady_grid(beats, beats[::4], 128.0)
        self.assertAlmostEqual(steady["beats"][0], beats[0], delta=0.015)

    def test_a_tempo_reading_a_fraction_off_still_gets_the_grid(self):
        # As Quema Quema (174.44) and Control (140.35): the song's tempo reading a little off its real one, which
        # left a 5-minute song without a grid in 0.5.0.
        rng = random.Random(4)
        for true, reading in ((174.0, 174.44), (140.0, 140.35), (175.0, 175.6)):
            period = 60.0 / true
            beats = sorted(round((i * period + rng.uniform(-0.01, 0.01)) / 0.02) * 0.02
                           for i in range(int(300 / period)) if rng.random() > 0.03)
            beats = sorted(beats + [b + period / 2 for b in beats if 40 < b < 60])  # a breakdown's extra beats
            steady = grid.steady_grid(beats, beats[::4], reading)
            self.assertIsNotNone(steady, reading)
            self.assertAlmostEqual(steady["bpm"], true, delta=0.02)
            self.assertGreater(steady["on_grid"], 0.9)

    def test_no_grid_for_a_song_that_changes_tempo_halfway(self):
        beats = beats_at(120, 150) + [150 + b for b in beats_at(150, 150, start=0.0)]
        self.assertIsNone(grid.steady_grid(beats, beats[::4], 120.0))

    def test_no_grid_for_a_song_that_changes_tempo(self):
        beats = beats_at(120, 100) + [100 + b for b in beats_at(150, 120, start=0.0)]
        self.assertIsNone(grid.steady_grid(beats, beats[::4], 120.0))

    def test_an_end_without_a_steady_beat_has_no_tempo(self):
        rng = random.Random(3)
        scattered = sorted(rng.uniform(0, 30) for _ in range(60))  # as 1111's outro, read at 365
        self.assertIsNone(grid.end_tempo(scattered, 0, 30)[0])
        self.assertAlmostEqual(grid.end_tempo(beats_at(174, 30), 0, 30)[0], 174.0, delta=0.5)

    def test_analyse_gives_the_grids_bars(self):
        beats = beats_at(140, 200)
        result = grid.analyse(beats + [b + 0.21 for b in beats if 60 < b < 75], beats[2::4], 200.0)
        self.assertEqual(result["beats_per_bar"], 4)
        self.assertAlmostEqual(result["first_downbeat_ms"], beats[2] * 1000, delta=15)
        self.assertAlmostEqual(result["grid_bpm"], 140.0, delta=0.05)


class BeatlessOutroTest(unittest.TestCase):
    def test_the_beat_stopping_is_sent_only_before_a_beatless_outro(self):
        beats = beats_at(128, 200)
        steady = grid.steady_grid(beats, beats[::4], 128.0)
        # The beat to the end, but for the last few seconds of a fade-out: not a beatless outro.
        self.assertIsNone(grid.app_ends(steady, beats, beats[-1] + 8.0)["beats_until_ms"])
        # 30 seconds of pads after the last beat: one.
        ends = grid.app_ends(steady, beats, beats[-1] + 30.0)
        self.assertAlmostEqual(ends["beats_until_ms"], beats[-1] * 1000, delta=1)


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
