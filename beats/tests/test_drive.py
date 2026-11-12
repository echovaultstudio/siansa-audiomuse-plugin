"""How energetic a song is (0.35, beats/drive.py): scored from the sound kept, and old energies on today's scale. The
real model is tried when the image is built (Dockerfile); here it is a stand-in."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beats import drive  # noqa: E402
from beats.drive import Arousal  # noqa: E402
from beats.store import Store  # noqa: E402


def vector(value):
    return np.full(200, value, dtype=np.float32).tobytes()


class Scored(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = Store(os.path.join(self.folder.name, "beats.db"))
        self.arousal = Arousal(self.folder.name, self.store)
        open(self.arousal.path, "w").close()  # "the model is there"
        self.asked = []

        def score(vectors):  # a stand-in: a song's arousal is its vector's first number, times ten
            self.asked.append(len(vectors))
            return np.asarray(vectors)[:, 0] * 10

        self.arousal.score = score

    def sound(self, song_id, value, source="audiomuse", energy=0.3):
        self.store.save_sound([{"song_id": song_id, "musicnn": vector(value), "energy": energy}], source)

    def test_nothing_without_the_model(self):
        os.remove(self.arousal.path)
        self.sound("a", 0.5)
        self.assertEqual(self.arousal.catch_up(), 0)
        self.assertEqual(self.arousal.of(["a"]), {})

    def test_every_song_with_a_sound_is_scored_once(self):
        self.sound("a", 0.5)
        self.sound("b", 0.7)
        self.store.sound_failed("c", "couldn't be read")  # no vector: left alone
        self.assertEqual(self.arousal.catch_up(), 2)
        self.assertAlmostEqual(self.arousal.of(["a", "b", "c"])["a"], 5.0, places=4)
        self.assertAlmostEqual(self.arousal.of(["b"])["b"], 7.0, places=4)
        self.assertNotIn("c", self.arousal.of(["c"]))
        self.assertEqual(self.arousal.catch_up(), 0)  # nothing new: the model isn't asked
        self.assertEqual(self.asked, [2])

    def test_a_song_heard_again_is_scored_again(self):
        self.sound("a", 0.5)
        self.arousal.catch_up()
        import time

        time.sleep(0.01)
        self.sound("a", 0.6, source="beats")
        self.assertEqual(self.arousal.catch_up(), 1)
        self.assertAlmostEqual(self.arousal.of(["a"])["a"], 6.0, places=4)
        self.assertEqual(self.arousal.count(), 1)

    def test_a_big_library_goes_in_batches(self):
        self.store.save_sound([{"song_id": f"s{i}", "musicnn": vector(0.1)} for i in range(drive.BATCH + 5)])
        self.assertEqual(self.arousal.catch_up(), drive.BATCH + 5)
        self.assertEqual(self.asked, [drive.BATCH, 5])


class OldEnergies(unittest.TestCase):
    def test_a_library_is_on_the_old_scale_or_todays_by_its_middle(self):
        self.assertTrue(drive.old_scale([0.17, 0.27, 0.29, 0.31, 0.33]))  # the owner's taken-over songs
        self.assertFalse(drive.old_scale([0.61, 0.71, 0.74, 0.78, 0.84]))  # heard today's way
        self.assertFalse(drive.old_scale([]))
        self.assertFalse(drive.old_scale([None, 0]))

    def test_an_old_energy_lands_where_todays_measure_put_the_same_song(self):
        # From the trial (12 Nov), old and today's measure of the same songs.
        for old, today in ((0.3682, 0.84), (0.2197, 0.7136), (0.0998, 0.5977), (0.4890, 0.879), (0.2215, 0.7518)):
            self.assertAlmostEqual(drive.converted(old), today, delta=0.04)
        self.assertIsNone(drive.converted(None))
        self.assertEqual(drive.converted(0), 0)

    def test_the_store_says_which_scale_and_looks_again_when_the_sound_changes(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        store = Store(os.path.join(folder.name, "beats.db"))
        self.assertFalse(store.energy_old_scale())
        store.save_sound([{"song_id": f"s{i}", "musicnn": vector(0.1), "energy": 0.2 + i / 100} for i in range(5)])
        self.assertTrue(store.energy_old_scale())
        store.save_sound([{"song_id": "heard", "musicnn": vector(0.1), "energy": 0.8}], "beats")
        self.assertTrue(store.energy_old_scale())  # a song heard here doesn't count


if __name__ == "__main__":
    unittest.main()
