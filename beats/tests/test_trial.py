"""The energy trial (beats/trial.py): the candidates worked out, kept and listed. The real models are tried when the
trial image is built (Dockerfile.trial); here they are stand-ins."""

import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beats import trial as trial_module  # noqa: E402
from beats.store import Store  # noqa: E402
from beats.trial import COLUMNS, HEADS, Trial  # noqa: E402


class Input:
    name = "in"


class Head:
    """A model that answers each row with fixed numbers plus the row's mean: so a patch's answer differs from the
    mean vector's only when the model isn't a straight line, as with the real ones."""

    def __init__(self, outputs):
        self.outputs = outputs

    def get_inputs(self):
        return [Input()]

    def run(self, _, feed):
        rows = feed["in"]
        return [np.array([[i + 1 + float(np.mean(r)) ** 2 for i in range(self.outputs)] for r in rows], dtype=np.float32)]


class Embedder(Head):
    def __init__(self, width):
        self.width = width

    def run(self, _, feed):
        rows = feed["in"]
        return [np.array([[float(np.mean(r))] * self.width for r in rows], dtype=np.float32)]


def with_stand_ins(trial):
    sessions = {name: Head(len(HEADS[name][2])) for name in HEADS}
    sessions["musicnn"], sessions["effnet"], sessions["effnet_output"] = Embedder(200), Embedder(1280), "embedding"
    trial._sessions = sessions
    return trial


class TrialTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = Store(os.path.join(self.folder.name, "beats.db"))

    def test_only_the_trial_image_has_it(self):
        self.assertFalse(Trial(self.folder.name, self.store).available)
        os.makedirs(os.path.join(self.folder.name, "trial"))
        for name in HEADS:
            open(os.path.join(self.folder.name, "trial", HEADS[name][0]), "w").close()
        self.assertTrue(Trial(self.folder.name, self.store).available)

    def test_the_page_is_whole_and_names_every_column(self):
        for column in COLUMNS:
            self.assertIn(f'"{column}"', trial_module.PAGE)
        self.assertIn("--w:${w}%", trial_module.PAGE)
        self.assertNotIn("%s", trial_module.PAGE)


@unittest.skipUnless(importlib.util.find_spec("librosa"), "librosa isn't installed here")
class MeasuredTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = Store(os.path.join(self.folder.name, "beats.db"))

    def song(self, loud, name):
        import soundfile

        rate = 16_000
        t = np.arange(rate * 12) / rate
        beat = (np.sin(2 * np.pi * 2 * t) > 0.95).astype(np.float32)  # two clicks a second
        audio = loud * (0.6 * np.sin(2 * np.pi * 60 * t) + 0.4 * beat * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        path = os.path.join(self.folder.name, name)
        soundfile.write(path, audio, rate)
        return path

    def test_the_ingredients_tell_a_loud_song_from_a_quiet_one_on_both_scales(self):
        loud = trial_module.ingredients(trial_module.hear.load(self.song(0.9, "loud.wav"))[0])
        quiet = trial_module.ingredients(trial_module.hear.load(self.song(0.09, "quiet.wav"))[0])
        self.assertGreater(loud["energy_now"], quiet["energy_now"])
        self.assertGreater(loud["energy_old"], quiet["energy_old"] * 5)  # the old one is the plain level: ten times
        self.assertLess(loud["energy_now"], quiet["energy_now"] * 2)  # today's is in dB: a step, not a multiple
        self.assertGreater(loud["bass"], 0.5)  # mostly the 60 Hz tone
        self.assertTrue(1.0 < loud["onsets"] < 4.0, loud["onsets"])
        self.assertAlmostEqual(loud["bass"], quiet["bass"], places=3)  # what it's made of doesn't change with the level

    def test_every_candidate_is_worked_out_kept_and_listed(self):
        trial = with_stand_ins(Trial(self.folder.name, self.store))
        found = trial.measure(self.song(0.5, "song.wav"))
        self.assertEqual(sorted(found), sorted(COLUMNS))
        # From every patch and from the one mean vector: both there, to compare.
        self.assertIn("deam_arousal_stored", found)
        with self.store.lock:
            self.store.db.execute("INSERT INTO songs (id, title, artist, album) VALUES ('s1', 'Song', 'Artist', 'Album')")
            self.store.db.commit()
        trial.keep("s1", found)
        trial.keep("s2", error="couldn't be read")  # a song that failed isn't listed
        rows = trial.rows()
        self.assertEqual([(r["id"], r["title"]) for r in rows], [("s1", "Song")])
        self.assertAlmostEqual(rows[0]["energy_now"], found["energy_now"])
        lines = trial.csv().splitlines()
        self.assertEqual(lines[0].split(",")[:5], ["artist", "album", "title", "tempo", "lufs"])
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith("Artist,Album,Song,"))


if __name__ == "__main__":
    unittest.main()
