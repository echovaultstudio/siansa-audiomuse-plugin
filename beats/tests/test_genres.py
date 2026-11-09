"""Genres by vote (beats/genres.py) and the style model's sound (beats/style.py): the mel bands, the patches, and the
vote of the sound with the tags."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats import genres, style  # noqa: E402

LABELS = ["Electronic---Trance", "Electronic---Progressive Trance", "Electronic---Drum n Bass",
          "Electronic---Deep House", "Electronic---House", "Electronic---Techno", "Rock---Punk", "Rock---Indie Rock",
          "Hip Hop---Boom Bap", "Funk / Soul---Soul", "Folk, World, & Country---Country", "Pop---Ballad"]


class TaxonomyTest(unittest.TestCase):
    def setUp(self):
        self.t = genres.Taxonomy(LABELS)

    def test_tags_are_matched_however_they_are_written(self):
        for tag in ("Drum And Bass", "drum & bass", "DnB", "Drum'n'Bass", "Drum n Bass"):
            self.assertEqual(self.t.match(tag), ({"Electronic---Drum n Bass"}, set()), tag)
        self.assertEqual(self.t.match("Electronic / Trance"), ({"Electronic---Trance"}, {"Electronic"}))
        self.assertEqual(self.t.match("Hip-Hop"), (set(), {"Hip Hop"}))
        self.assertEqual(self.t.match("Soul"), ({"Funk / Soul---Soul"}, set()))
        self.assertEqual(self.t.match("Country"), ({"Folk, World, & Country---Country"}, set()))
        self.assertEqual(self.t.match("Alternative Rock / Rock"), (set(), {"Rock"}))
        self.assertEqual(self.t.match("Opera"), (set(), set()))


class VoteTest(unittest.TestCase):
    def setUp(self):
        self.t = genres.Taxonomy(LABELS)

    def test_the_sound_leads_and_tags_confirm(self):
        sound = {"Electronic---Progressive Trance": 0.62, "Electronic---Trance": 0.41, "Electronic---House": 0.08}
        voted = genres.vote(self.t, sound, ["Electronic / Trance"])
        self.assertEqual(voted[0][0], "Electronic---Trance")  # 0.41 + 0.35 + 0.10
        self.assertEqual(voted[1][0], "Electronic---Progressive Trance")  # 0.62 + 0.10
        self.assertAlmostEqual(dict(voted)["Electronic---House"], 0.18)  # 0.08 + 0.10: kept, low
        self.assertEqual(genres.family(voted), "Electronic")

    def test_a_wrong_tag_loses_to_the_sound(self):
        # Tagged Rock by its album, but every patch hears drum and bass.
        voted = genres.vote(self.t, {"Electronic---Drum n Bass": 0.81}, ["Punk"])
        self.assertEqual(voted[0][0], "Electronic---Drum n Bass")
        self.assertEqual(dict(voted)["Rock---Punk"], 0.35)

    def test_without_sound_the_tags_vote(self):
        self.assertEqual(genres.vote(self.t, {}, ["Deep House"]), [("Electronic---Deep House", 0.35)])
        self.assertEqual(genres.vote(self.t, {}, ["Opera"]), [])
        self.assertEqual(genres.vote(self.t, None, None), [])

    def test_a_song_keeps_a_few_genres_and_always_its_best(self):
        sound = {label: 0.3 for label in LABELS}
        self.assertEqual(len(genres.vote(self.t, sound, [])), genres.VOTE_KEEP)
        self.assertEqual(genres.vote(self.t, {"Pop---Ballad": 0.05}, []), [("Pop---Ballad", 0.05)])

    def test_a_radio_plays_the_most_precise_genre_named(self):
        self.assertEqual(genres.target(self.t, "Electronic / Trance"), ({"Electronic---Trance"}, set()))
        self.assertEqual(genres.target(self.t, "Electronic"), (set(), {"Electronic"}))

    def test_agreement_with_the_tags(self):
        rows = [
            ([("Electronic---Trance", 0.7), ("Electronic---House", 0.2)], self.t.match("Trance")),  # both agree
            ([("Electronic---Techno", 0.7), ("Electronic---Trance", 0.2)], self.t.match("Trance")),  # in its top 3
            ([("Rock---Punk", 0.7)], self.t.match("Trance")),  # neither
            ([("Rock---Punk", 0.7)], self.t.match("Opera")),  # tags name nothing: not counted
        ]
        self.assertEqual(genres.agreement(rows), {"songs": 3, "style": round(2 / 3, 3), "family": round(2 / 3, 3)})


try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class MelTest(unittest.TestCase):
    """The model's input, as Essentia's TensorflowInputMusiCNN (checked against Essentia when it's installed)."""

    def test_shape_and_silence(self):
        import numpy as np

        bands = style.mel(np.zeros(style.RATE * 3, dtype=np.float32))
        self.assertEqual(bands.shape, (1 + (style.RATE * 3 - 512) // 256, 96))
        self.assertEqual(float(bands.max()), 0.0)  # log10(1 + 0)

    def test_a_tone_lands_in_its_band(self):
        import numpy as np

        t = np.arange(style.RATE * 2) / style.RATE
        low = style.mel(0.5 * np.sin(2 * np.pi * 110 * t)).mean(axis=0)
        high = style.mel(0.5 * np.sin(2 * np.pi * 4000 * t)).mean(axis=0)
        self.assertLess(int(np.argmax(low)), 12)
        self.assertGreater(int(np.argmax(high)), 70)

    def test_patches_are_spread_through_a_long_song(self):
        import numpy as np

        self.assertEqual(style.patches(np.zeros((300, 96), np.float32)).shape, (3, 128, 96))
        self.assertEqual(style.patches(np.zeros((50, 96), np.float32)).shape, (1, 128, 96))  # padded
        bands = np.arange(20_000, dtype=np.float32)[:, None].repeat(96, 1)
        spread = style.patches(bands)
        self.assertEqual(spread.shape[0], style.MAX_PATCHES)
        self.assertEqual(float(spread[0, 0, 0]), 0.0)  # the first patch at the start…
        self.assertGreater(float(spread[-1, -1, 0]), 19_000)  # …and the last at the end

    def test_as_essentia(self):
        try:
            import essentia.standard as es
        except ImportError:
            self.skipTest("Essentia isn't installed")
        import numpy as np

        rng = np.random.RandomState(3)
        t = np.arange(style.RATE * 2) / style.RATE
        x = (0.3 * np.sin(2 * np.pi * 330 * t) + 0.05 * rng.randn(t.size)).astype(np.float32)
        ours = style.mel(x)
        algo = es.TensorflowInputMusiCNN()
        theirs = np.array([algo(x[i * 256:i * 256 + 512]) for i in range(len(ours))])
        self.assertLess(float(np.abs(ours - theirs).max()), 0.001)


if __name__ == "__main__":
    unittest.main()
