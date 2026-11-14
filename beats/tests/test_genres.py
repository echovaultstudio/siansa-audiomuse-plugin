"""Genres (beats/genres.py) and the style model's sound (beats/style.py): the mel bands, the patches, and which
genres a song is in by its sound."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats import genres, style  # noqa: E402

LABELS = ["Electronic---Electro", "Hip Hop---Electro", "Electronic---Trance", "Electronic---Progressive Trance", "Electronic---Drum n Bass",
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


class GenresTest(unittest.TestCase):
    def setUp(self):
        self.t = genres.Taxonomy(LABELS)

    def test_a_song_is_in_every_genre_its_sound_scores_enough(self):
        ranked = [("Electronic---Progressive Trance", 0.62), ("Electronic---Trance", 0.41), ("Electronic---House", 0.15),
                  ("Electronic---Techno", 0.149), ("Pop---Ballad", 0.02)]
        self.assertEqual(genres.heard(ranked), ranked[:3])
        self.assertEqual(genres.heard(list(reversed(ranked))), ranked[:3])  # in any order
        # Only among its best three (0.36): a faint fourth guess doesn't make a genre.
        many = [(label, 0.3 - i * 0.001) for i, label in enumerate(LABELS)]
        self.assertEqual(len(genres.heard(many)), 3)
        self.assertEqual(len(genres.heard(many, rank=1)), 1)
        self.assertEqual(len(genres.heard(many, rank=len(LABELS))), len(LABELS))

    def test_a_song_is_always_in_its_best_genre(self):
        self.assertEqual(genres.heard([("Pop---Ballad", 0.05), ("Rock---Punk", 0.03)]), [("Pop---Ballad", 0.05)])
        self.assertEqual(genres.heard([]), [])
        self.assertEqual(genres.heard(None), [])

    def test_the_tags_play_no_part(self):
        for name in ("vote", "TAG_STYLE", "TAG_GENRE", "VOTE_MIN", "VOTE_KEEP", "RADIO_MIN", "family"):
            self.assertFalse(hasattr(genres, name), name)

    def test_a_radio_plays_the_most_precise_genre_named(self):
        self.assertEqual(genres.target(self.t, "Electronic / Trance"), ({"Electronic---Trance"}, set()))
        self.assertEqual(genres.target(self.t, "Electronic"), (set(), {"Electronic"}))
        self.assertEqual(genres.target(self.t, "Hip Hop / Electro"), ({"Hip Hop---Electro"}, set()))
        self.assertEqual(genres.target(self.t, "Electro"), ({"Electronic---Electro", "Hip Hop---Electro"}, set()))

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


class PrecisionAndPassesTest(unittest.TestCase):
    """Half precision and several songs in one pass (0.41, the owner, 13 Nov): what's chosen, and that a pass gives
    each song what hearing it alone would, with the CPU fallbacks still in place. No GPU on the runner: the model's
    scoring is stood in for."""

    def model(self, precision="auto"):
        return style.StyleModel("/nowhere", precision=precision)

    def test_precision_asked_for_wins_and_auto_needs_tensor_cores(self):
        self.assertFalse(self.model("fp32")._wants_half("cuda"))
        self.assertTrue(self.model("fp16")._wants_half("cuda:1"))
        self.assertFalse(self.model("auto")._wants_half("cuda"))  # no torch or no card here: full precision, no error
        self.assertFalse(self.model()._wants_half("cuda"))
        self.assertEqual(self.model().precision, None)  # not loaded yet
        self.assertEqual(self.model().songs_at_once, 1)  # the CPU, one at a time
        self.assertGreaterEqual(style.TENSOR_CORES_FROM, 7)  # Volta and on: Turing (7.5), Ampere (8.x)
        self.assertLessEqual(style.SONGS_AT_ONCE_GPU * style.MODELS["maest"].most * style.MODELS["maest"].patch * style.BANDS * 2,
                             50_000_000)  # 16 MAEST windows in half precision: well under a 6 GB card

    def test_a_pass_of_two_songs_scores_each_as_alone(self):
        import numpy as np

        m = self.model()
        m._labels = LABELS
        m.device, m._session = "cpu", object()  # "loaded" on the CPU; its scoring stood in for below
        heard = []

        def scores(session, batch):  # a score per style from the patch's loudness, so each song's differ
            heard.append(len(batch))
            level = batch.reshape(len(batch), -1).mean(axis=1, keepdims=True)
            return np.clip(level * (1 + np.arange(len(LABELS)))[None, :] / 20, 0, 1)

        m._cpu_scores = scores
        m.load = lambda: m._session
        t = np.arange(style.RATE * 4) / style.RATE
        quiet, loud = 0.1 * np.sin(2 * np.pi * 220 * t), 0.9 * np.sin(2 * np.pi * 220 * t)
        together = m.many([(quiet, style.RATE), (loud, style.RATE)])
        alone = [m(quiet), m(loud)]
        self.assertEqual(heard[0], heard[1] + heard[2])  # one run for the pass, then one each
        for a, b in zip(together, alone):
            self.assertEqual([l for l, _ in a[:5]], [l for l, _ in b[:5]])
            np.testing.assert_allclose([v for _, v in a], [v for _, v in b], atol=1e-9)
        self.assertNotEqual(together[0][0][1], together[1][0][1])  # the two songs got their own scores

    def test_half_precision_that_fails_goes_back_to_full(self):
        m = self.model("fp16")
        m.half = True
        m._full_precision("a test")
        self.assertFalse(m.half)
        self.assertIn("full precision", m.problem)

    def test_a_pass_that_does_not_fit_the_gpu_makes_it_one_at_a_time(self):
        m = self.model()
        m._torch = object()  # "on a GPU"
        self.assertEqual(m.songs_at_once, style.SONGS_AT_ONCE_GPU)
        self.assertTrue(m.one_at_a_time("out of memory"))
        self.assertEqual(m.songs_at_once, 1)
        self.assertFalse(m.one_at_a_time("again"))  # said once
        self.assertIn("one song at a time", m.problem)
